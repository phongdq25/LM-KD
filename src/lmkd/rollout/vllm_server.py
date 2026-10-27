"""Rollout on a dedicated GPU via vLLM server with NCCL weight synchronization (T26).

Server launch command (outside framework):
    CUDA_VISIBLE_DEVICES=7 trl vllm-serve --model <student path> --tensor-parallel-size 1 --port 8000
"""
from __future__ import annotations

import inspect
from typing import Any

from lmkd.compat.trl_api import get_vllm_client
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ROLLOUT
from lmkd.rollout.base import Rollout, RolloutEngine, SamplingSpec, ScoreSpec, eos_ids
from lmkd.rollout.weight_sync import iter_student_weights


@ROLLOUT.register("vllm_server", extras=("trl", "vllm"))
class VLLMServer(RolloutEngine):
    """Rollout engine in a separate server process with NCCL weight synchronization (T26)."""

    class Config(BaseConfig):
        host: str = "127.0.0.1"
        port: int = 8000
        group_port: int = 51216
        sync_every: int = 1
        timeout_s: float = 600.0
        extra_eos_ids: list[int] = []

    def setup(self, rt: Any) -> None:
        self.rt = rt
        self._tok = rt.student_tokenizer
        self._last_sync_step = -1
        self.client = None

        self._main_call(self._connect)

    def _connect(self):
        cfg, rt = self.cfg, self.rt
        if rt.engine.is_main:
            self.client = get_vllm_client(
                host=cfg.host,
                server_port=cfg.port,
                group_port=cfg.group_port,
                connection_timeout=cfg.timeout_s,
            )
            assert self.client is not None
            self.client.init_communicator(device=rt.engine.device)

    def _main_call(self, fn):
        """Broadcast rank-zero results and failures before the next collective."""
        eng = self.rt.engine
        value, error = None, None
        if eng.is_main:
            try:
                value = fn()
            except Exception as exc:
                if getattr(eng, "world_size", 1) == 1:
                    raise
                error = f"{type(exc).__name__}: {exc}"
        if getattr(eng, "world_size", 1) > 1:
            value, error = eng.all_gather_object((value, error))[0]
        if error:
            raise RuntimeError(f"vllm_server rank0: {error}")
        return value

    def _maybe_sync(self) -> None:
        step = self.rt.state.step
        if self._last_sync_step >= 0 and (
            step == self._last_sync_step or (step - self._last_sync_step) < self.cfg.sync_every
        ):
            return

        spec = self.rt.config.models.student
        mode = "merge" if spec.lora is not None else "full"
        engine = self.rt.engine
        model = self.rt.student_model

        error = None
        if getattr(engine, "is_partitioned", False):
            for name, tensor in engine.iter_full_state_dict(model):
                if engine.is_main and self.client is not None and error is None:
                    try:
                        self.client.update_named_param(name, tensor)
                    except Exception as exc:
                        # Drain the collective weight iterator even after a failed network transfer.
                        error = f"{type(exc).__name__}: {exc}"
        elif engine.is_main and self.client is not None:
            try:
                for name, tensor in iter_student_weights(self.rt, mode):
                    self.client.update_named_param(name, tensor)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        if engine.is_main and self.client is not None and error is None:
            try:
                self.client.reset_prefix_cache()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if getattr(engine, "world_size", 1) > 1:
            errors = engine.all_gather_object(error)
            error = next((e for e in errors if e is not None), None)
        if error:
            raise RuntimeError(f"vllm_server weight sync failed: {error}")
        if hasattr(engine, "barrier"):
            engine.barrier()

        self._last_sync_step = step

    def _call_client_generate(
        self, flat_prompts: list[list[int]], spec: SamplingSpec
    ) -> list[list[Rollout]]:
        assert self.client is not None
        sig = inspect.signature(self.client.generate)
        if "prompts" not in sig.parameters:
            raise NotImplementedError("needs token-id API, TRL>=1.16")

        gen_kwargs: dict[str, Any] = {
            "n": spec.n,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "top_k": spec.top_k if spec.top_k > 0 else 0,
            "max_tokens": spec.max_new_tokens,
            "repetition_penalty": spec.repetition_penalty,
            "logprobs": 0 if spec.return_logprobs else None,
        }
        extra: dict[str, Any] = {}
        if spec.stop:
            extra["stop"] = spec.stop
        stop_ids = list(dict.fromkeys([*self.cfg.extra_eos_ids, *spec.stop_token_ids]))
        if stop_ids:
            extra["stop_token_ids"] = stop_ids
        if spec.seed is not None:
            extra["seed"] = spec.seed
        if extra:
            gen_kwargs["generation_kwargs"] = extra

        res = self.client.generate(flat_prompts, **gen_kwargs)
        return self._payload_rollouts(flat_prompts, spec, res)

    def _payload_rollouts(self, prompts, spec, res):
        count = len(prompts) * spec.n
        completions = res.get("completion_ids", [])
        if len(completions) != count:
            raise ValueError(f"vllm_server expected {count} completions, got {len(completions)}")
        logprobs, token_ids = res.get("logprobs"), res.get("logprob_token_ids")
        reasons = res.get("finish_reasons")
        for values in (logprobs, token_ids, reasons):
            if values is not None and len(values) != count:
                raise ValueError("vllm_server completion metadata count mismatch")
        all_rollouts: list[list[Rollout]] = []
        for i, prompt in enumerate(prompts):
            p_rollouts: list[Rollout] = []
            for j in range(spec.n):
                flat_idx = i * spec.n + j
                ids = list(completions[flat_idx])
                lp = self._sampled_logp(ids, None if logprobs is None else logprobs[flat_idx],
                                        None if token_ids is None else token_ids[flat_idx])
                text = self._tok.decode(ids, skip_special_tokens=True)
                # Pinned TRL discards finish_reason; infer conservatively from tokens/length.
                reason = reasons[flat_idx] if reasons is not None else None
                if reason is None:
                    stopped = (bool(ids) and ids[-1] in eos_ids(self._tok, [*self.cfg.extra_eos_ids, *spec.stop_token_ids]))
                    stopped |= bool(spec.stop and any(text.endswith(s) for s in spec.stop))
                    reason = "stop" if stopped or len(ids) < spec.max_new_tokens else "length"
                if reason not in ("stop", "length"):
                    raise ValueError(f"unsupported vllm finish_reason {reason!r}")
                p_rollouts.append(Rollout(list(prompt), ids, lp, reason, text))
            all_rollouts.append(p_rollouts)
        return all_rollouts

    @staticmethod
    def _sampled_logp(ids, values, candidates):
        if values is None:
            return None
        if len(values) != len(ids) or (candidates is not None and len(candidates) != len(ids)):
            raise ValueError("vllm_server logprobs must match completion token count")
        result = []
        for i, (token, lp) in enumerate(zip(ids, values)):
            if isinstance(lp, (list, tuple)):
                index = candidates[i].index(token) if candidates is not None else 0
                if not lp or index >= len(lp):
                    raise ValueError("missing sampled-token logprob")
                lp = lp[index]
            result.append(float(lp))
        return result

    def _run_distributed_batch(self, items: list[Any], call_fn: Any) -> list[Any]:
        """Gather items across all ranks, execute on rank 0, and scatter results back."""
        engine = self.rt.engine
        world_size = getattr(engine, "world_size", 1)

        if world_size > 1:
            all_items_per_rank = engine.all_gather_object(items)
            flat_items = [x for rank_items in all_items_per_rank for x in rank_items]
        else:
            all_items_per_rank = [items]
            flat_items = items

        all_results = self._main_call(lambda: call_fn(flat_items))

        if world_size > 1:
            rank = getattr(engine, "rank", 0)
            offset = sum(len(x) for x in all_items_per_rank[:rank])
            return all_results[offset : offset + len(items)]

        return all_results

    def generate(
        self, prompts: list[list[int]], spec: SamplingSpec, lora: Any = None
    ) -> list[list[Rollout]]:
        self._maybe_sync()
        return self._run_distributed_batch(
            prompts, lambda flat_prompts: self._call_client_generate(flat_prompts, spec)
        )

    def score(self, seqs: list[list[int]], spec: ScoreSpec) -> list[Any]:
        self._maybe_sync()

        def _call_score(flat_seqs: list[list[int]]) -> list[Any]:
            if self.client is None:
                return [[] for _ in flat_seqs]
            if hasattr(self.client, "score"):
                return self.client.score(flat_seqs, top_k=spec.top_k)
            res = self.client.generate(
                prompts=flat_seqs,
                max_tokens=1,
                temperature=0.0,
                prompt_logprobs=spec.top_k,
            )
            if isinstance(res, dict) and "prompt_logprobs" in res:
                return res["prompt_logprobs"]
            return [[] for _ in flat_seqs]

        return self._run_distributed_batch(seqs, _call_score)

    def reset_prefix_cache(self) -> None:
        if self.client is not None:
            self.client.reset_prefix_cache()
