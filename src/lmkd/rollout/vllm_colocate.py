"""vLLM rollout engine sharing the trainer's GPU (sleep/wake) with in-process weight sync (T08)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from lmkd.compat import vllm_api
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ROLLOUT
from lmkd.rollout.base import Rollout, RolloutEngine, SamplingSpec, ScoreSpec, eos_ids
from lmkd.rollout.weight_sync import iter_student_weights, save_lora_adapter


def make_llm(path: str, tokenizer: str | None = None, *, world_size: int = 1, rank: int = 0, seed: int = 0,
             tensor_parallel_size: int = 1, **kw):
    """Shared ``vllm.LLM`` constructor (rollout engine and ``vllm_scorer`` teacher)."""
    vllm = vllm_api.import_vllm()
    if tensor_parallel_size > 1:     # weight sync ships tensors to the TP workers with a pickled callable
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    if world_size > 1:
        kw["distributed_executor_backend"] = "external_launcher"
        seed = seed + rank // tensor_parallel_size
    return vllm.LLM(model=path, tokenizer=tokenizer or path, tensor_parallel_size=tensor_parallel_size,
                    seed=seed, **kw)


def load_weights(llm, weights, chunk_bytes: int = 1 << 28) -> None:
    """Push ``(name, tensor)`` pairs into vLLM. In-process model: direct ``load_weights``; otherwise
    (tensor parallel, workers in other processes) stream CPU chunks to every worker via ``collective_rpc``."""
    try:
        model = vllm_api.in_process_model(llm)
    except (AttributeError, RuntimeError):
        model = None
    if model is not None:
        model.load_weights(weights)
        return
    batch, size = [], 0
    for name, t in weights:
        t = t.detach().to("cpu")
        batch.append((name, t))
        size += t.numel() * t.element_size()
        if size >= chunk_bytes:
            llm.collective_rpc(vllm_api.worker_load_weights, args=(batch,))
            batch, size = [], 0
    if batch:
        llm.collective_rpc(vllm_api.worker_load_weights, args=(batch,))


def score_with_llm(llm, seqs: list[list[int]], spec: ScoreSpec) -> list[list[dict[int, float]]]:
    """``prompt_logprobs`` top-k per position. Entry t is the distribution OF token t (target-aligned);
    entry 0 is ``{}``. The actual token is always included by vLLM."""
    sp = vllm_api.sampling_params(max_tokens=1, temperature=0.0, prompt_logprobs=spec.top_k)
    outs = llm.generate([vllm_api.tokens_prompt(s) for s in seqs], sp, use_tqdm=False)
    res = []
    for o in outs:
        res.append([{} if d is None else {int(t): float(l.logprob) for t, l in d.items()}
                    for d in o.prompt_logprobs])
    return res


@ROLLOUT.register("vllm_colocate", source="Self-Distill/OPSD-main/opsd_trainer.py:301-354,1175-1245")
class VLLMColocate(RolloutEngine):
    """vLLM engine on the training GPU; weights follow the student (full: ``load_weights``; LoRA:
    ``merge`` or ``hot_load``). While training it sleeps to give its memory back.

    ``response_logp`` is the RAW logprob at temperature 1 of each sampled token (``logprobs=0`` under
    vLLM's default ``raw_logprobs`` mode), the same convention as the ``hf`` engine.
    Sync is lazy: weights are pushed at the first ``generate`` after the step changed (or after the
    ``rollout_weight_sync`` callback marked them stale; with that callback present ``every`` is honoured
    and the step-change auto detection is switched off).
    Source: Self-Distill/OPSD-main/opsd_trainer.py:301-354,1175-1250.
    """

    class Config(BaseConfig):
        gpu_memory_utilization: float = 0.4
        tensor_parallel_size: int = 1
        enable_sleep_mode: bool = True
        sleep_level: int = 1
        max_model_len: int | None = None
        enable_prefix_caching: bool = True
        lora_mode: str = "hot_load"   # hot_load | merge
        sync_every: int = 1
        model: str = "student"
        dtype: str = "bfloat16"
        max_lora_rank: int = 64
        seed: int = 0
        enforce_eager: bool = False
        extra_eos_ids: list[int] = []
        max_logprobs: int = 20

    auto_stale = True      # RolloutWeightSync.setup switches this off

    def setup(self, rt: Any) -> None:
        self.rt = rt
        cfg = self.cfg
        spec = getattr(rt.config.models, cfg.model)
        tok = getattr(rt, f"{cfg.model}_tokenizer")
        self._tok = tok
        self._lora = spec.lora is not None and cfg.lora_mode == "hot_load"
        self._merge = spec.lora is not None and cfg.lora_mode == "merge"
        if self._lora and cfg.enable_sleep_mode and cfg.sleep_level >= 2:
            raise ValueError("sleep_level=2 discards the base weights, which hot_load never re-sends: use "
                             "sleep_level=1 or lora_mode='merge'")
        if cfg.lora_mode not in ("hot_load", "merge"):
            raise ValueError(f"lora_mode must be 'hot_load' or 'merge', got '{cfg.lora_mode}'")
        self.llm = make_llm(
            spec.path, spec.tokenizer or spec.path, world_size=rt.engine.world_size, rank=rt.engine.rank,
            seed=cfg.seed, tensor_parallel_size=cfg.tensor_parallel_size, dtype=cfg.dtype,
            gpu_memory_utilization=cfg.gpu_memory_utilization, enable_sleep_mode=cfg.enable_sleep_mode,
            max_model_len=cfg.max_model_len, enable_prefix_caching=cfg.enable_prefix_caching,
            enable_lora=self._lora, max_lora_rank=cfg.max_lora_rank, enforce_eager=cfg.enforce_eager,
            max_logprobs=cfg.max_logprobs)
        self._stale = True
        self._last_sync_step = -1
        self._resync = False
        self._lora_id = 0
        self._lora_request = None
        self._lora_root = Path(rt.config.train.output_dir) / "rollout_lora" / f"rank{rt.engine.rank}"
        if cfg.enable_sleep_mode:
            self._sleep()

    # ------------------------------------------------------------------ sleep / sync
    def _sleep(self) -> None:
        self.llm.sleep(level=self.cfg.sleep_level)
        if self.cfg.sleep_level >= 2:
            self._resync = True            # level 2 dropped the weights themselves

    def _wake(self, tags: list[str]) -> None:
        torch.cuda.empty_cache()
        self.llm.wake_up(tags=tags)

    def mark_stale(self) -> None:
        self._stale = True

    def _needs_sync(self) -> bool:
        step = self.rt.state.step
        if self._last_sync_step < 0 or self._resync:
            return True
        stale = self._stale or (self.auto_stale and step != self._last_sync_step)
        return stale and step - self._last_sync_step >= self.cfg.sync_every

    def _maybe_sync(self) -> None:
        sleeping = self.cfg.enable_sleep_mode
        if sleeping:
            self._wake(["weights"])        # level 1: restores the offloaded weights even if no sync follows
        if not self._needs_sync():
            return
        step = self.rt.state.step
        if self._lora:
            path = save_lora_adapter(self.rt, self._lora_root, step)
            self._lora_id += 1
            self._lora_request = vllm_api.lora_request(f"step{step}", self._lora_id, str(path))
        else:
            load_weights(self.llm, iter_student_weights(self.rt, "merge" if self._merge else "full"))
        self.update_done(step)

    def update_done(self, step: int) -> None:
        self.llm.reset_prefix_cache()
        self._stale = False
        self._resync = False
        self._last_sync_step = step

    def update_weights(self, named_tensors) -> None:
        load_weights(self.llm, named_tensors)

    def reset_prefix_cache(self) -> None:
        self.llm.reset_prefix_cache()

    def sleep(self, level: int = 1) -> None:
        self.llm.sleep(level=level)

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.llm.wake_up(tags=tags)

    # ------------------------------------------------------------------ generation
    def generate(self, prompts, spec: SamplingSpec, lora=None) -> list[list[Rollout]]:

        self._maybe_sync()
        if self.cfg.enable_sleep_mode:
            self._wake(["kv_cache"])
        eos = eos_ids(self._tok, self.cfg.extra_eos_ids)
        sp = vllm_api.sampling_params(
            n=spec.n, temperature=spec.temperature, top_p=spec.top_p, top_k=spec.top_k if spec.top_k > 0 else -1,
            max_tokens=spec.max_new_tokens, stop=spec.stop or None,
            stop_token_ids=(eos[1:] + list(spec.stop_token_ids)) or None,
            repetition_penalty=spec.repetition_penalty, seed=spec.seed,
            logprobs=0 if spec.return_logprobs else None)
        try:
            outs = self.llm.generate([vllm_api.tokens_prompt(list(p)) for p in prompts], sp,
                                     lora_request=lora or self._lora_request, use_tqdm=False)
        finally:
            if self.cfg.enable_sleep_mode:
                self._sleep()
        res = []
        for p, o in zip(prompts, outs):
            rs = []
            for c in o.outputs:
                ids = [int(t) for t in c.token_ids]
                lp = None
                if spec.return_logprobs:
                    lp = [float(c.logprobs[t][ids[t]].logprob) for t in range(len(ids))]
                rs.append(Rollout(list(p), ids, lp, c.finish_reason or "stop", c.text))
            res.append(rs)
        return res

    def score(self, seqs, spec: ScoreSpec):
        self._maybe_sync()
        if self.cfg.enable_sleep_mode:
            self._wake(["kv_cache"])
        try:
            return score_with_llm(self.llm, seqs, spec)
        finally:
            if self.cfg.enable_sleep_mode:
                self._sleep()
