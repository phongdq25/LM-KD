"""HF ``generate`` rollout engine (CPU-friendly; used by tests and unsupported architectures)."""
from __future__ import annotations

import contextlib
from typing import Any

import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ROLLOUT
from lmkd.models.adapters import unwrap
from lmkd.rollout.base import Rollout, RolloutEngine, SamplingSpec, eos_ids


@ROLLOUT.register("hf", source="white-box-distillation/same-tokenizer/distillm/sampler.py")
class HFRollout(RolloutEngine):
    """transformers.generate with manual left padding; token ids in, token ids out.

    ``response_logp`` is the RAW logprob at temperature 1 (``output_logits``, before temperature /
    top-p / top-k), identical to vLLM ``raw_logprobs``.  Formula: lp_t = log_softmax(logits_t)[y_t].
    Fixes vs. DistiLLM sampler: response labels are built by ``rollouts_to_samples`` (target-aligned),
    and response length is found from the first EOS, never by filtering the pad value (pad may == eos).
    """

    class Config(BaseConfig):
        batch_size: int = 16            # prompts per generate call
        model: str = "student"          # student | teacher
        extra_eos_ids: list[int] = []

    def setup(self, rt: Any) -> None:
        self.rt = rt

    def _model_tok(self):
        if self.cfg.model == "teacher":
            return unwrap(self.rt.teacher_model), self.rt.teacher_tokenizer
        return unwrap(self.rt.student_model), self.rt.student_tokenizer

    @torch.no_grad()
    def generate(self, prompts, spec: SamplingSpec, lora=None) -> list[list[Rollout]]:
        model, tok = self._model_tok()
        was_training = model.training
        model.eval()
        eos = eos_ids(tok, [*self.cfg.extra_eos_ids, *spec.stop_token_ids])
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
        device = next(model.parameters()).device
        results: list[list[Rollout]] = []
        try:
            for s in range(0, len(prompts), self.cfg.batch_size):
                chunk = prompts[s: s + self.cfg.batch_size]
                P = max(len(p) for p in chunk)
                ids = torch.full((len(chunk), P), pad, dtype=torch.long)
                am = torch.zeros((len(chunk), P), dtype=torch.long)
                for i, p in enumerate(chunk):
                    ids[i, P - len(p):] = torch.tensor(p, dtype=torch.long)
                    am[i, P - len(p):] = 1
                kw: dict[str, Any] = {}
                if spec.stop:
                    kw.update(stop_strings=spec.stop, tokenizer=tok)
                with contextlib.ExitStack() as stack:
                    if spec.seed is not None:           # seed only this call: never reset the trainer's RNG
                        stack.enter_context(torch.random.fork_rng(
                            devices=[device] if device.type == "cuda" else []))
                        torch.manual_seed(spec.seed)
                    out = self._generate(model, ids.to(device), am.to(device), spec, eos, pad, kw)
                seqs = out.sequences.cpu()
                for i, p in enumerate(chunk):
                    rs = []
                    for j in range(spec.n):
                        row = i * spec.n + j
                        resp = seqs[row, P:].tolist()
                        reason = "length"
                        for k, t in enumerate(resp):                 # first EOS; never filter by pad value
                            if t in eos:
                                resp, reason = resp[: k + 1], "stop"
                                break
                        text = tok.decode(resp, skip_special_tokens=True)
                        if reason == "length" and spec.stop and any(text.endswith(x) for x in spec.stop):
                            reason = "stop"                          # HF stopped on a stop string
                        logp = None
                        if spec.return_logprobs:
                            logp = [float(torch.log_softmax(out.logits[t][row].float(), -1)[tok_id])
                                    for t, tok_id in enumerate(resp)]
                        rs.append(Rollout(list(p), resp, logp, reason, text))
                    results.append(rs)
        finally:
            if was_training:
                model.train()
        return results

    def _generate(self, model, ids, am, spec, eos, pad, kw):
        # top_k=0 DISABLES top-k; None would fall back to the model's generation_config (e.g. 20 / 50),
        # which would silently differ from vLLM (top_k=-1).
        extra = dict(kw)
        eng = getattr(getattr(self, "rt", None), "engine", None)
        if eng is not None and getattr(eng, "is_partitioned", False) and getattr(eng, "world_size", 1) > 1:
            extra.setdefault("synced_gpus", True)
        return model.generate(
            input_ids=ids, attention_mask=am,
            do_sample=spec.temperature > 0, temperature=max(spec.temperature, 1e-5),
            top_p=spec.top_p, top_k=(spec.top_k if spec.top_k > 0 else 0),
            max_new_tokens=spec.max_new_tokens, repetition_penalty=spec.repetition_penalty,
            num_return_sequences=spec.n, eos_token_id=eos, pad_token_id=pad,
            return_dict_in_generate=True, output_logits=spec.return_logprobs, **extra)

    def update_weights(self, named_tensors) -> None:
        return None            # shares the training model
