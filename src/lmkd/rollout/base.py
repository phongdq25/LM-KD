"""Rollout engine interface (DESIGN §9). Token ids in, token ids out. Never decode+re-tokenize.

Implemented here: data classes + ``rollouts_to_samples`` (the ONLY way to turn generations into
training samples, guaranteeing target-aligned labels) + small helpers. Engines: ``rollout/hf.py``,
``rollout/vllm_colocate.py`` (weight sync: ``rollout/weight_sync.py``).

Convention: ``Rollout.response_logp`` is the RAW logprob (temperature 1, no top-p/top-k) of each sampled
token, like vLLM ``logprobs_mode="raw_logprobs"``.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import torch

from lmkd.core.batch import FIELDS, Batch, FieldSpec
from lmkd.core.plugin import Plugin
from lmkd.data.labels import make_labels

FIELDS.register(FieldSpec("rollout.rollout_logp", "token", torch.float32, 0.0, "student",
                          "logprob of each sampled token under the ROLLOUT engine (raw, T=1; 0 on prompt); target-aligned"))


@dataclass
class SamplingSpec:
    """Sampling parameters of one generation call (n, temperature, top-p/k, length, stop, logprobs, seed)."""
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 1024
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    repetition_penalty: float = 1.0
    seed: int | None = None
    return_logprobs: bool = True      # logprob of each sampled token (IS correction)


@dataclass
class Rollout:
    """One sampled completion: prompt ids, response ids (EOS included), optional per-token logprobs and finish
    reason.
    """
    prompt_ids: list[int]             # no padding
    response_ids: list[int]           # includes EOS if generated
    response_logp: list[float] | None = None
    finish_reason: str = "stop"       # stop | length
    text: str = ""


@dataclass
class ScoreSpec:
    """How to score given sequences (top-k prompt logprobs per position)."""
    top_k: int = 0                     # prompt_logprobs top-k per position (vLLM limit applies)
    start: int = 0                     # first target position to score


class RolloutEngine(Plugin):
    """Generates completions for the current student (HF or vLLM) and keeps its weights in sync."""
    def setup(self, rt: Any) -> None:
        self.rt = rt

    def generate(self, prompts: list[list[int]], spec: SamplingSpec, lora: Any = None) -> list[list[Rollout]]:
        """Returns, for each prompt, ``spec.n`` rollouts."""
        raise NotImplementedError

    def score(self, seqs: list[list[int]], spec: ScoreSpec) -> list[Any]:
        raise NotImplementedError

    def update_weights(self, named_tensors: Iterator[tuple[str, torch.Tensor]]) -> None:
        raise NotImplementedError

    def sleep(self, level: int = 1) -> None:
        return None

    def wake_up(self, tags: list[str] | None = None) -> None:
        return None

    def reset_prefix_cache(self) -> None:
        return None


def sampling_from_cfg(d: dict | None) -> SamplingSpec:
    """``SamplingSpec(**d)`` with a readable error for unknown keys."""
    d = d or {}
    valid = [f.name for f in fields(SamplingSpec)]
    bad = sorted(set(d) - set(valid))
    if bad:
        raise ValueError(f"unknown sampling keys {bad}; valid keys: {valid}")
    return SamplingSpec(**d)


def prompts_from_batch(batch: Batch, side: str = "student") -> list[list[int]]:
    """Prompt token ids of every row (``input_ids[:prompt_len]``). Relies on right padding: never
    filters by pad value (pad may equal eos)."""
    ids, am, plens = batch[f"{side}.input_ids"], batch[f"{side}.attention_mask"], batch[f"{side}.prompt_len"]
    out = []
    for i in range(ids.shape[0]):
        plen = int(plens[i])
        if plen > 0 and bool((am[i, :plen] == 0).any()):
            raise ValueError("left-padded prompt in batch")
        out.append(ids[i, :plen].tolist())
    return out


def eos_ids(tokenizer: Any, extra: Sequence[int] = ()) -> list[int]:
    """``[tokenizer.eos_token_id] + extra`` without None and duplicates (order kept)."""
    out: list[int] = []
    for t in [getattr(tokenizer, "eos_token_id", None), *extra]:
        if t is not None and t not in out:
            out.append(int(t))
    return out


def rollouts_to_samples(rollouts: list[Rollout], *, side: str = "student", meta: list[dict] | None = None,
                        copy_meta: list[dict] | None = None) -> list[dict]:
    """Rollout -> sample dict with target-aligned labels (prompt masked). Handles pad==eos safely
    because lengths come from the rollout, not from token values.

    ``copy_meta``: one source row per rollout; every ``meta.*`` key of that row is copied to the sample."""
    if copy_meta is not None and len(copy_meta) != len(rollouts):
        raise ValueError("copy_meta must have one row per rollout")
    out = []
    for i, r in enumerate(rollouts):
        ids = list(r.prompt_ids) + list(r.response_ids)
        plen = len(r.prompt_ids)
        s = {
            f"{side}.input_ids": ids,
            f"{side}.attention_mask": [1] * len(ids),
            f"{side}.labels": make_labels(ids, plen).tolist(),
            f"{side}.prompt_len": plen,
        }
        if r.response_logp is not None:
            if len(r.response_logp) != len(r.response_ids):
                raise ValueError("response_logp must have one entry per response token")
            # target-aligned: entry t is the logprob of input_ids[t]
            s["rollout.rollout_logp"] = [0.0] * plen + list(r.response_logp)
        if copy_meta is not None:
            s.update({k: v for k, v in copy_meta[i].items() if k.startswith("meta.")})
        if meta is not None:
            s.update(meta[i])
        out.append(s)
    return out

