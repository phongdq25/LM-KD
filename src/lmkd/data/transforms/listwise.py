"""Tokenize preference records into contiguous listwise response members."""
from __future__ import annotations

import hashlib
import json
from math import exp, nan
from typing import Any, Literal, cast

import torch

from lmkd.core.batch import FIELDS, FieldKind, FieldSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.labels import make_labels
from lmkd.data.transforms.base import Transform

for _key, _kind, _dtype, _pad, _side in (
    ("group.members", "meta", None, 0, None),
    ("group.size", "scalar", torch.long, 0, None),
    ("group.valid", "scalar", torch.bool, 0, None),
    ("pref.teacher_score", "scalar", torch.float32, 0, None),
    ("pref.reward", "scalar", torch.float32, 0, None),
    ("pref.gate", "scalar", torch.float32, 0, None),
    ("cache.topk_ids", "token_k", torch.long, 0, "student"),
    ("cache.topk_p", "token_k", torch.float32, 0, "student"),
    ("cache.tail_p", "token", torch.float32, 0, "student"),
    ("cache.adv_ids", "token_k", torch.long, -1, "student"),
    ("cache.adv_val", "token_k", torch.float32, 0, "student"),
    ("cache.token_logp", "token", torch.float32, 0, "student"),
    ("cache.seq_logp", "scalar", torch.float32, 0, None),
    ("pref.ctpd_groups", "ragged", torch.long, -1, None),
    ("pref.ctpd_w", "ragged", torch.float32, 0, None),
    ("meta.cache_tokenize_fp", "meta", None, 0, None),
    ("meta.raw_prompt", "meta", None, 0, None),
):
    FIELDS.register(FieldSpec(_key, cast(FieldKind, _kind), _dtype, _pad,
                              cast(Literal["student", "teacher"] | None, _side)))


def _is_messages(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict) and "role" in item and "content" in item for item in value
    )


def tokenize_fingerprint(ids: list[int], labels: list[int]) -> str:
    """Stable hash of the exact target positions used by an offline cache."""
    payload = json.dumps([ids, labels], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _chat(tok: Any, messages: list[dict], *, generate: bool) -> str:
    if not hasattr(tok, "apply_chat_template"):
        raise ValueError("chat messages need a tokenizer with apply_chat_template")
    return str(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generate))


def _response_text(tok: Any, prompt: Any, response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and "content" in response:
        return str(response["content"])
    if _is_messages(response):
        if _is_messages(prompt):
            base = list(prompt)
            full_messages = response if response[:len(base)] == base else [*base, *response]
            full = _chat(tok, full_messages, generate=False)
            prompt_text = _chat(tok, base, generate=True)
            if full.startswith(prompt_text):
                return full[len(prompt_text):]
        return str(response[-1]["content"])
    return str(response)


def _normalize(row: dict, tok: Any, format_plain: bool) -> tuple[str, list[str]]:
    if "responses" in row:
        if "prompt" not in row:
            raise ValueError("a multi-response row requires prompt")
        prompt_value, values = row["prompt"], row["responses"]
    elif "chosen" in row and "rejected" in row:
        values = [row["chosen"], row["rejected"]]
        if "prompt" in row:
            prompt_value = row["prompt"]
        elif _is_messages(values[0]) and _is_messages(values[1]):
            if values[0][:-1] != values[1][:-1]:
                raise ValueError("chosen and rejected conversations must share the prompt")
            prompt_value = values[0][:-1]
        else:
            raise ValueError("chosen/rejected need prompt or shared conversations")
    else:
        raise ValueError("expected prompt/responses or chosen/rejected")
    if len(values) < 2:
        raise ValueError("preference rows need at least two responses")
    if _is_messages(prompt_value):
        prompt = _chat(tok, prompt_value, generate=True)
    elif format_plain and getattr(tok, "chat_template", None):
        prompt = _chat(tok, [{"role": "user", "content": str(prompt_value)}], generate=True)
    else:
        prompt = str(prompt_value)
    responses = [_response_text(tok, prompt_value, value) for value in values]
    return prompt, responses


def _score_values(row: dict, cfg: Any, n: int) -> list[float]:
    if cfg.mcq_confidence_filter:
        mcq = row.get("scores_mcq")
        if mcq is None:
            raise ValueError("mcq_confidence_filter requires scores_mcq")
        probs = [exp(float(x)) for x in mcq]
        if not (any(x > 0.5 for x in probs) and any(x <= 0.5 for x in probs)):
            return []
    if cfg.score_mix:
        weights = {"mcq": "scores_mcq", "avglogp": "scores_avglogp"}
        scores = [sum(float(weight) * float(row[weights[key]][i])
                      for key, weight in cfg.score_mix.items()) for i in range(n)]
    else:
        raw = row.get(cfg.score_field) if cfg.score_field else None
        scores = [float(x) for x in raw] if raw is not None else [nan] * n
    if len(scores) != n:
        raise ValueError("teacher scores must match response count")
    return scores


def _tokenize_response(tok: Any, prompt: str, response: str, max_length: int,
                       max_prompt_length: int) -> tuple[list[int], int] | None:
    full_ids = list(tok(prompt + response, add_special_tokens=False)["input_ids"])
    prompt_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
    boundary = len(prompt_ids)
    if full_ids[:boundary] != prompt_ids:
        boundary = max(0, boundary - 1)
    if full_ids[:boundary] != prompt_ids[:boundary]:
        raise ValueError("prompt and response tokenizations differ by more than one token")
    response_ids = full_ids[boundary:]
    if not response_ids:
        return None
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None and response_ids[-1] != eos:
        response_ids.append(int(eos))
    kept_prompt = full_ids[:boundary][-max_prompt_length:]
    budget = max_length - len(kept_prompt)
    if budget < 1:
        return None
    response_ids = response_ids[:budget]
    if not response_ids:
        return None
    return kept_prompt + response_ids, len(kept_prompt)


@TRANSFORMS.register("tokenize_listwise", source="integrations/preference_distillation.py:196-285")
class TokenizeListwise(Transform):
    """Create one group of target-aligned student rows per preference prompt.

    Source: human_alignment/integrations/preference_distillation.py:196-285.
    """

    provides = ("group.members",)

    class Config(BaseConfig):
        format_plain_prompts: bool = True
        max_length: int | None = None
        max_prompt_length: int = 1800
        score_field: str | None = "teacher_scores"
        score_mix: dict[str, float] | None = None
        mcq_confidence_filter: bool = False
        passthrough: list[str] = []

    def __call__(self, sample: dict, ctx: DataContext) -> dict | None:
        tok = ctx.student_tokenizer
        prompt, responses = _normalize(sample, tok, self.cfg.format_plain_prompts)
        scores = _score_values(sample, self.cfg, len(responses))
        if not scores:
            return None
        sid = str(sample.get("meta.id", sample.get("id", "")))
        rows = []
        max_length = self.cfg.max_length or ctx.max_length
        for i, response in enumerate(responses):
            encoded = _tokenize_response(tok, prompt, response, max_length, self.cfg.max_prompt_length)
            if encoded is None:
                return None
            ids, prompt_len = encoded
            row = {"student.input_ids": ids, "student.attention_mask": [1] * len(ids),
                   "student.labels": make_labels(ids, prompt_len).tolist(),
                   "student.prompt_len": prompt_len, "group.index": i,
                   "group.size": len(responses), "group.valid": True,
                   "group.uid": sid, "pref.teacher_score": scores[i],
                   "meta.id": f"{sid}:{i}", "meta.prompt_text": prompt,
                   "meta.response_text": response, "meta.raw_prompt": sample.get("prompt", prompt)}
            if "rewards" in sample:
                row["pref.reward"] = float(sample["rewards"][i])
            for key in self.cfg.passthrough:
                if key in sample:
                    row[key] = sample[key][i]
            rows.append(row)
        return {"group.members": rows, "meta.id": sid}
