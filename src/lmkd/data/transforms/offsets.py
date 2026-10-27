"""``token_offsets``: char offset of every training token (cross-tokenizer span service, DESIGN §11.4)."""
from __future__ import annotations

from typing import Any, Literal

import torch

from lmkd.core.batch import FIELDS, FieldSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.spans import prompt_response_offsets
from lmkd.data.transforms.base import Transform

FIELDS.register(FieldSpec("student.offsets", "token_k", torch.long, -1, "student", "char [start,end) per token"))
FIELDS.register(FieldSpec("teacher.offsets", "token_k", torch.long, -1, "teacher", "char [start,end) per token"))


@TRANSFORMS.register("token_offsets", source="IMPACT/code/sra_span_utils.py:283-314")
class TokenOffsets(Transform):
    """Per-token char offsets ``[start, end)`` of the exact ``input_ids`` used for training (student, optionally
    teacher).

    Prompt and response are tokenized separately and the response offsets shifted by ``len(prompt_text)``
    (``compute_separate_offsets`` of IMPACT, not the joint ``compute_text_offsets``); the rebuilt ids must equal the
    training ids or the sample is dropped / raises (``on_mismatch``). Needs ``tokenize_sft(keep_text=true)``; with
    ``teacher: true`` use ``tokenize_sft(teacher_prompt_text=same)`` so both sides share one char space.
    """
    requires = ("student.input_ids", "student.prompt_len")
    provides = ("student.offsets",)

    class Config(BaseConfig):
        teacher: bool = False
        add_eos: bool = True                       # must equal tokenize_sft.add_eos
        prompt_add_special_tokens: bool = True     # must equal tokenize_sft.prompt_add_special_tokens
        on_mismatch: Literal["drop", "raise"] = "drop"

    @classmethod
    def required_keys(cls, cfg: Any) -> tuple[str, ...]:
        keys = tuple(cls.requires)
        if cfg is not None and cfg.teacher:
            keys += ("teacher.input_ids", "teacher.prompt_len")
        return keys

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        keys = tuple(cls.provides)
        if cfg is not None and cfg.teacher:
            keys += ("teacher.offsets",)
        return keys

    def _side(self, tok: Any, sample: dict, side: str, prompt_key: str) -> tuple[list | None, str]:
        res = prompt_response_offsets(
            tok, sample[prompt_key], sample["meta.response_text"], sample[f"{side}.input_ids"],
            int(sample[f"{side}.prompt_len"]), add_eos=self.cfg.add_eos,
            prompt_add_special_tokens=self.cfg.prompt_add_special_tokens)
        return ([[s, e] for s, e in res.offsets] if res.ok and res.offsets else None), res.reason

    def __call__(self, sample: dict, ctx: DataContext) -> dict | None:
        for key in ("meta.prompt_text", "meta.response_text"):
            if key not in sample:
                raise KeyError(f"token_offsets needs {key!r} (tokenize_sft with keep_text=true)")
        sides = [("student", ctx.student_tokenizer, "meta.prompt_text")]
        if self.cfg.teacher:
            if ctx.teacher_tokenizer is None:
                raise ValueError("token_offsets(teacher=true) needs a teacher tokenizer")
            sides.append(("teacher", ctx.teacher_tokenizer, "meta.teacher_prompt_text"
                          if "meta.teacher_prompt_text" in sample else "meta.prompt_text"))
        out = dict(sample)
        for side, tok, prompt_key in sides:
            offs, reason = self._side(tok, sample, side, prompt_key)
            if offs is None:
                if self.cfg.on_mismatch == "raise":
                    raise ValueError(f"token_offsets({side}): {reason} (sample {sample.get('meta.id')})")
                ctx.extra["offsets_dropped"] = ctx.extra.get("offsets_dropped", 0) + 1
                return None
            out[f"{side}.offsets"] = offs
        return out
