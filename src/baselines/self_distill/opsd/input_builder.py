"""OPSD Teacher Input Builder (ReasonFirst).

Teacher writes an analysis of the reference solution first and uses it as context.
"""

from __future__ import annotations

from typing import Any

from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TEACHER_INPUTS
from lmkd.data.transforms.tokenize import render_prompt
from lmkd.rollout.base import SamplingSpec
from lmkd.teachers.input_builders import (
    TeacherInputBuilder,
    _collate_teacher_prompts_and_responses,
    _truncate_prompt,
)


@TEACHER_INPUTS.register("reason_first")
class ReasonFirst(TeacherInputBuilder):
    """Teacher writes an analysis of the reference solution first and uses it as context (OPSD reason_first)."""

    provides = ("teacher.input_ids", "teacher.attention_mask", "teacher.labels", "teacher.prompt_len")

    class Config(BaseConfig):
        reasoning_template: str
        transition_template: str
        fields: list[str] = ["solution"]
        sampling: dict = {}
        use_chat_template: bool = True
        chat_template_kwargs: dict = {}
        max_prompt_len: int = 4096
        truncation_side: str = "left"
        tokenizer: str = "student"

    def build(self, batch: Batch, rt: Any) -> Batch:
        if getattr(rt, "rollout", None) is None:
            raise ValueError("reason_first teacher input requires 'rollout:' to be configured in runtime")

        tok = (
            rt.student_tokenizer if self.cfg.tokenizer == "student" else (rt.teacher_tokenizer or rt.student_tokenizer)
        )
        batch_size = batch.batch_size
        r_prompt_ids_list: list[list[int]] = []
        meta_list: list[dict[str, Any]] = []

        for b in range(batch_size):
            meta_b = {k.removeprefix("meta."): v[b] for k, v in batch.meta.items() if k.startswith("meta.")}
            for f in self.cfg.fields:
                if f not in meta_b:
                    raise KeyError(f"ReasonFirst: required meta field {f!r} missing. Available meta: {sorted(meta_b)}")
            meta_list.append(meta_b)
            r_text_b = self.cfg.reasoning_template.format(**meta_b)
            r_p_text = render_prompt(tok, r_text_b, self.cfg.use_chat_template, **self.cfg.chat_template_kwargs)
            r_p_ids = tok(r_p_text, add_special_tokens=True)["input_ids"]
            r_p_ids = _truncate_prompt(r_p_ids, self.cfg.max_prompt_len, self.cfg.truncation_side)
            r_prompt_ids_list.append(r_p_ids)

        spec = SamplingSpec(**{**self.cfg.sampling, "n": 1})
        rolls = rt.rollout.generate(r_prompt_ids_list, spec)

        teacher_prompts: list[list[int]] = []
        for b in range(batch_size):
            meta_b = meta_list[b]
            try:
                t_text = self.cfg.transition_template.format(**meta_b)
            except Exception:
                t_text = self.cfg.transition_template
            t_ids = tok(t_text, add_special_tokens=False)["input_ids"]

            roll = rolls[b][0]
            resp_ids = list(roll.response_ids)
            if tok.eos_token_id is not None and resp_ids and resp_ids[-1] == tok.eos_token_id:
                resp_ids = resp_ids[:-1]

            body = resp_ids + t_ids
            prompt_prefix = r_prompt_ids_list[b]
            if len(prompt_prefix) + len(body) > self.cfg.max_prompt_len:
                avail = max(0, self.cfg.max_prompt_len - len(prompt_prefix))
                t_prompt = prompt_prefix + body[:avail]
            else:
                t_prompt = prompt_prefix + body
            teacher_prompts.append(t_prompt)

        return _collate_teacher_prompts_and_responses(batch, rt, tok, teacher_prompts)


__all__ = ["ReasonFirst"]
