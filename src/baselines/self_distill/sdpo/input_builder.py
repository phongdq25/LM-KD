"""SDPO: Self-Distillation with Preference Optimization - Teacher Input Builder (SDPOReprompt).

SDPO teacher prompts use the first successful peer response and/or verifier feedback.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from pydantic import Field

from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TEACHER_INPUTS
from lmkd.data.transforms.tokenize import render_prompt
from lmkd.teachers.input_builders import (
    TeacherInputBuilder,
    _collate_teacher_prompts_and_responses,
    _truncate_prompt,
)


@TEACHER_INPUTS.register("sdpo_reprompt")
class SDPOReprompt(TeacherInputBuilder):
    """SDPO teacher prompts use the first successful peer response and/or verifier feedback."""

    requires = ("student.input_ids", "student.prompt_len", "student.attention_mask", "rollout.reward", "group.uid")
    provides = (
        "teacher.input_ids",
        "teacher.attention_mask",
        "teacher.labels",
        "teacher.prompt_len",
        "rollout.has_signal",
    )

    class Config(BaseConfig):
        reprompt_template: str
        solution_template: str = ""
        feedback_template: str = ""
        success_threshold: float = 0.5
        dont_reprompt_on_self_success: bool = True
        include_feedback: bool = True
        max_reprompt_len: int = Field(default=10240, gt=0)
        feedback_only_without_solution: bool = False
        remove_thinking_from_demonstration: bool = False
        truncation_side: str = "left"
        chat_template_kwargs: dict = {}
        write_loss_mask: bool = True

    @classmethod
    def provided_keys(cls, cfg: Config) -> tuple[str, ...]:
        return cls.provides + (("seq.loss_mask",) if cfg.write_loss_mask else ())

    def _select(self, batch: Batch) -> tuple[dict[str, list[int]], list[str | None], list[str]]:
        uids, texts = batch.meta["group.uid"], batch.meta["meta.response_text"]
        success: dict[str, list[int]] = {uid: [] for uid in uids}
        for i, uid in enumerate(uids):
            if float(batch["rollout.reward"][i]) >= self.cfg.success_threshold:
                success[uid].append(i)
        solutions: list[str | None] = []
        feedbacks: list[str] = []
        available = batch.meta.get("meta.feedback", [""] * len(uids))
        for i, uid in enumerate(uids):
            peers = [j for j in success[uid] if j != i or not self.cfg.dont_reprompt_on_self_success]
            solution = texts[peers[0]] if peers else None
            if solution is not None and self.cfg.remove_thinking_from_demonstration:
                solution = re.sub(r"<think>.*?</think>\s*", "", solution, flags=re.DOTALL)
            feedback = available[i] if self.cfg.include_feedback else ""
            if self.cfg.feedback_only_without_solution and solution is not None:
                feedback = ""
            solutions.append(solution)
            feedbacks.append(feedback)
        return success, solutions, feedbacks

    def build(self, batch: Batch, rt: Any) -> Batch:
        batch = batch.copy()
        tok = rt.student_tokenizer
        success, solutions, feedbacks = self._select(batch)
        prompts: list[list[int]] = []
        signals: list[float] = []
        for i, (solution, feedback) in enumerate(zip(solutions, feedbacks)):
            messages = batch.meta.get("meta.messages", [None] * batch.batch_size)[i]
            raw_prompts = batch.meta.get("meta.raw_prompt") or batch.meta.get("meta.prompt_text") or []
            prompt = messages[-1]["content"] if messages else raw_prompts[i]
            has_signal = solution is not None or bool(feedback)
            if has_signal:
                prompt = self.cfg.reprompt_template.format(
                    prompt=prompt,
                    solution=self.cfg.solution_template.format(successful_previous_attempt=solution)
                    if solution is not None
                    else "",
                    feedback=self.cfg.feedback_template.format(feedback_raw=feedback) if feedback else "",
                )
            system = [m for m in messages[:-1] if m["role"] == "system"] if messages else []
            text = render_prompt(
                tok, system + [{"role": "user", "content": prompt}], True, **self.cfg.chat_template_kwargs
            )
            ids = tok(text, add_special_tokens=not bool(tok.chat_template))["input_ids"]
            prompts.append(_truncate_prompt(ids, self.cfg.max_reprompt_len, self.cfg.truncation_side))
            signals.append(float(has_signal))
        batch = _collate_teacher_prompts_and_responses(batch, rt, tok, prompts)
        signal = torch.tensor(signals, device=batch["student.input_ids"].device)
        batch["rollout.has_signal"] = signal
        if self.cfg.write_loss_mask:
            batch["seq.loss_mask"] = signal[:, None].expand_as(batch["student.labels"]).clone() * batch.get(
                "seq.loss_mask", 1.0
            )
        rt.state.log("sdpo/success_group_fraction", sum(bool(v) for v in success.values()) / len(success))
        rt.state.log("sdpo/success_sample_fraction", sum(s is not None for s in solutions) / batch.batch_size)
        rt.state.log("sdpo/feedback_used_fraction", sum(bool(f) for f in feedbacks) / batch.batch_size)
        rt.state.log("sdpo/reprompt_sample_fraction", signal.mean())
        return batch


__all__ = ["SDPOReprompt"]
