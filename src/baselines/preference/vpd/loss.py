"""Variational Preference Distillation (VPD) / PAD.

Plackett-Luce negative log-likelihood of teacher ranking over response groups.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from baselines.preference.common import (
    _reduction_term,
    _score_matrix,
    plackett_luce_log_probability,
    sequence_logps,
)
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def vpd_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor, *, beta: float = 1.0) -> torch.Tensor:
    """Vanilla preference distillation via teacher-induced ranking likelihood."""
    _score_matrix(student_scores, "student_scores")
    _score_matrix(teacher_scores, "teacher_scores")
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student_scores and teacher_scores must have the same shape")
    rankings = teacher_scores.argsort(dim=-1, descending=True)
    return -plackett_luce_log_probability(beta * student_scores, rankings)


@CRITERIONS.register(
    "vpd",
)
class VPD(Criterion):
    """Plackett-Luce negative log-likelihood of teacher ranking over response groups.

    Chosen response is index 0, rejected is index 1.
    Formula: -log PL(argsort_desc(teacher_scores) | beta * student_scores).
    """

    needs_teacher: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ("group.size", "group.valid", "pref.teacher_score")

    class Config(BaseConfig):
        beta: float = 10.0
        average_log_prob: bool = True
        reduction: str = "mean"
        fill_missing_from_teacher: bool = False
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        spec = TeacherSignalSpec(kind="sampled_logp") if self.cfg.fill_missing_from_teacher else None
        return FeatureRequest(student=SideRequest(logits="view"), teacher_signal=spec)

    def forward(self, ctx: LossContext) -> LossOutput:
        seq, _, _ = sequence_logps(
            ctx.student_logits(),
            ctx.batch["student.labels"],
            average=self.cfg.average_log_prob,
            chunk_size=self.cfg.chunk_size,
        )
        n = int(ctx.batch["group.size"][0])
        b = seq.shape[0] // n
        s = seq.view(b, n)
        t = ctx.batch["pref.teacher_score"].view(b, n)
        valid = ctx.batch["group.valid"].view(b, n)

        missing = valid & t.isnan()
        if missing.any():
            if not self.cfg.fill_missing_from_teacher or ctx.teacher is None or ctx.teacher.token_logp is None:
                raise ValueError("VPD/PPD requires teacher scores")
            mask = ctx.supervised_mask()
            ref_seq = (ctx.teacher.token_logp * mask).sum(-1)
            if self.cfg.average_log_prob:
                ref_seq = ref_seq / mask.sum(-1).clamp_min(1.0)
            t = torch.where(missing, ref_seq.view(b, n), t)

        if valid.all():
            loss = vpd_loss(s, t, beta=self.cfg.beta)
        else:
            losses = [
                vpd_loss(s[i : i + 1, : int(valid[i].sum())], t[i : i + 1, : int(valid[i].sum())], beta=self.cfg.beta)
                for i in range(b)
            ]
            loss = torch.cat(losses)

        term = _reduction_term(loss, self.cfg.reduction, n)
        return LossOutput({"vpd": term}, metrics={"loss": loss.mean().detach()})


PAD = VPD
