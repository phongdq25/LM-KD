"""Preference-Based Self-Distillation (PBSD).

Teacher and student rollout preference distillation.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from baselines.preference.common import _check_pairwise, _reduction_term, pairwise, sequence_logps
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def pbsd_loss(
    teacher_on_teacher: torch.Tensor,
    student_on_teacher: torch.Tensor,
    teacher_on_student: torch.Tensor,
    student_on_student: torch.Tensor,
    *,
    beta: float = 0.1,
    loss_type: str = "logistic_gap",
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preference-Based Self-Distillation objective (Yu et al., 2026)."""
    delta = (teacher_on_teacher - student_on_teacher) - (teacher_on_student - student_on_student)
    if loss_type == "logistic_gap":
        losses = -F.logsigmoid(-beta * delta)
    elif loss_type == "mse":
        losses = (torch.sigmoid(delta) - 0.5) ** 2
    elif loss_type == "mse_to_zero":
        losses = delta * delta
    else:
        raise ValueError(f"Unknown PBSD loss type: {loss_type}")
    if gate is None:
        gate = torch.ones_like(delta)
    return (losses * gate).sum() / gate.sum().clamp(min=1.0)


@CRITERIONS.register(
    "pbsd",
)
class PBSD(Criterion):
    """Preference-Based Self-Distillation (PBSD) on teacher and student rollouts.

    Teacher rollout is index 0, student rollout is index 1.
    Formula: delta = (T(y_t) - S(y_t)) - (T(y_s) - S(y_s)); loss = -log sigmoid(-beta * delta).
    """

    needs_teacher: ClassVar[bool] = True
    requires: ClassVar[tuple[str, ...]] = ("group.size", "pref.gate")

    class Config(BaseConfig):
        beta: float = 0.1
        loss_type: str = "logistic_gap"
        normalize_by_length: bool = False
        reduction: str = "mean"
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="sampled_logp"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        n = _check_pairwise(ctx.batch, "pbsd")
        if ctx.teacher is None or ctx.teacher.token_logp is None:
            raise ValueError("PBSD requires teacher with sampled_logp signal")

        seq_s, _, mask = sequence_logps(
            ctx.student_logits(),
            ctx.batch["student.labels"],
            average=self.cfg.normalize_by_length,
            chunk_size=self.cfg.chunk_size,
        )
        seq_t = (ctx.teacher.token_logp * mask).sum(-1)
        if self.cfg.normalize_by_length:
            seq_t = seq_t / mask.sum(-1).clamp_min(1.0)

        t_on_t, t_on_s = pairwise(seq_t)
        s_on_t, s_on_s = pairwise(seq_s)
        gate, _ = pairwise(ctx.batch["pref.gate"])

        loss = pbsd_loss(t_on_t, s_on_t, t_on_s, s_on_s, beta=self.cfg.beta, loss_type=self.cfg.loss_type, gate=gate)
        term = _reduction_term(loss.unsqueeze(0), self.cfg.reduction, n)
        return LossOutput({"pbsd": term}, metrics={"loss": loss.detach()})
