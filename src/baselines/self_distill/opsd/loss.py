"""On-Policy Self-Distillation (OPSD) losses.

Source: Self-Distill/OPSD-main (opsd_trainer.py)
Supports:
- Generalized Jensen-Shannon Divergence (GJSD) with optional token clipping
- Sampled-token reverse-KL policy gradient (Tinker variant)
- Dedicated OPSD Criterion
"""

from __future__ import annotations

from functools import partial
from typing import ClassVar

import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.functional.divergences import generalized_jsd
from lmkd.losses.functional.rl import masked_mean
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.logits_view import chunked_token_divergence
from lmkd.teachers.signals import TeacherSignalSpec


def opsd_tinker_loss(
    student_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampled-token reverse-KL policy gradient loss and mean advantage."""
    adv = (teacher_logp - student_logp).detach()
    token_loss = -adv * student_logp
    if mask is not None:
        loss = masked_mean(token_loss, mask)
        adv_mean = masked_mean(adv, mask)
    else:
        loss = token_loss.mean()
        adv_mean = adv.mean()
    return loss, adv_mean


@CRITERIONS.register("tinker_rkl", source="Self-Distill/OPSD-main/opsd_trainer.py:705-726")
@CRITERIONS.register("opsd_tinker", source="Self-Distill/OPSD-main/opsd_trainer.py:705-726")
class TinkerRKL(Criterion):
    """Sampled-token reverse-KL policy gradient (Thinking Machines / OPSD).

    Formula:
        A_t = stop_gradient(log T(y_t) - log S(y_t))
        L   = -sum_t A_t * log S(y_t)
    """

    needs_teacher = True
    schedulable_fields: ClassVar[tuple[str, ...] | None] = ("temperature",)

    class Config(BaseConfig):
        temperature: float = 1.0

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="sampled_logp", temperature=self.cfg.temperature),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.student is None or ctx.student.logits is None:
            raise ValueError("tinker_rkl requires student logits view")
        if ctx.teacher is None or ctx.teacher.token_logp is None:
            raise ValueError("tinker_rkl requires teacher with sampled_logp signal")

        labels = ctx.batch["student.labels"]
        mask = ctx.supervised_mask()
        s = ctx.student.logits.with_temperature(self.cfg.temperature).target_logp(labels)
        a = (ctx.teacher.token_logp - s).detach()

        loss_term = LossTerm(-(a * s * mask).sum(), "token_sum")
        adv_mean = ((a * mask).sum() / mask.sum().clamp(min=1)).detach()
        return LossOutput({"pg": loss_term}, metrics={"adv_mean": adv_mean})


@CRITERIONS.register("opsd", source="Self-Distill/OPSD-main/opsd_trainer.py:382")
class OPSD(Criterion):
    """On-Policy Self-Distillation (OPSD) criterion via Generalized JSD."""

    needs_teacher = True

    class Config(BaseConfig):
        beta: float = 0.0
        token_clip: float | None = 0.05
        clip_mode: str = "elementwise"
        temperature: float = 1.1
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.student is None or ctx.student.logits is None:
            raise ValueError("OPSD requires student logits view")
        if ctx.teacher is None or ctx.teacher.logits is None:
            raise ValueError("OPSD requires teacher logits view")

        s_view = ctx.student_logits()
        t_view = ctx.teacher.logits
        mask = ctx.supervised_mask()
        fn = partial(
            generalized_jsd,
            beta=self.cfg.beta,
            token_clip=self.cfg.token_clip,
            clip_mode=self.cfg.clip_mode,
            temperature=self.cfg.temperature,
        )
        per_tok = chunked_token_divergence(s_view, t_view, fn, mask=mask, chunk_size=self.cfg.chunk_size)
        loss_val = per_tok.sum()
        denom = mask.float().sum().clamp(min=1.0)
        metrics = {
            "opsd/loss_mean": (loss_val / denom).detach(),
        }
        return LossOutput({"opsd": LossTerm(loss_val, "token_sum")}, metrics=metrics)


__all__ = [
    "OPSD",
    "TinkerRKL",
    "generalized_jsd",
    "opsd_tinker_loss",
]
