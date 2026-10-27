"""Self-distillation criteria (OPSD tinker_rkl, etc.)."""
from __future__ import annotations

from typing import ClassVar

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


@CRITERIONS.register("tinker_rkl", source="Self-Distill/OPSD-main/opsd_trainer.py:705-726")
class TinkerRKL(Criterion):
    """Sampled-token reverse-KL policy gradient (Thinking Machines / OPSD).

    Formula:
        A_t = stop_gradient(log T(y_t) - log S(y_t))
        L   = -sum_t A_t * log S(y_t)

    Normalization note:
        The original implementation in Self-Distill/OPSD-main/opsd_trainer.py:705-726 computes
        the mean over masked tokens within a single micro-batch:
            loss = -(advantage * student_log_probs_sampled_masked).mean()
        This corresponds to token_sum / Z when running a single micro-batch.
        With multiple micro-batches and gradient accumulation, LM-KD accumulates token_sum
        numerators and divides by the global supervised token count Z across all micro-batches
        and ranks, ensuring exact equivalence across accumulation settings.
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

