"""Direct Preference Distillation Optimization (DP3O).

Blends hard DPO and soft DPO using teacher reward differences.
Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/preference/losses.py:109-124
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


def dp3o_loss(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    reward_diff: torch.Tensor | None = None,
    *,
    beta: float = 0.1,
    alpha: float = 0.2,
) -> torch.Tensor:
    """DP3O objective: alpha * hard_DPO + (1 - alpha) * soft_DPO."""
    logits = beta * (chosen - rejected)
    hard = -F.logsigmoid(logits)
    if alpha == 1.0:
        return hard
    if reward_diff is None or reward_diff.shape != chosen.shape:
        raise ValueError("DP3O with dp3o_alpha < 1 requires one reward difference per pair")
    target = torch.sigmoid(reward_diff.to(logits.dtype))
    soft = -F.logsigmoid(logits) * target - F.logsigmoid(-logits) * (1 - target)
    return alpha * hard + (1 - alpha) * soft


@CRITERIONS.register(
    "dp3o",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/preference/losses.py:109",
)
class DP3O(Criterion):
    """Direct Preference Distillation Optimization (DP3O) blending hard and soft DPO.

    Chosen response is index 0, rejected is index 1.
    Formula: alpha * DPO_hard + (1 - alpha) * soft_label_DPO(sigmoid(reward_diff)).
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/preference/losses.py:109-124
    """

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        alpha: float = 0.2
        beta: float = 0.05
        average_log_prob: bool = False
        reduction: str = "mean"
        chunk_size: int = 1024

    @classmethod
    def required_keys(cls, cfg: BaseConfig) -> tuple[str, ...]:
        c = cfg if isinstance(cfg, cls.Config) else cls.Config(**cfg.model_dump())
        return ("group.size", "pref.reward") if c.alpha < 1.0 else ("group.size",)

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="sampled_logp"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        n = _check_pairwise(ctx.batch, "dp3o")
        if ctx.teacher is None or ctx.teacher.token_logp is None:
            raise ValueError("DP3O requires teacher with sampled_logp signal")

        seq_pol, _, mask = sequence_logps(
            ctx.student_logits(),
            ctx.batch["student.labels"],
            average=self.cfg.average_log_prob,
            chunk_size=self.cfg.chunk_size,
        )
        ref_seq = (ctx.teacher.token_logp * mask).sum(-1)
        if self.cfg.average_log_prob:
            ref_seq = ref_seq / mask.sum(-1).clamp_min(1.0)

        lr_c, lr_r = pairwise(seq_pol - ref_seq)
        reward_diff = None
        if self.cfg.alpha < 1.0:
            rew_c, rew_r = pairwise(ctx.batch["pref.reward"])
            reward_diff = rew_c - rew_r

        loss = dp3o_loss(lr_c, lr_r, reward_diff=reward_diff, beta=self.cfg.beta, alpha=self.cfg.alpha)
        term = _reduction_term(loss, self.cfg.reduction, n)
        metrics = {
            "accuracy": (lr_c > lr_r).float().mean().detach(),
            "margin": (self.cfg.beta * (lr_c - lr_r)).mean().detach(),
        }
        return LossOutput({"dp3o": term}, metrics=metrics)
