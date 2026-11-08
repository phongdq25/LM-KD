"""RLSD: Reinforcement Learning Self-Distillation - Advantage Modulation Method (STGCA).

Paper: Reinforcement Learning with Self-Distillation

Key Method Characteristics:
1. Self-Teacher Guided Confidence Advantage (STGCA):
   Modulates task verifier advantages (e.g. GRPO centered rewards) using the confidence
   delta between the privileged self-teacher and the student:
       delta_t = (log T(y_t) - log S(y_t)) * mask_t
       weight_t = exp(sign(A_t) * delta_t)
       weight_clipped = clamp(weight_t, 1 - c, 1 + c)
       w_eff = (1 - lambda) + lambda * weight_clipped
       A_eff = A * detach(w_eff)
2. Dynamic Lambda Schedule:
   Linear warmup followed by linear decay over training steps, allowing early exploration
   and late convergence to pure policy optimization.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import Field

from baselines.self_distill.common import effective_lambda, masked_mean
from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import ADV_TRANSFORMS
from lmkd.data.labels import supervised_mask


def rlsd_reweight(
    adv: torch.Tensor,
    teacher_logp: torch.Tensor,
    student_logp: torch.Tensor,
    mask: torch.Tensor,
    lam: float,
    clip: float,
    negative_only: bool = False,
    thought_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RLSD sign-dependent detached token weighting.

    Boosts positive tokens when teacher confidence is higher than student;
    penalizes negative tokens when teacher confidence contradicts the student.
    """
    delta = (teacher_logp.detach() - student_logp.detach()) * mask
    weights = (adv.sign() * delta).exp() * mask
    clipped = weights.clamp(1 - clip, 1 + clip)
    reweight = ((1 - lam) + lam * clipped) * mask
    if negative_only:
        neg = (adv.sum(-1, keepdim=True) < 0).float()
        reweight = neg * reweight + (1 - neg) * mask
    if thought_mask is not None:
        reweight = thought_mask * reweight + (1 - thought_mask) * mask
    result = adv * reweight.detach()
    metrics = {
        "rlsd/delta_mean": masked_mean(delta, mask),
        "rlsd/weight_mean": masked_mean(weights.detach(), mask),
        "rlsd/clipped_weight_mean": masked_mean(clipped.detach(), mask),
        "rlsd/clip_low_ratio": masked_mean((weights < 1 - clip).float(), mask),
        "rlsd/clip_high_ratio": masked_mean((weights > 1 + clip).float(), mask),
    }
    return result, metrics


class LambdaSchedule(BaseConfig):
    """RLSD warmup and decay durations measured in rollout batches."""

    warmup_steps: int = Field(default=0, ge=0)
    decay_steps: int = Field(default=0, ge=0)


@ADV_TRANSFORMS.register("rlsd_reweight")
class RLSDReweight(Plugin):
    """A * ((1-lambda) + lambda*clip(exp(sign(A)*(logT-logS)))) with detached weights."""

    needs_teacher_logp = True

    class Config(BaseConfig):
        lam: float = Field(default=0.5, ge=0, le=1)
        clip: float = Field(default=0.2, ge=0, le=1)
        negative_only: bool = False
        thought_only: bool = False
        lam_schedule: LambdaSchedule = LambdaSchedule()

    def setup(self, rt: Any) -> None:
        if self.cfg.thought_only:
            raise NotImplementedError("RLSD thought_only is optional and not implemented; use thought_only=False")

    def __call__(
        self,
        adv: torch.Tensor,
        batch: Any,
        teacher_logp: torch.Tensor,
        student_logp: torch.Tensor,
        state: Any,
    ) -> torch.Tensor:
        schedule = self.cfg.lam_schedule
        lam = effective_lambda(self.cfg.lam, state.step, schedule.warmup_steps, schedule.decay_steps)
        result, metrics = rlsd_reweight(
            adv,
            teacher_logp,
            student_logp,
            supervised_mask(batch["student.labels"]),
            lam,
            self.cfg.clip,
            self.cfg.negative_only,
        )
        state.log("rlsd/effective_lambda", lam)
        for key, value in metrics.items():
            state.log(key, value)
        return result


# Method class alias
STGCA = RLSDReweight

__all__ = [
    "RLSDReweight",
    "STGCA",
    "LambdaSchedule",
    "rlsd_reweight",
    "effective_lambda",
]
