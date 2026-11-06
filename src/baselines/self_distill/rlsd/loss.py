"""RLSD: Reinforcement Learning Self-Distillation - Policy Optimization Loss.

Paper: Reinforcement Learning with Self-Distillation
Source: Self-Distill/RLSD-main (verl/workers/actor/dp_opsd_actor.py:426-579)

Key Method Characteristics:
1. Clipped surrogate policy gradient with asymmetric lower/upper clipping thresholds:
   pg1 = -A_eff * ratio
   pg2 = -A_eff * clamp(ratio, 1 - clip_low, 1 + clip_high)
   loss = max(pg1, pg2)
2. Negative advantage c-clipping: Optional bound on the penalty for negative advantage tokens.
3. Global token count normalization across micro-batches and data-parallel ranks.
"""

from __future__ import annotations

import torch
from pydantic import Field

from baselines.self_distill.common import masked_mean
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.rl import PPOClip
from lmkd.models.forward import FeatureRequest, SideRequest


def ppo_clip_per_token(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    clip_low: float,
    clip_high: float,
    clip_c: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped PPO numerator per token (RLSD dp_opsd_actor.py:426-579)."""
    ratio = (logp - old_logp.detach()).clamp(-20, 20).exp()
    pg1 = -adv.detach() * ratio
    pg2 = -adv.detach() * ratio.clamp(1 - clip_low, 1 + clip_high)
    loss = torch.maximum(pg1, pg2)
    if clip_c is not None:
        loss = torch.where(adv < 0, torch.minimum(loss, -adv.detach() * clip_c), loss)
    return loss, (pg2 > pg1).float(), old_logp.detach() - logp


def grpo_loss(
    logits: torch.Tensor,
    old_logp: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
) -> torch.Tensor:
    """Group Relative Policy Optimization (GRPO) clipped surrogate loss."""
    logp = logits.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = labels.ne(-100)
    per_token_loss, _, _ = ppo_clip_per_token(logp, old_logp, advantages, clip_low, clip_high)
    return (per_token_loss * mask).sum() / mask.sum().clamp_min(1)


@CRITERIONS.register("rlsd", source="Self-Distill/RLSD-main/verl/workers/actor/dp_opsd_actor.py:426-579", force=True)
@CRITERIONS.register("rlsd_loss", source="Self-Distill/RLSD-main/verl/workers/actor/dp_opsd_actor.py:426-579", force=True)
class RLSDLoss(Criterion):
    """RLSD policy optimization loss over STGCA modulated advantages."""

    requires = ("student.labels", "rollout.advantage")
    supports_packing = False

    class Config(BaseConfig):
        clip_low: float = Field(default=0.2, ge=0, lt=1)
        clip_high: float = Field(default=0.28, ge=0)
        clip_c: float | None = Field(default=None, gt=1)
        temperature: float = 1.0

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx: LossContext) -> LossOutput:
        s_view = ctx.student_logits()
        if self.cfg.temperature != 1.0:
            s_view = s_view.with_temperature(self.cfg.temperature)
        logp = s_view.target_logp(ctx.batch["student.labels"])
        old = ctx.batch.get("rollout.old_logp", logp.detach())
        loss, clip, kl = ppo_clip_per_token(
            logp, old, ctx.batch["rollout.advantage"], self.cfg.clip_low, self.cfg.clip_high, self.cfg.clip_c
        )
        mask = ctx.supervised_mask()
        loss = loss * ctx.batch.get("rollout.is_weight", 1.0)
        return LossOutput(
            {"pg": LossTerm((loss * mask).sum(), "token_sum")},
            {"clipfrac": masked_mean(clip, mask).detach(), "approx_kl": masked_mean(kl, mask).detach()},
        )



__all__ = [
    "RLSDLoss",
    "PPOClip",
    "ppo_clip_per_token",
    "grpo_loss",
]
