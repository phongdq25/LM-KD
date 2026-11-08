"""T-SPIN: Triplet Self-Play Fine-Tuning.

T-SPIN incorporates a triplet of responses:
  y: human / chosen target response
  y': current self-play rollout response
  y_0: base / prototype model response (proto anchor)

Objective:
L_TSPIN = -log sigmoid(alpha * (log pi(y) - log pi(y'))) - beta * log sigmoid(alpha * (log pi(y') - log pi(y_0)))

"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from baselines.self_play.common import _reduction_term, pairwise, sequence_logps, triplet
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.models.forward import FeatureRequest, SideRequest


def tspin_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    policy_proto_logps: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triplet SPIN loss function.

    Args:
        policy_chosen_logps: Log probabilities of policy for ground truth response y.
        policy_rejected_logps: Log probabilities of policy for self-play response y'.
        policy_proto_logps: Log probabilities of policy for base prototype response y_0.
        alpha: Temperature coefficient on log probability differences.
        beta: Weight on the historical progression anchor term (y' vs y_0).

    Returns:
        losses: Per-triplet T-SPIN loss.
        chosen_rewards: Detached reward alpha * policy_chosen.
        rejected_rewards: Detached reward alpha * policy_rejected.
    """
    u = alpha * (policy_chosen_logps - policy_rejected_logps)
    v = alpha * (policy_rejected_logps - policy_proto_logps)
    losses = -F.logsigmoid(u) - beta * F.logsigmoid(v)

    chosen_rewards = (alpha * policy_chosen_logps).detach()
    rejected_rewards = (alpha * policy_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


@CRITERIONS.register("tspin")
class TSPIN(Criterion):
    """Triplet Self-Play Fine-Tuning (T-SPIN) criterion."""

    needs_teacher: ClassVar[bool] = False

    class Config(BaseConfig):
        alpha: float = 1.0
        beta: float = 0.1
        average_log_prob: bool = False
        reduction: str = "mean"
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx: LossContext) -> LossOutput:
        n = int(ctx.batch["group.size"][0]) if "group.size" in ctx.batch else 2

        seq, _, _ = sequence_logps(
            ctx.student_logits(),
            ctx.batch["student.labels"],
            average=self.cfg.average_log_prob,
            chunk_size=self.cfg.chunk_size,
        )

        if n == 3:
            pol_chosen, pol_rejected, pol_proto = triplet(seq)
        elif n == 2:
            pol_chosen, pol_rejected = pairwise(seq)
            if "cache.proto_logp" in ctx.batch:
                pol_proto = ctx.batch["cache.proto_logp"]
            elif "proto.labels" in ctx.batch:
                # Pre-evaluated proto logp if available in cache
                pol_proto = pol_rejected
            else:
                pol_proto = pol_rejected
        else:
            raise ValueError(f"TSPIN expects group.size 2 or 3, got {n}")

        losses, chosen_rewards, rejected_rewards = tspin_loss(
            pol_chosen,
            pol_rejected,
            pol_proto,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
        )

        term = _reduction_term(losses, self.cfg.reduction, n)
        metrics = {
            "loss": losses.mean().detach(),
            "reward_chosen": chosen_rewards.mean(),
            "reward_rejected": rejected_rewards.mean(),
            "margin_u": (self.cfg.alpha * (pol_chosen - pol_rejected)).mean().detach(),
            "margin_v": (self.cfg.alpha * (pol_rejected - pol_proto)).mean().detach(),
        }
        return LossOutput({"tspin": term}, metrics=metrics)
