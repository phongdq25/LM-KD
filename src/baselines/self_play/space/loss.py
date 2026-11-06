"""SPACE: Self-Play via Noise Contrastive Estimation (Wu et al., 2024).

SPACE formulates self-play alignment as Noise Contrastive Estimation (NCE), treating the
reference/generator model as the noise distribution and learning an energy-based discriminator.

Objective:
  log_ratio_real = log pi(y) - log pi_ref(y)
  real_loss = logsigmoid(log_ratio_real - log(mu))
  log_ratio_gen = log pi_ref(y') - log pi(y')
  gen_loss = mu * logsigmoid(log_ratio_gen + log(mu))
  L_SPACE = -(real_loss + gen_loss)

"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from baselines.self_play.common import _check_pairwise, _reduction_term, pairwise, sequence_logps
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def space_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    mu: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SPACE Noise Contrastive Estimation loss.

    Args:
        policy_chosen_logps: Log probabilities of policy for real response y.
        policy_rejected_logps: Log probabilities of policy for generated response y'.
        reference_chosen_logps: Log probabilities of reference noise model for real response y.
        reference_rejected_logps: Log probabilities of reference noise model for generated response y'.
        mu: Noise contrastive ratio hyperparameter.

    Returns:
        losses: Per-pair SPACE NCE loss.
        chosen_rewards: Detached log ratio for real response.
        rejected_rewards: Detached log ratio for generated response.
    """
    log_mu = torch.log(torch.as_tensor(mu, dtype=policy_chosen_logps.dtype, device=policy_chosen_logps.device))

    log_ratio_real = policy_chosen_logps - reference_chosen_logps
    real_loss = F.logsigmoid(log_ratio_real - log_mu)

    log_ratio_gen = reference_rejected_logps - policy_rejected_logps
    gen_loss = mu * F.logsigmoid(log_ratio_gen + log_mu)

    losses = -(real_loss + gen_loss)

    chosen_rewards = log_ratio_real.detach()
    rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


@CRITERIONS.register("space")
class SPACE(Criterion):
    """Self-Play via Noise Contrastive Estimation (SPACE) criterion."""

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        mu: float = 1.0
        average_log_prob: bool = False
        reduction: str = "mean"
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="sampled_logp"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        n = _check_pairwise(ctx.batch, "space")
        if ctx.teacher is None or ctx.teacher.token_logp is None:
            raise ValueError("SPACE requires a reference/noise model providing sampled_logp signal")

        seq, _, mask = sequence_logps(
            ctx.student_logits(),
            ctx.batch["student.labels"],
            average=self.cfg.average_log_prob,
            chunk_size=self.cfg.chunk_size,
        )
        pol_chosen, pol_rejected = pairwise(seq)

        ref_seq = (ctx.teacher.token_logp * mask).sum(-1)
        if self.cfg.average_log_prob:
            ref_seq = ref_seq / mask.sum(-1).clamp_min(1.0)
        ref_chosen, ref_rejected = pairwise(ref_seq)

        losses, chosen_rewards, rejected_rewards = space_loss(
            pol_chosen,
            pol_rejected,
            ref_chosen,
            ref_rejected,
            mu=self.cfg.mu,
        )

        term = _reduction_term(losses, self.cfg.reduction, n)
        metrics = {
            "loss": losses.mean().detach(),
            "reward_chosen": chosen_rewards.mean(),
            "reward_rejected": rejected_rewards.mean(),
            "margin": (chosen_rewards - rejected_rewards).mean(),
        }
        return LossOutput({"space": term}, metrics=metrics)
