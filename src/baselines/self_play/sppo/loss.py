"""SPPO: Self-Play Preference Optimization (Wu et al., ICML 2024).

SPPO frames preference learning as a two-player zero-sum game, updating the policy via
iterative soft win-rate regression against reference models:
  logits_w = log pi(y_w) - log pi_ref(y_w)
  logits_l = log pi(y_l) - log pi_ref(y_l)
  loss_w = (logits_w - (1 / beta) * (P_win - 0.5))^2
  loss_l = (logits_l - (1 / beta) * (P_lose - 0.5))^2
  L_SPPO = (loss_w + loss_l) / 2

"""

from __future__ import annotations

from typing import ClassVar

import torch

from baselines.self_play.common import _check_pairwise, _reduction_term, pairwise, sequence_logps
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def sppo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    chosen_probs_win: torch.Tensor | None = None,
    chosen_probs_lose: torch.Tensor | None = None,
    *,
    beta: float = 0.001,
    loss_type: str = "sppo",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SPPO squared margin regression loss.

    Args:
        policy_chosen_logps: Log probabilities of policy for winner y_w.
        policy_rejected_logps: Log probabilities of policy for loser y_l.
        reference_chosen_logps: Log probabilities of reference model for winner y_w.
        reference_rejected_logps: Log probabilities of reference model for loser y_l.
        chosen_probs_win: Soft win rate of winner against candidate set (P_win).
        chosen_probs_lose: Soft win rate of loser against candidate set (P_lose).
        beta: Learning parameter (eta = 1/beta).
        loss_type: "sppo" (separate win/lose probabilities) or "sppo_single" (pairwise probability).

    Returns:
        losses: Per-pair SPPO loss.
        chosen_rewards: Detached reward beta * logits_w.
        rejected_rewards: Detached reward beta * logits_l.
    """
    logits_w = policy_chosen_logps - reference_chosen_logps
    logits_l = policy_rejected_logps - reference_rejected_logps

    if chosen_probs_win is None:
        chosen_probs_win = torch.full_like(logits_w, 0.75)
    else:
        chosen_probs_win = chosen_probs_win.to(device=logits_w.device, dtype=logits_w.dtype)

    if chosen_probs_lose is None:
        chosen_probs_lose = torch.full_like(logits_l, 0.25)
    else:
        chosen_probs_lose = chosen_probs_lose.to(device=logits_l.device, dtype=logits_l.dtype)

    inv_beta = 1.0 / beta
    if loss_type == "sppo":
        loss_w = (logits_w - inv_beta * (chosen_probs_win - 0.5)) ** 2
        loss_l = (logits_l - inv_beta * (chosen_probs_lose - 0.5)) ** 2
        losses = (loss_w + loss_l) / 2.0
    elif loss_type == "sppo_single":
        loss_w = (logits_w - inv_beta * (chosen_probs_win - 0.5)) ** 2
        loss_l = (logits_l + inv_beta * (chosen_probs_win - 0.5)) ** 2
        losses = (loss_w + loss_l) / 2.0
    else:
        raise ValueError(f"Unknown SPPO loss_type: {loss_type}. Must be 'sppo' or 'sppo_single'")

    chosen_rewards = (beta * logits_w).detach()
    rejected_rewards = (beta * logits_l).detach()
    return losses, chosen_rewards, rejected_rewards


@CRITERIONS.register("sppo")
class SPPO(Criterion):
    """Self-Play Preference Optimization (SPPO) criterion."""

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        beta: float = 0.001
        loss_type: str = "sppo"
        default_win_prob: float = 0.75
        default_lose_prob: float = 0.25
        average_log_prob: bool = False
        reduction: str = "mean"
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="sampled_logp"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        n = _check_pairwise(ctx.batch, "sppo")
        if ctx.teacher is None or ctx.teacher.token_logp is None:
            raise ValueError("SPPO requires a teacher/reference model providing sampled_logp signal")

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

        p_win = None
        p_lose = None
        for k_win in ("pref.chosen_probs_win", "chosen_probs_win", "pref.win_prob"):
            if k_win in ctx.batch:
                val = ctx.batch[k_win]
                p_win = val[::2] if val.shape[0] == seq.shape[0] else val
                break
        for k_lose in ("pref.chosen_probs_lose", "chosen_probs_lose", "pref.lose_prob"):
            if k_lose in ctx.batch:
                val = ctx.batch[k_lose]
                p_lose = val[::2] if val.shape[0] == seq.shape[0] else val
                break
        if p_win is None and "pref.chosen_probs" in ctx.batch:
            val = ctx.batch["pref.chosen_probs"]
            p_win = val[::2] if val.shape[0] == seq.shape[0] else val
            p_lose = 1.0 - p_win
        if p_win is None:
            p_win = torch.full_like(pol_chosen, self.cfg.default_win_prob)
        if p_lose is None:
            p_lose = torch.full_like(pol_rejected, self.cfg.default_lose_prob)

        losses, chosen_rewards, rejected_rewards = sppo_loss(
            pol_chosen,
            pol_rejected,
            ref_chosen,
            ref_rejected,
            chosen_probs_win=p_win,
            chosen_probs_lose=p_lose,
            beta=self.cfg.beta,
            loss_type=self.cfg.loss_type,
        )

        term = _reduction_term(losses, self.cfg.reduction, n)
        metrics = {
            "loss": losses.mean().detach(),
            "reward_chosen": chosen_rewards.mean(),
            "reward_rejected": rejected_rewards.mean(),
            "margin": (chosen_rewards - rejected_rewards).mean(),
        }
        return LossOutput({"sppo": term}, metrics=metrics)
