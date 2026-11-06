"""RSPO: Regularized Self-Play Preference Optimization.

RSPO adds divergence regularization to the base SPPO formulation to anchor the self-play
progression against drift or over-optimization:
  L_SPPO = ((logits_w - (1 / beta) * (P_win - 0.5))^2 + (logits_l - (1 / beta) * (P_lose - 0.5))^2) / 2
  L_reg = mean((log pi(y_w) - log pi_base(y_w))^2) + mean((log pi(y_l) - log pi_base(y_l))^2)   [reverse-KL]
  L_RSPO = L_SPPO + reg_coef * L_reg

Source: APSP/RSPO/sppo/trainer_reg.py:850-950
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


def rspo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    base_chosen_logps: torch.Tensor | None = None,
    base_rejected_logps: torch.Tensor | None = None,
    chosen_probs_win: torch.Tensor | None = None,
    chosen_probs_lose: torch.Tensor | None = None,
    *,
    beta: float = 0.001,
    reg_coef: float = 0.5,
    loss_type: str = "sppo_reversekl",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RSPO regularized preference loss.

    Args:
        policy_chosen_logps: Log probabilities of policy for winner y_w.
        policy_rejected_logps: Log probabilities of policy for loser y_l.
        reference_chosen_logps: Log probabilities of reference model (iter t-1).
        reference_rejected_logps: Log probabilities of reference model (iter t-1).
        base_chosen_logps: Log probabilities of base model (iter 0 / SFT).
        base_rejected_logps: Log probabilities of base model (iter 0 / SFT).
        chosen_probs_win: Win rate of winner.
        chosen_probs_lose: Win rate of loser.
        beta: Learning parameter (eta = 1/beta).
        reg_coef: Weight multiplier on regularization loss term.
        loss_type: Regularization variant ("sppo_reversekl", "sppo_entropy", "sppo_forwardreverse").

    Returns:
        losses: Per-pair total RSPO loss.
        chosen_rewards: Detached reward beta * logits_w.
        rejected_rewards: Detached reward beta * logits_l.
        reg_loss: Detached scalar regularization term.
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
    loss_w = (logits_w - inv_beta * (chosen_probs_win - 0.5)) ** 2
    loss_l = (logits_l - inv_beta * (chosen_probs_lose - 0.5)) ** 2
    sppo_losses = (loss_w + loss_l) / 2.0

    if base_chosen_logps is None:
        base_chosen_logps = reference_chosen_logps
    if base_rejected_logps is None:
        base_rejected_logps = reference_rejected_logps

    logits_diff_w = policy_chosen_logps - base_chosen_logps
    logits_diff_l = policy_rejected_logps - base_rejected_logps

    if loss_type == "sppo_reversekl":
        reg_loss = (logits_diff_w**2).mean() + (logits_diff_l**2).mean()
    elif loss_type == "sppo_reverseklnoent":
        reg_loss = (
            -(policy_chosen_logps * base_chosen_logps).mean() - (policy_rejected_logps * base_rejected_logps).mean()
        )
    elif loss_type == "sppo_entropy":
        reg_loss = (policy_chosen_logps**2).mean() + (policy_rejected_logps**2).mean()
    elif loss_type == "sppo_forwardreverse":
        r_cap = 10.0
        pi_ref_over_pi_w = (-logits_diff_w).clamp(max=r_cap).exp()
        pi_ref_over_pi_l = (-logits_diff_l).clamp(max=r_cap).exp()
        forward = pi_ref_over_pi_w.mean() + pi_ref_over_pi_l.mean()
        reverse = (logits_diff_w**2).mean() + (logits_diff_l**2).mean()
        reg_loss = forward + reverse
    elif loss_type.startswith("sppo_forward1reverse"):
        reverse_coef = float(loss_type.replace("sppo_forward1reverse", ""))
        r_cap = 10.0
        pi_ref_over_pi_w = (-logits_diff_w).clamp(max=r_cap).exp()
        pi_ref_over_pi_l = (-logits_diff_l).clamp(max=r_cap).exp()
        forward = pi_ref_over_pi_w.mean() + pi_ref_over_pi_l.mean()
        reverse = (logits_diff_w**2).mean() + (logits_diff_l**2).mean()
        reg_loss = forward + reverse_coef * reverse
    elif loss_type in ("sppo", "sppo_single"):
        reg_loss = torch.zeros((), device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)
    else:
        raise ValueError(f"Unknown RSPO loss_type: {loss_type}")

    total_losses = sppo_losses + reg_coef * reg_loss
    chosen_rewards = (beta * logits_w).detach()
    rejected_rewards = (beta * logits_l).detach()
    return total_losses, chosen_rewards, rejected_rewards, reg_loss.detach()


@CRITERIONS.register("rspo", source="APSP/RSPO/sppo/trainer_reg.py:850")
class RSPO(Criterion):
    """Regularized Self-Play Preference Optimization (RSPO) criterion."""

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        beta: float = 0.001
        reg_coef: float = 0.5
        loss_type: str = "sppo_reversekl"
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
        n = _check_pairwise(ctx.batch, "rspo")
        if ctx.teacher is None or ctx.teacher.token_logp is None:
            raise ValueError("RSPO requires a teacher/reference model providing sampled_logp signal")

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

        base_chosen, base_rejected = None, None
        if "cache.base_logp" in ctx.batch:
            base_chosen, base_rejected = pairwise(ctx.batch["cache.base_logp"])

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
        if p_win is None:
            p_win = torch.full_like(pol_chosen, self.cfg.default_win_prob)
        if p_lose is None:
            p_lose = torch.full_like(pol_rejected, self.cfg.default_lose_prob)

        losses, chosen_rewards, rejected_rewards, reg_loss = rspo_loss(
            pol_chosen,
            pol_rejected,
            ref_chosen,
            ref_rejected,
            base_chosen_logps=base_chosen,
            base_rejected_logps=base_rejected,
            chosen_probs_win=p_win,
            chosen_probs_lose=p_lose,
            beta=self.cfg.beta,
            reg_coef=self.cfg.reg_coef,
            loss_type=self.cfg.loss_type,
        )

        term = _reduction_term(losses, self.cfg.reduction, n)
        metrics = {
            "loss": losses.mean().detach(),
            "reward_chosen": chosen_rewards.mean(),
            "reward_rejected": rejected_rewards.mean(),
            "reg_loss": reg_loss,
            "margin": (chosen_rewards - rejected_rewards).mean(),
        }
        return LossOutput({"rspo": term}, metrics=metrics)
