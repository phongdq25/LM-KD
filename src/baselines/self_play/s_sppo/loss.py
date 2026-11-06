"""S-SPPO: Semantic-Calibrated Self-Play Preference Optimization.

S-SPPO calibrates soft win-rate targets using the semantic similarity between the two
responses, preventing artificial margin over-penalization when both responses are
semantically close:
  calib = 1 - sim_score
  p_c_win = 0.5 + calib * (P_win - 0.5)
  p_c_lose = 0.5 + calib * (P_lose - 0.5)
  L_base = ((logits_w - (1 / beta) * (p_c_win - 0.5))^2 + (logits_l - (1 / beta) * (p_c_lose - 0.5))^2) / 2
  L_S-SPPO = L_base + lam * <h_w, h_l>

Source: APSP/SPPO/sppo/trainer.py:914-928
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


def sspppo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    sim_score: torch.Tensor | None = None,
    chosen_probs_win: torch.Tensor | None = None,
    chosen_probs_lose: torch.Tensor | None = None,
    h_chosen: torch.Tensor | None = None,
    h_rejected: torch.Tensor | None = None,
    *,
    beta: float = 0.001,
    lam: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Semantic-Calibrated SPPO loss function.

    Args:
        policy_chosen_logps: Log probabilities of policy for winner y_w.
        policy_rejected_logps: Log probabilities of policy for loser y_l.
        reference_chosen_logps: Log probabilities of reference model for winner y_w.
        reference_rejected_logps: Log probabilities of reference model for loser y_l.
        sim_score: Semantic similarity cos(phi(y_w), phi(y_l)) in [0, 1].
        chosen_probs_win: Soft win rate of winner against candidate set.
        chosen_probs_lose: Soft win rate of loser against candidate set.
        h_chosen: Optional representation vector for chosen response.
        h_rejected: Optional representation vector for rejected response.
        beta: Inverse temperature parameter.
        lam: Weight for hidden representation repulsion term.

    Returns:
        losses: Per-pair calibrated SPPO loss.
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

    if sim_score is None:
        sim_score = torch.zeros_like(logits_w)
    else:
        sim_score = sim_score.to(device=logits_w.device, dtype=logits_w.dtype)

    calib = 1.0 - sim_score
    p_c_win = 0.5 + calib * (chosen_probs_win - 0.5)
    p_c_lose = 0.5 + calib * (chosen_probs_lose - 0.5)

    inv_beta = 1.0 / beta
    loss_w = (logits_w - inv_beta * (p_c_win - 0.5)) ** 2
    loss_l = (logits_l - inv_beta * (p_c_lose - 0.5)) ** 2
    losses = (loss_w + loss_l) / 2.0

    if h_chosen is not None and h_rejected is not None and lam != 0.0:
        repulsion = (h_chosen * h_rejected).sum(dim=-1)
        losses = losses + lam * repulsion

    chosen_rewards = (beta * logits_w).detach()
    rejected_rewards = (beta * logits_l).detach()
    return losses, chosen_rewards, rejected_rewards


@CRITERIONS.register("s_sppo", source="APSP/SPPO/sppo/trainer.py:914")
class SSPPO(Criterion):
    """Semantic-Calibrated Self-Play Preference Optimization (S-SPPO) criterion."""

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        beta: float = 0.001
        lam: float = 0.0
        default_win_prob: float = 0.75
        default_lose_prob: float = 0.25
        use_hidden_repulsion: bool = False
        average_log_prob: bool = False
        reduction: str = "mean"
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        use_hidden = self.cfg.use_hidden_repulsion or self.cfg.lam != 0.0
        return FeatureRequest(
            student=SideRequest(logits="view", hidden="last" if use_hidden else "none"),
            teacher_signal=TeacherSignalSpec(kind="sampled_logp"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        n = _check_pairwise(ctx.batch, "sspppo")
        if ctx.teacher is None or ctx.teacher.token_logp is None:
            raise ValueError("S-SPPO requires a teacher/reference model providing sampled_logp signal")

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

        sim_score = None
        for k_sim in ("pref.sim_score", "sim_score"):
            if k_sim in ctx.batch:
                val = ctx.batch[k_sim]
                sim_score = val[::2] if val.shape[0] == seq.shape[0] else val
                break

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

        h_chosen, h_rejected = None, None
        if (self.cfg.use_hidden_repulsion or self.cfg.lam != 0.0) and hasattr(ctx.student, "last_hidden"):
            # Pool last hidden states by supervised mask or mean
            h = ctx.student.last_hidden  # [B, T, D]
            denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
            pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
            h_chosen, h_rejected = pairwise(pooled)

        losses, chosen_rewards, rejected_rewards = sspppo_loss(
            pol_chosen,
            pol_rejected,
            ref_chosen,
            ref_rejected,
            sim_score=sim_score,
            chosen_probs_win=p_win,
            chosen_probs_lose=p_lose,
            h_chosen=h_chosen,
            h_rejected=h_rejected,
            beta=self.cfg.beta,
            lam=self.cfg.lam,
        )

        term = _reduction_term(losses, self.cfg.reduction, n)
        metrics = {
            "loss": losses.mean().detach(),
            "reward_chosen": chosen_rewards.mean(),
            "reward_rejected": rejected_rewards.mean(),
            "margin": (chosen_rewards - rejected_rewards).mean(),
        }
        if sim_score is not None:
            metrics["sim_score"] = sim_score.mean().detach()
        return LossOutput({"s_sppo": term}, metrics=metrics)


S_SPPO = SSPPO
s_sppo_loss = sspppo_loss
