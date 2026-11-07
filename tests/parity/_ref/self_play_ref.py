"""Reference implementations for self-play fine-tuning baselines:

- preference_loss (SPIN)
- tspin_loss (T-SPIN)
- space_loss (SPACE)
- orig_sppo_loss (SPPO)
- orig_sspppo_loss (S-SPPO)
- orig_rspo_reversekl_loss (RSPO)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """DPO loss (SPIN's objective)."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def tspin_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    policy_proto_logps: torch.FloatTensor,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """T-SPIN: -logσ(α(log π(y) - log π(y'))) - β·logσ(α(log π(y') - log π(y_0)))."""
    u = alpha * (policy_chosen_logps - policy_rejected_logps)
    v = alpha * (policy_rejected_logps - policy_proto_logps)
    losses = -F.logsigmoid(u) - beta * F.logsigmoid(v)

    chosen_rewards = (alpha * policy_chosen_logps).detach()
    rejected_rewards = (alpha * policy_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def space_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    mu: float = 1.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """SPACE NCE loss."""
    log_mu = torch.log(torch.as_tensor(mu, dtype=policy_chosen_logps.dtype, device=policy_chosen_logps.device))

    log_ratio_real = policy_chosen_logps - reference_chosen_logps
    real_loss = F.logsigmoid(log_ratio_real - log_mu)

    log_ratio_gen = reference_rejected_logps - policy_rejected_logps
    gen_loss = mu * F.logsigmoid(log_ratio_gen + log_mu)

    losses = -(real_loss + gen_loss)

    chosen_rewards = log_ratio_real.detach()
    rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def orig_sppo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    chosen_probs_win,
    chosen_probs_lose,
    beta=0.001,
):
    logits_w = policy_chosen_logps - reference_chosen_logps
    logits_l = policy_rejected_logps - reference_rejected_logps
    loss_w = (logits_w - (1 / beta) * (chosen_probs_win - 0.5)) ** 2
    loss_l = (logits_l - (1 / beta) * (chosen_probs_lose - 0.5)) ** 2
    return (loss_w + loss_l) / 2


def orig_sspppo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    chosen_probs_win,
    chosen_probs_lose,
    sim_score,
    beta=0.001,
    h_chosen=None,
    h_rejected=None,
    lam=0.0,
):
    logits_w = policy_chosen_logps - reference_chosen_logps
    logits_l = policy_rejected_logps - reference_rejected_logps
    calib = (1.0 - sim_score).to(policy_chosen_logps.device)
    p_c_win = 0.5 + calib * (chosen_probs_win - 0.5)
    p_c_lose = 0.5 + calib * (chosen_probs_lose - 0.5)
    loss_w = (logits_w - (1 / beta) * (p_c_win - 0.5)) ** 2
    loss_l = (logits_l - (1 / beta) * (p_c_lose - 0.5)) ** 2
    losses = (loss_w + loss_l) / 2
    if h_chosen is not None and h_rejected is not None and lam != 0.0:
        repulsion = (h_chosen * h_rejected).sum(dim=-1)
        losses = losses + lam * repulsion
    return losses


def orig_rspo_reversekl_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    base_chosen_logps,
    base_rejected_logps,
    chosen_probs_win,
    chosen_probs_lose,
    beta=0.001,
    reg_coef=0.5,
):
    logits_w = policy_chosen_logps - reference_chosen_logps
    logits_l = policy_rejected_logps - reference_rejected_logps
    loss_w = (logits_w - (1 / beta) * (chosen_probs_win - 0.5)) ** 2
    loss_l = (logits_l - (1 / beta) * (chosen_probs_lose - 0.5)) ** 2
    losses = (loss_w + loss_l) / 2

    logits_diff_w = policy_chosen_logps - base_chosen_logps
    logits_diff_l = policy_rejected_logps - base_rejected_logps
    reg_loss = (logits_diff_w ** 2).mean() + (logits_diff_l ** 2).mean()

    total_loss = losses + reg_loss * reg_coef
    return total_loss, reg_loss
