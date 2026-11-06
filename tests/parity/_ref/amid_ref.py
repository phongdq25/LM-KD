"""Exact reference code for AMID distillation loss as provided by the user.

Used only for numerical parity testing in tests/parity/test_amid_parity.py.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def amid_ref(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    no_model_batch: dict[str, Any],
    alpha: float = 0.5,
    lam: float = 0.5,
    div_name: str = "fkl",
    div_order: str = "pr",
    ab_alpha: float = 0.2,
    ab_beta: float = 0.7,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Exact verbatim reference implementation of AMID."""
    if temperature != 1.0:
        logits = logits / temperature
        teacher_logits = teacher_logits / temperature

    # Teacher and Student distribution
    p = F.softmax(teacher_logits, dim=-1)
    q = F.softmax(logits, dim=-1)
    logp = F.log_softmax(teacher_logits, dim=-1)
    logq = F.log_softmax(logits, dim=-1)

    # Basic
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)

    # Assistant distribution
    if lam <= 0.0:
        r = q
        logr = F.log_softmax(logits, dim=-1)
    elif lam >= 1.0:
        r = p
        logr = F.log_softmax(teacher_logits, dim=-1)
    else:
        if alpha >= 1.0:
            logr_unnorm = lam * logp + (1.0 - lam) * logq
            r = F.softmax(logr_unnorm, dim=-1)
            logr = F.log_softmax(logr_unnorm, dim=-1)
        else:
            t1 = math.log(lam) + 0.5 * (1.0 - alpha) * logp
            t2 = math.log(1.0 - lam) + 0.5 * (1.0 - alpha) * logq

            logr_unnorm = 2.0 / (1.0 - alpha) * torch.logaddexp(t1, t2)
            r = F.softmax(logr_unnorm, dim=-1)
            logr = F.log_softmax(logr_unnorm, dim=-1)

    # Divergence
    if div_name == "fkl":
        if div_order == "pr":
            prod_probs = torch.masked_fill(p * (logp - logr), inf_mask, 0)
        elif div_order == "qr":
            prod_probs = torch.masked_fill(q * (logq - logr), inf_mask, 0)
        elif div_order == "rp":
            prod_probs = torch.masked_fill(r * (logr - logp), inf_mask, 0)
        elif div_order == "rq":
            prod_probs = torch.masked_fill(r * (logr - logq), inf_mask, 0)
        else:
            raise ValueError(f"Check div_order: {div_order}")
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        return distil_loss

    elif div_name == "ab":
        apb = ab_alpha + ab_beta

        if div_order == "pr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logp + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logp, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
        elif div_order == "qr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logq + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(ab_alpha * logq, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(ab_beta * logr, dim=-1))
        elif div_order == "rp":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logp, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(ab_alpha * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(ab_beta * logp, dim=-1))
        elif div_order == "rq":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logq, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(ab_alpha * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(ab_beta * logq, dim=-1))
        else:
            raise ValueError(f"Check div_order: {div_order}")
        divergence = -(term1 - term2 - term3) / (ab_alpha * ab_beta)

        safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)

        masked_sum = (safe_divergence * mask).sum()
        mask_sum = mask.sum()
        loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum
        return loss

    else:
        raise ValueError(f"Check div_name: {div_name}")

