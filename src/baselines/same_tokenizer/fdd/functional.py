"""Functional operations for Feature Dynamics Distillation (FDD).

Source: Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def soft_label_distill_loss(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft label distillation loss (KL divergence between student and teacher logits).

    Formula:
        p_t = softmax(t_logits / T)
        log_p_s = log_softmax(s_logits / T)
        loss = T^2 * KL(p_t || p_s)
    """
    s_logp = F.log_softmax(s_logits / temperature, dim=-1)
    t_prob = F.softmax(t_logits / temperature, dim=-1)
    kl = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=-1) * (temperature**2)
    if mask is not None:
        denom = mask.sum().clamp_min(1.0)
        return (kl * mask).sum() / denom
    return kl.mean()


def _project_opt_if_needed(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    """Project hidden states through OPT decoder.project_out if present."""
    if model is None:
        return hidden
    curr = model
    for _ in range(5):
        decoder = getattr(curr, "decoder", None)
        if decoder is not None and getattr(decoder, "project_out", None) is not None:
            proj = decoder.project_out
            if callable(proj):
                return proj(hidden)
        if hasattr(curr, "module"):
            curr = curr.module
        elif hasattr(curr, "model"):
            curr = curr.model
        else:
            break
    return hidden


def compute_fdd_loss(
    s_hiddens: list[torch.Tensor],
    t_hiddens: list[torch.Tensor],
    mask: torch.Tensor | None,
    s_head: Any,
    t_head: Any,
    temperature: float = 1.0,
    w_traj: float = 1.0,
    w_der: float = 1.0,
    s_model: Any = None,
    t_model: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Feature Dynamics Distillation (FDD) loss across mapped layers.

    Args:
        s_hiddens: List of student hidden states [B, T, D_s] for each mapped layer.
        t_hiddens: List of teacher hidden states [B, T, D_t] for each mapped layer.
        mask: Supervised token mask [B, T] (1 = valid token, 0 = ignore/pad).
        s_head: Student lm_head projection module.
        t_head: Teacher lm_head projection module.
        temperature: Distillation temperature.
        w_traj: Weight for trajectory loss term.
        w_der: Weight for first derivative (delta) loss term.
        s_model: Optional student model (for OPT decoder.project_out check).
        t_model: Optional teacher model (for OPT decoder.project_out check).

    Returns:
        tuple (total_loss, traj_loss, der_loss)
    """
    if len(s_hiddens) != len(t_hiddens):
        raise ValueError(f"s_hiddens ({len(s_hiddens)}) and t_hiddens ({len(t_hiddens)}) length mismatch")
    if not s_hiddens:
        raise ValueError("s_hiddens must not be empty")

    device = s_hiddens[0].device
    traj_loss = torch.tensor(0.0, device=device)
    der_loss = torch.tensor(0.0, device=device)
    pre_s_hidden_logs: torch.Tensor | None = None
    pre_t_hidden_logs: torch.Tensor | None = None
    i = 0

    s_head_dtype = next(s_head.parameters()).dtype if hasattr(s_head, "parameters") else s_hiddens[0].dtype
    t_head_dtype = next(t_head.parameters()).dtype if hasattr(t_head, "parameters") else t_hiddens[0].dtype

    for s_hidden, t_hidden in zip(s_hiddens, t_hiddens):
        s_h = _project_opt_if_needed(s_model, s_hidden)
        t_h = _project_opt_if_needed(t_model, t_hidden)

        s_hidden_logits = s_head(s_h.to(s_head_dtype))
        with torch.no_grad():
            t_hidden_logits = t_head(t_h.to(t_head_dtype))

        traj_loss = traj_loss + soft_label_distill_loss(s_hidden_logits, t_hidden_logits, mask, temperature)

        s_hidden_logs = F.log_softmax(s_hidden_logits, dim=-1)
        with torch.no_grad():
            t_hidden_logs = F.log_softmax(t_hidden_logits, dim=-1)

        if i > 0:
            assert pre_s_hidden_logs is not None and pre_t_hidden_logs is not None
            delta_hidden_student = s_hidden_logs - pre_s_hidden_logs
            delta_hidden_teacher = t_hidden_logs - pre_t_hidden_logs

            cos_sim = F.cosine_similarity(delta_hidden_student, delta_hidden_teacher, dim=-1, eps=1e-5)
            cos_sim_loss = 1.0 - cos_sim
            if mask is not None:
                denom = mask.sum().clamp_min(1.0)
                cos_sim_loss = (cos_sim_loss * mask).sum() / denom
            else:
                cos_sim_loss = cos_sim_loss.mean()

            der_loss = der_loss + cos_sim_loss

        pre_s_hidden_logs = s_hidden_logs
        pre_t_hidden_logs = t_hidden_logs
        i += 1

    avg_traj = traj_loss / max(i, 1)
    avg_der = der_loss / (i - 1) if i > 1 else torch.tensor(0.0, device=device)
    total_loss = w_traj * avg_traj + w_der * avg_der

    return total_loss, avg_traj, avg_der

