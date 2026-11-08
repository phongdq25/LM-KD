"""Contrastive distillation loss primitives (Tier 1: Pure PyTorch, zero framework dependencies)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["info_nce_loss"]


def info_nce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature: float = 0.07,
    bidirectional: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard InfoNCE / contrastive loss between query and key representations.

    Formula:
      q_norm = normalize(q, dim=-1)
      k_norm = normalize(k, dim=-1)
      scores = (q_norm @ k_norm.T) / temperature
      loss = cross_entropy(scores, arange(B))

    Args:
        q: Query representations [B, D].
        k: Key/target representations [B, D] (or [M, D] where diagonal matches q[:M]).
        temperature: Softmax scaling factor.
        bidirectional: If True, computes symmetric InfoNCE (q->k and k->q) averaged.
        reduction: "mean" | "sum" | "none".
    """
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError(f"Expected 2D tensors for q and k, got q={q.shape}, k={k.shape}")
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    B = q.shape[0]
    scores = torch.matmul(q_norm, k_norm.t()) / temperature
    targets = torch.arange(B, device=scores.device, dtype=torch.long)
    loss = F.cross_entropy(scores, targets, reduction=reduction)
    if bidirectional:
        loss_rev = F.cross_entropy(scores.t(), targets, reduction=reduction)
        loss = (loss + loss_rev) * 0.5
    return loss
