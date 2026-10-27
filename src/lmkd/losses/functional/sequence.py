"""Target-aligned sequence log-probability utilities for preference objectives."""
from __future__ import annotations

import torch

from lmkd.models.logits_view import LogitsView


def sequence_logps(view: LogitsView, labels: torch.Tensor, *, average: bool,
                   chunk_size: int = 1024) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (sequence [R], token [R,T], supervised mask [R,T]).

    Matches causal_token_log_probs at target positions 1..T-1; source:
    human_alignment/mechanisms/training/distillation_losses.py:132-155.
    """
    if labels.ndim != 2 or tuple(labels.shape) != view.shape_bt:
        raise ValueError("labels must have shape [R,T] matching the logits view")
    mask = labels.ne(-100).clone()
    mask[:, 0] = False
    safe_labels = labels.masked_fill(~mask, -100)
    token = view.target_logp(safe_labels, chunk_size=chunk_size)
    token = token.masked_fill(~mask, 0.0)
    seq = token.sum(-1)
    if average:
        seq = seq / mask.sum(-1).clamp_min(1)
    return seq, token, mask


def group_view(x: torch.Tensor, n: int) -> torch.Tensor:
    """Reshape contiguous response rows [R,...] into prompt groups [R/n,n,...]."""
    if n < 2 or x.shape[0] % n:
        raise ValueError(f"row count {x.shape[0]} must be divisible by group size {n} >= 2")
    return x.reshape(x.shape[0] // n, n, *x.shape[1:])


def pairwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select chosen (index 0) and rejected (index 1) from contiguous pairs."""
    grouped = group_view(x, 2)
    return grouped[:, 0], grouped[:, 1]


def masked_seq_mean(token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean supervised token values for each sequence, with zero for empty rows."""
    if token.shape != mask.shape:
        raise ValueError("token and mask shapes must match")
    return (token * mask).sum(-1) / mask.sum(-1).clamp_min(1)
