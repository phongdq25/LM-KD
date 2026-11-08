"""Shared helper functions and reduction utilities for Self-Play Fine-Tuning baselines.

Supports canonical methods from self-play literature:
SPIN, T-SPIN, SPACE, SPPO, S-SPPO, and RSPO.
"""

from __future__ import annotations

import torch

from lmkd.core.batch import Batch
from lmkd.losses.base import LossTerm
from lmkd.models.logits_view import LogitsView


def _check_pairwise(batch: Batch, name: str) -> int:
    """Ensure batch contains paired responses (chosen at 0, rejected at 1)."""
    n = int(batch["group.size"][0]) if "group.size" in batch else 2
    valid = batch.get("group.valid")
    if n != 2 or (valid is not None and not valid.all()):
        raise ValueError(f"{name.upper()} requires pairwise chosen and rejected responses (group.size=2)")
    return n


def _reduction_term(loss: torch.Tensor, reduction: str, n: int = 2) -> LossTerm:
    """Return LossTerm normalized according to reduction mode."""
    if reduction == "mean":
        return LossTerm(loss.mean(), "mean")
    if reduction == "seq_sum":
        return LossTerm(loss.sum() * n, "seq_sum")
    raise ValueError(f"Unsupported reduction: {reduction}")


def pairwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select chosen (index 0) and rejected (index 1) from contiguous pairs."""
    if x.shape[0] % 2 != 0:
        raise ValueError(f"row count {x.shape[0]} must be divisible by 2 for pairwise operations")
    grouped = x.reshape(x.shape[0] // 2, 2, *x.shape[1:])
    return grouped[:, 0], grouped[:, 1]


def triplet(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select chosen (index 0), rejected (index 1), and proto (index 2) from contiguous triplets."""
    if x.shape[0] % 3 != 0:
        raise ValueError(f"row count {x.shape[0]} must be divisible by 3 for triplet operations")
    grouped = x.reshape(x.shape[0] // 3, 3, *x.shape[1:])
    return grouped[:, 0], grouped[:, 1], grouped[:, 2]


def sequence_logps(
    view: LogitsView,
    labels: torch.Tensor,
    *,
    average: bool = False,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (sequence [R], token [R,T], supervised mask [R,T]).

    Computes token log probabilities for target tokens (labels != -100)
    using chunked logits evaluation for memory efficiency.
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
