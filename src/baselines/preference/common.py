"""Shared helper functions and reduction utilities for preference distillation baselines."""

from __future__ import annotations

import torch

from lmkd.core.batch import Batch
from lmkd.losses.base import LossTerm
from lmkd.models.logits_view import LogitsView


def _score_matrix(scores: torch.Tensor, name: str) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError(f"{name} must have shape [batch, responses] with at least two responses")
    if not torch.isfinite(scores).all():
        raise ValueError(f"{name} contains a non-finite value")
    return scores


def plackett_luce_log_probability(scores: torch.Tensor, rankings: torch.Tensor) -> torch.Tensor:
    """Return the log probability of each ranking under Plackett-Luce scores."""
    _score_matrix(scores, "scores")
    if rankings.shape != scores.shape:
        raise ValueError("rankings must have the same shape as scores")
    ordered = scores.gather(1, rankings.long())
    denominators = torch.logcumsumexp(ordered.flip(-1), dim=-1).flip(-1)
    return (ordered - denominators).sum(dim=-1)


def _check_pairwise(batch: Batch, name: str) -> int:
    n = int(batch["group.size"][0])
    valid = batch.get("group.valid")
    if n != 2 or (valid is not None and not valid.all()):
        raise ValueError(f"{name.upper()} requires exactly chosen and rejected responses")
    return n


def _reduction_term(loss: torch.Tensor, reduction: str, n: int) -> LossTerm:
    if reduction == "mean":
        return LossTerm(loss.mean(), "mean")
    if reduction == "seq_sum":
        return LossTerm(loss.sum() * n, "seq_sum")
    raise ValueError(f"Unsupported reduction: {reduction}")


def pairwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select chosen (index 0) and rejected (index 1) from contiguous pairs."""
    if x.shape[0] % 2 != 0:
        raise ValueError(f"row count {x.shape[0]} must be divisible by 2")
    grouped = x.reshape(x.shape[0] // 2, 2, *x.shape[1:])
    return grouped[:, 0], grouped[:, 1]


def sequence_logps(
    view: LogitsView, labels: torch.Tensor, *, average: bool, chunk_size: int = 1024
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
