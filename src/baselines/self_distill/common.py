"""Shared numerical primitives and distribution utilities for self-distillation baselines.

Provides:
- `masked_mean`: Numerically safe masked averaging.
- `add_tail`: SDPO residual probability bucket (allocates 1 - sum(P_k) to tail index).
- `renorm_topk`: Renormalizes log probabilities over a top-k subset.
- `effective_lambda`: Linear warmup and decay schedule for teacher advantage weighting.
"""

from __future__ import annotations

import torch


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Finite masked mean, safe for batches with zero supervised tokens."""
    return (value * mask).sum() / mask.sum().clamp_min(1)


def add_tail(logp: torch.Tensor) -> torch.Tensor:
    """Append SDPO residual probability bucket (core_algos.py:1112-1118).

    Guarantees proper simplex normalization over the top-k subset by allocating
    residual probability mass 1 - sum(P_k) to an explicit tail state.
    """
    log_mass = logp.logsumexp(-1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([logp, (-log_mass.expm1()).log()], -1)


def renorm_topk(logp: torch.Tensor) -> torch.Tensor:
    """Renormalize log probabilities over the top-k subset."""
    return logp - logp.logsumexp(-1, keepdim=True)


def effective_lambda(lam: float, step: int, warmup_steps: int = 0, decay_steps: int = 0) -> float:
    """Linear warmup followed by linear decay for teacher advantage weighting (RLSD)."""
    if warmup_steps > 0 and step < warmup_steps:
        return lam * step / warmup_steps
    if decay_steps > 0 and step >= warmup_steps:
        return lam * max(1 - (step - warmup_steps) / decay_steps, 0)
    return lam


__all__ = [
    "masked_mean",
    "add_tail",
    "renorm_topk",
    "effective_lambda",
]
