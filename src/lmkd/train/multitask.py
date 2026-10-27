"""Local GradMag computations from IMPACT/code/alm_multitask.py:55-108."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

GRADMAG_MODES = frozenset({"approx_gradmag", "approx_gradmag_preserve_mag"})


def last_block_params(rt: Any) -> list[nn.Parameter]:
    """Trainable parameters of the student's last transformer block, including LoRA adapters."""
    layers = rt.student_adapter.layers(rt.student_model)
    if not layers:
        raise ValueError("GradMag needs a student transformer block")
    params = [p for p in layers[-1].parameters() if p.requires_grad]
    if not params:
        raise ValueError("GradMag needs trainable parameters in the last student block")
    return params


def global_grad_norm(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    """Global L2 norm with the reference's 1e-12 stabilizer."""
    return torch.sqrt(torch.stack([g.pow(2).sum() for g in grads]).sum() + 1e-12)


def gradmag_weights(task_gradients: Sequence[Sequence[torch.Tensor]], mode: str,
                    eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse task gradient norms; preserve_mag makes weights sum to one."""
    if mode not in GRADMAG_MODES:
        raise ValueError(f"unknown GradMag mode {mode!r}")
    if not task_gradients:
        raise ValueError("GradMag needs at least one task")
    norms = torch.stack([global_grad_norm(grads) for grads in task_gradients])
    weights = 1.0 / (norms + eps)
    if mode == "approx_gradmag_preserve_mag":
        weights = weights / weights.sum()
    return weights, norms


def task_grads(losses: Sequence[torch.Tensor], params: Sequence[nn.Parameter]) -> list[list[torch.Tensor]]:
    """Partial task gradients without writing .grad; unused parameters receive zeros."""
    if not params:
        raise ValueError("GradMag needs at least one gradient parameter")
    out = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True) if loss.requires_grad else [None] * len(params)
        out.append([g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)])
    return out
