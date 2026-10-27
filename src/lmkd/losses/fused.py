"""Fused loss functions using Liger Kernel (T26, DESIGN §11)."""
from __future__ import annotations

import torch
import torch.nn as nn

from lmkd.models.logits_view import LogitsView


def liger_available() -> bool:
    """Return True if liger_kernel is installed and can be imported."""
    try:
        import liger_kernel  # noqa: F401

        return True
    except ImportError:
        return False


def _get_hidden_and_weight(
    view: LogitsView, lm_head: nn.Module | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract positional hidden states and lm_head weight matrix from a LogitsView."""
    h = view.hidden
    if h is None:
        raise ValueError("view must have hidden states")
    head = lm_head if lm_head is not None else getattr(view, "_head", None)
    if head is None or not hasattr(head, "weight"):
        raise ValueError("lm_head with weight tensor is required")
    w = head.weight
    if not isinstance(w, torch.Tensor):
        raise ValueError("lm_head.weight must be a Tensor")
    return h, w


def fused_nll(
    view: LogitsView,
    labels: torch.Tensor,
    mask: torch.Tensor,
    lm_head: nn.Module | None = None,
) -> torch.Tensor:
    """Compute fused negative log-likelihood via LigerFusedLinearCrossEntropyLoss.

    Returns the token_sum numerator. Target-aligned inputs are shifted:
    h[:, :-1] predicts labels[:, 1:].
    """
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    h, W = _get_hidden_and_weight(view, lm_head)
    valid = mask[:, 1:].bool()
    if not valid.any():
        return torch.zeros((), device=h.device, dtype=h.dtype, requires_grad=True)

    x = h[:, :-1][valid].to(dtype=W.dtype).contiguous()
    y = labels[:, 1:][valid].contiguous()

    loss_fn = LigerFusedLinearCrossEntropyLoss(reduction="sum", ignore_index=-100)
    return loss_fn(W, x, y)


def fused_jsd(
    s_view: LogitsView,
    t_view: LogitsView,
    labels: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    lm_head_s: nn.Module | None = None,
    lm_head_t: nn.Module | None = None,
) -> torch.Tensor:
    """Compute fused generalized JSD loss via LigerFusedLinearJSDLoss.

    Returns the token_sum numerator. Target-aligned inputs are shifted:
    h[:, :-1] predicts labels[:, 1:].
    """
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

    h_s, W_s = _get_hidden_and_weight(s_view, lm_head_s)
    h_t, W_t = _get_hidden_and_weight(t_view, lm_head_t)

    valid = mask[:, 1:].bool()
    if not valid.any():
        return torch.zeros((), device=h_s.device, dtype=h_s.dtype, requires_grad=True)

    x_s = h_s[:, :-1][valid].to(dtype=W_s.dtype).contiguous()
    x_t = h_t[:, :-1][valid].to(dtype=W_t.dtype).contiguous()
    y = labels[:, 1:][valid].contiguous()

    loss_fn = LigerFusedLinearJSDLoss(
        weight_hard_loss=0.0,
        weight_soft_loss=1.0,
        beta=beta,
        ignore_index=-100,
        temperature=temperature,
        compiled=False,
    )
    n_valid = (y != -100).sum()
    if n_valid == 0:
        return torch.zeros((), device=h_s.device, dtype=h_s.dtype, requires_grad=True)
    mean_loss = loss_fn(x_s, W_s, x_t, W_t, y)
    return mean_loss * n_valid
