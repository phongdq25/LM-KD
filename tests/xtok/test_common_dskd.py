"""Unit tests for DSKD helpers in _common.py (T12)."""

import torch
import torch.nn as nn

from lmkd.losses.cross_tokenizer._common import DetachedLinear, view_from_target_hidden


def test_view_from_target_hidden():
    torch.manual_seed(42)
    B, T, D, V = 2, 7, 16, 25
    head = nn.Linear(D, V, bias=False)
    h_tgt = torch.randn(B, T, D)

    view = view_from_target_hidden(h_tgt, head)
    full = view.full()  # [B, T, V] target-aligned logits

    # For any target index t >= 1, view.full()[:, t] should equal head(h_tgt[:, t])
    expected = head(h_tgt)
    for t in range(1, T):
        assert torch.allclose(full[:, t], expected[:, t], atol=1e-5), f"Mismatch at target {t}"


def test_detached_linear_no_grad_to_head():
    torch.manual_seed(42)
    head = nn.Linear(16, 32, bias=True)
    det_linear = DetachedLinear(head)

    x = torch.randn(2, 5, 16, requires_grad=True)
    out = det_linear(x)
    loss = out.sum()
    loss.backward()

    # Input x should have gradients
    assert x.grad is not None
    assert torch.all(x.grad.abs() > 0)

    # Underlying head weight and bias should have NO gradient
    assert head.weight.grad is None
    if head.bias is not None:
        assert head.bias.grad is None
