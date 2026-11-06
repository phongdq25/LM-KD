"""GPU tests for Liger fused loss parity (T26)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("liger_kernel")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")]

from lmkd.losses.functional.divergences import generalized_jsd  # noqa: E402
from lmkd.losses.fused import fused_jsd, fused_nll  # noqa: E402
from lmkd.models.logits_view import LogitsView  # noqa: E402


def test_fused_nll_parity():
    """Hidden and random lm_head (V=1000): fused numerator equals -(target_logp*mask).sum() (rtol 1e-3, bf16);
    hidden gradient matches (rtol 1e-2)."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    B, T, D, V = 2, 8, 32, 1000

    h_ref = torch.randn(B, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    h_fused = h_ref.detach().clone().requires_grad_(True)

    head = nn.Linear(D, V, bias=False, device=device, dtype=torch.bfloat16)
    labels = torch.randint(0, V, (B, T), device=device)
    mask = torch.tensor([[0, 1, 1, 1, 1, 0, 1, 1], [0, 1, 0, 1, 1, 1, 1, 0]], device=device, dtype=torch.float32)

    # Reference NLL via LogitsView
    view_ref = LogitsView(h_ref, head)
    ref_logp = view_ref.target_logp(labels)
    ref_loss = -(ref_logp * mask).sum()
    ref_loss.backward()

    # Fused NLL
    view_fused = LogitsView(h_fused, head)
    fused_loss = fused_nll(view_fused, labels, mask, lm_head=head)
    fused_loss.backward()

    # Value check (rtol 1e-3 in bf16)
    assert torch.isclose(fused_loss.float(), ref_loss.float(), rtol=1e-3, atol=1e-2)

    # Gradient check on active positions
    valid_mask = mask[:, 1:].bool()
    ref_grad = h_ref.grad[:, :-1][valid_mask]
    fused_grad = h_fused.grad[:, :-1][valid_mask]
    assert torch.allclose(fused_grad.float(), ref_grad.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
def test_fused_jsd_parity(beta: float):
    """Compare fused JSD against gjsd (T04) at beta in {0.1, 0.5, 0.9}: rtol 1e-3."""
    device = torch.device("cuda")
    torch.manual_seed(int(beta * 100))
    B, T, D, V = 2, 8, 32, 1000

    h_s = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
    h_t = torch.randn(B, T, D, device=device, dtype=torch.float32)

    head_s = nn.Linear(D, V, bias=False, device=device, dtype=torch.float32)
    head_t = nn.Linear(D, V, bias=False, device=device, dtype=torch.float32)

    labels = torch.randint(0, V, (B, T), device=device)
    mask = torch.tensor([[0, 1, 1, 1, 1, 0, 1, 1], [0, 1, 0, 1, 1, 1, 1, 0]], device=device, dtype=torch.float32)

    s_view = LogitsView(h_s, head_s)
    t_view = LogitsView(h_t, head_t)

    # Reference JSD
    ref_per_tok = generalized_jsd(s_view.full(), t_view.full(), beta=beta)
    ref_loss = (ref_per_tok * mask).sum()

    # Fused JSD
    fused_loss = fused_jsd(s_view, t_view, labels, mask, beta=beta, temperature=1.0, lm_head_s=head_s, lm_head_t=head_t)

    assert torch.isclose(fused_loss.float(), ref_loss.float(), rtol=1e-3, atol=1e-2)
