"""Unit tests for E-SCOT criterion (T13)."""

from __future__ import annotations

import torch

from baselines.cross_tokenizer.escot import compute_renyi_entropies, escot_weights


def test_weights_mean_one():
    """For every sample: mean of w over us equals lam + (1-lam)*1 = 1 (tol 1e-5)."""
    torch.manual_seed(42)
    # Simulate batch of 3 samples with different counts of unique student positions
    counts = [10, 25, 40]
    lam = 0.25

    b_idx_list = []
    for b, c in enumerate(counts):
        b_idx_list.append(torch.full((c,), b, dtype=torch.long))
    b_idx = torch.cat(b_idx_list)

    # Random positive delta values
    total_u = sum(counts)
    q = torch.rand(total_u) + 0.1

    w = escot_weights(b_idx, q, lam=lam)
    assert w.shape == (total_u,)

    for b, c in enumerate(counts):
        mask = b_idx == b
        w_b = w[mask]
        mean_w = w_b.mean().item()
        assert abs(mean_w - 1.0) < 1e-5, f"Sample {b} mean weight {mean_w} != 1.0"


def test_renyi_entropies_shannon_limit():
    """Alpha=1 matches normalized Shannon entropy."""
    torch.manual_seed(42)
    N, V = 8, 30
    logits = torch.randn(N, V)
    probs = torch.softmax(logits, dim=-1)

    ent = compute_renyi_entropies(probs, V=V, alphas=[1.0])
    assert ent.shape == (N, 1)

    p = probs.clamp(min=1e-12)
    shannon_expected = -(p * torch.log(p)).sum(dim=-1, keepdim=True) / torch.log(torch.tensor(float(V)))
    assert torch.allclose(ent, shannon_expected, atol=1e-5)
