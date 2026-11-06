"""Parity checks against the checked-out SGL signal implementations."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from baselines.longCoT.sgl_spectral.spectral import analytic_gradients_and_entropies, effective_rank, step_strengths
from baselines.longCoT.ssft import compact_ig

SGL_SRC = Path(__file__).resolve().parents[3] / "Long-CoT-Disitllation/SpectralGuidedLearning/src"
sys.path.insert(0, str(SGL_SRC))
from sgl.signals.spectral import analyze_gradient_matrix  # noqa: E402
from sgl.ssft.attribution import compact  # noqa: E402


@pytest.mark.parity
def test_analytic_grad_matches_autograd() -> None:
    torch.manual_seed(19)
    h = torch.randn(7, 5, requires_grad=True)
    w = torch.randn(11, 5)
    y = torch.randint(0, 11, (7,))
    ref = torch.nn.functional.cross_entropy(h @ w.T, y, reduction="sum")
    grad = torch.autograd.grad(ref, h)[0]
    got, ent = analytic_gradients_and_entropies(h.detach(), y, w, chunk=3)
    assert torch.allclose(got, grad, atol=1e-5, rtol=0)
    probs = (h.detach() @ w.T).softmax(-1)
    expected_ent = -(probs * probs.log()).sum(-1)
    assert torch.allclose(ent, expected_ent, atol=1e-5, rtol=0)


@pytest.mark.parity
def test_spectral_matches_sgl() -> None:
    torch.manual_seed(20)
    g = torch.randn(13, 6)
    spans = [(0, 4), (4, 9), (9, 13)]
    ref = analyze_gradient_matrix(g, spans, 0.93)
    u, s, _ = torch.linalg.svd(g.float(), full_matrices=False)
    k = effective_rank(s, 0.93)
    got = step_strengths((u[:, :k] ** 2).sum(-1), spans)
    assert k == ref.k_star
    assert got == pytest.approx(ref.step_strengths, abs=1e-5, rel=0)


@pytest.mark.parity
def test_ssft_compact_matches_sgl() -> None:
    scores = [[0.1, -0.5, 0.2], [], [-0.35, 0.6]]
    assert compact_ig(scores) == compact(scores)
