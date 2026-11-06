"""Numerical parity tests for DWA-KD between LM-KD implementation and MTA/dwa_mta reference."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tests.parity._ref.dwa_ref import (
    ref_calculate_alignment_loss,
    ref_compute_dwa_band_penalty,
    ref_softdtw,
)

from baselines.cross_tokenizer.dwa import (
    DWAKD,
    SoftDTW,
    compute_dwa_band_penalty,
    pairwise_cosine_distance,
    soft_dtw_divergence,
)
from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext, ModelPairInfo
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


@pytest.mark.parity
def test_soft_dtw_forward_parity():
    """Verify soft-DTW forward values match reference numba calculation."""
    torch.manual_seed(42)
    B, N, M = 2, 16, 20
    D = torch.rand(B, N, M, dtype=torch.float64)

    sdtw = SoftDTW(gamma=1.5, backend="antidiag")
    val_our = sdtw.forward_with_cost_matrix(D)
    val_ref = ref_softdtw(D, gamma=1.5)

    torch.testing.assert_close(val_our, val_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parity
def test_soft_dtw_backward_parity():
    """Verify soft-DTW gradients match reference numba backward pass."""
    torch.manual_seed(43)
    B, N, M = 1, 12, 14
    D_our = torch.rand(B, N, M, dtype=torch.float64, requires_grad=True)
    D_ref = D_our.clone().detach().requires_grad_(True)

    sdtw = SoftDTW(gamma=1.2, backend="antidiag")
    val_our = sdtw.forward_with_cost_matrix(D_our)
    val_our.sum().backward()

    val_ref = ref_softdtw(D_ref, gamma=1.2)
    val_ref.sum().backward()

    torch.testing.assert_close(D_our.grad, D_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parity
def test_dwa_band_penalty_parity():
    """Verify adaptive dynamic window penalty matches original DWA-KD formula."""
    torch.manual_seed(44)
    s_len, t_len = 25, 30
    align = torch.softmax(torch.randn(s_len, t_len), dim=-1)

    pen_our = compute_dwa_band_penalty(
        align,
        s_len=s_len,
        t_len=t_len,
        band_width=4.0,
        band_penalty=1.5,
        center_blend=0.7,
        entropy_coef=2.0,
        warmup_scale=0.8,
    )
    pen_ref = ref_compute_dwa_band_penalty(
        align,
        s_len=s_len,
        t_len=t_len,
        band_width=4.0,
        band_penalty=1.5,
        center_blend=0.7,
        entropy_coef=2.0,
        pen_scale=0.8,
    )

    torch.testing.assert_close(pen_our, pen_ref, atol=1e-6, rtol=0)


@pytest.mark.parity
def test_alignment_loss_parity():
    """Verify pairwise alignment loss over multiple sequences matches reference."""
    torch.manual_seed(45)
    B, L_s, L_t, D = 2, 15, 18, 16
    s_embs = torch.randn(B, L_s, D, dtype=torch.float32)
    t_embs = torch.randn(B, L_t, D, dtype=torch.float32)
    s_mask = torch.tensor([[1] * 12 + [0] * 3, [1] * 10 + [0] * 5], dtype=torch.long)
    t_mask = torch.tensor([[1] * 14 + [0] * 4, [1] * 15 + [0] * 3], dtype=torch.long)
    last_align = torch.softmax(torch.randn(B, L_s, L_t), dim=-1)

    our_dtw = SoftDTW(gamma=1.0, backend="antidiag")
    our_loss = torch.tensor(0.0)
    for i in range(B):
        s_len = s_mask[i].sum().item()
        t_len = t_mask[i].sum().item()
        pair_loss, _ = soft_dtw_divergence(
            our_dtw,
            s_seq=s_embs[i, :s_len],
            t_seq=t_embs[i, :t_len],
            align_matrix=last_align[i],
            band_width=3.0,
            band_penalty=1.0,
            center_blend=0.7,
            entropy_coef=2.0,
            band_warmup_scale=1.0,
        )
        our_loss = our_loss + pair_loss

    ref_loss = ref_calculate_alignment_loss(
        s_embs,
        t_embs,
        s_mask,
        t_mask,
        last_align=last_align,
        gamma=1.0,
        band_width=3.0,
        band_penalty=1.0,
        center_blend=0.7,
        entropy_coef=2.0,
        pen_scale=1.0,
    )

    torch.testing.assert_close(our_loss, ref_loss, atol=1e-5, rtol=1e-5)

