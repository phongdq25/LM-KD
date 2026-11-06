"""Parity tests for Self-Play Fine-Tuning baselines against original APSP repo implementation."""

from __future__ import annotations

import pytest
import torch
from tests.parity._ref.apsp_ref import (
    orig_rspo_reversekl_loss,
    orig_sppo_loss,
    orig_sspppo_loss,
)
from tests.parity._ref.apsp_ref import (
    preference_loss as orig_spin_loss,
)
from tests.parity._ref.apsp_ref import (
    space_loss as orig_space_loss,
)
from tests.parity._ref.apsp_ref import (
    tspin_loss as orig_tspin_loss,
)

from baselines.self_play import (
    rspo_loss,
    s_sppo_loss,
    space_loss,
    spin_loss,
    sppo_loss,
    tspin_loss,
)


@pytest.mark.parity
def test_spin_parity():
    torch.manual_seed(42)
    p_c = torch.randn(8)
    p_r = torch.randn(8)
    r_c = torch.randn(8)
    r_r = torch.randn(8)

    for beta in (0.05, 0.1, 0.5):
        for ls in (0.0, 0.1):
            for ref_free in (False, True):
                loss_ours, rew_c_ours, rew_r_ours = spin_loss(
                    p_c, p_r, r_c, r_r, beta=beta, label_smoothing=ls, reference_free=ref_free
                )
                loss_ref, rew_c_ref, rew_r_ref = orig_spin_loss(
                    p_c, p_r, r_c, r_r, beta=beta, label_smoothing=ls, reference_free=ref_free
                )
                torch.testing.assert_close(loss_ours, loss_ref, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(rew_c_ours, rew_c_ref, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(rew_r_ours, rew_r_ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parity
def test_tspin_parity():
    torch.manual_seed(42)
    p_c = torch.randn(8)
    p_r = torch.randn(8)
    p_0 = torch.randn(8)

    for alpha in (0.5, 1.0, 2.0):
        for beta in (0.0, 0.1, 0.5):
            loss_ours, rew_c_ours, rew_r_ours = tspin_loss(p_c, p_r, p_0, alpha=alpha, beta=beta)
            loss_ref, rew_c_ref, rew_r_ref = orig_tspin_loss(p_c, p_r, p_0, alpha=alpha, beta=beta)
            torch.testing.assert_close(loss_ours, loss_ref, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(rew_c_ours, rew_c_ref, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(rew_r_ours, rew_r_ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parity
def test_space_parity():
    torch.manual_seed(42)
    p_c = torch.randn(8)
    p_r = torch.randn(8)
    r_c = torch.randn(8)
    r_r = torch.randn(8)

    for mu in (0.5, 1.0, 2.0):
        loss_ours, rew_c_ours, rew_r_ours = space_loss(p_c, p_r, r_c, r_r, mu=mu)
        loss_ref, rew_c_ref, rew_r_ref = orig_space_loss(p_c, p_r, r_c, r_r, mu=mu)
        torch.testing.assert_close(loss_ours, loss_ref, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(rew_c_ours, rew_c_ref, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(rew_r_ours, rew_r_ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parity
def test_sppo_and_s_sppo_parity():
    torch.manual_seed(42)
    p_c = torch.randn(6)
    p_r = torch.randn(6)
    r_c = torch.randn(6)
    r_r = torch.randn(6)
    p_win = torch.tensor([0.8, 0.75, 0.9, 0.65, 0.85, 0.7])
    p_lose = torch.tensor([0.2, 0.25, 0.1, 0.35, 0.15, 0.3])
    sim = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9, 0.2])
    beta = 0.001

    # SPPO parity
    loss_sppo, _, _ = sppo_loss(p_c, p_r, r_c, r_r, chosen_probs_win=p_win, chosen_probs_lose=p_lose, beta=beta)
    expected_sppo = orig_sppo_loss(p_c, p_r, r_c, r_r, p_win, p_lose, beta=beta)
    torch.testing.assert_close(loss_sppo, expected_sppo, atol=1e-6, rtol=1e-5)

    # S-SPPO parity without hidden
    loss_s_sppo, _, _ = s_sppo_loss(
        p_c, p_r, r_c, r_r, sim_score=sim, chosen_probs_win=p_win, chosen_probs_lose=p_lose, beta=beta
    )
    expected_sspppo = orig_sspppo_loss(p_c, p_r, r_c, r_r, p_win, p_lose, sim, beta=beta)
    torch.testing.assert_close(loss_s_sppo, expected_sspppo, atol=1e-6, rtol=1e-5)

    # S-SPPO parity with hidden repulsion
    h_c = torch.randn(6, 16)
    h_r = torch.randn(6, 16)
    loss_s_sppo_h, _, _ = s_sppo_loss(
        p_c,
        p_r,
        r_c,
        r_r,
        sim_score=sim,
        chosen_probs_win=p_win,
        chosen_probs_lose=p_lose,
        h_chosen=h_c,
        h_rejected=h_r,
        beta=beta,
        lam=0.5,
    )
    expected_sspppo_h = orig_sspppo_loss(
        p_c, p_r, r_c, r_r, p_win, p_lose, sim, beta=beta, h_chosen=h_c, h_rejected=h_r, lam=0.5
    )
    torch.testing.assert_close(loss_s_sppo_h, expected_sspppo_h, atol=1e-6, rtol=1e-5)


@pytest.mark.parity
def test_rspo_parity():
    torch.manual_seed(42)
    p_c = torch.randn(6)
    p_r = torch.randn(6)
    r_c = torch.randn(6)
    r_r = torch.randn(6)
    b_c = torch.randn(6)
    b_r = torch.randn(6)
    p_win = torch.tensor([0.8, 0.75, 0.9, 0.65, 0.85, 0.7])
    p_lose = torch.tensor([0.2, 0.25, 0.1, 0.35, 0.15, 0.3])
    beta = 0.001
    reg_coef = 0.5

    loss_rspo, _, _, reg_val = rspo_loss(
        p_c,
        p_r,
        r_c,
        r_r,
        base_chosen_logps=b_c,
        base_rejected_logps=b_r,
        chosen_probs_win=p_win,
        chosen_probs_lose=p_lose,
        beta=beta,
        reg_coef=reg_coef,
    )
    expected_rspo, expected_reg = orig_rspo_reversekl_loss(
        p_c, p_r, r_c, r_r, b_c, b_r, p_win, p_lose, beta=beta, reg_coef=reg_coef
    )
    torch.testing.assert_close(loss_rspo, expected_rspo, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reg_val, expected_reg, atol=1e-6, rtol=1e-5)
