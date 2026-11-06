"""Parity tests for AMID distillation loss against the user's reference implementation.

Verifies mathematical parity ($atol < 10^{-5}$) across:
  - Divergences: FKL and AB
  - Orders: pr, qr, rp, rq
  - Mixture parameters: alpha >= 1.0, alpha < 1.0, lam <= 0, lam >= 1, 0 < lam < 1
  - Gradients: d(loss) / d(student_logits)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tests.parity._ref.amid_ref import amid_ref

from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.losses.functional.divergences import amid
from lmkd.losses.token_kd import TokenKD
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherOutput


@pytest.fixture
def sample_logits():
    torch.manual_seed(42)
    B, T, V = 2, 8, 32
    s = torch.randn(B, T, V, dtype=torch.float32)
    t = torch.randn(B, T, V, dtype=torch.float32)
    labels = torch.randint(0, V, (B, T))
    labels[:, :2] = -100
    no_model_batch = {"label": labels}
    mask = (labels != -100).int()
    return s, t, labels, mask, no_model_batch


@pytest.mark.parametrize("div_order", ["pr", "qr", "rp", "rq"])
@pytest.mark.parametrize("alpha", [0.5, 1.5])
@pytest.mark.parametrize("lam", [0.0, 0.4, 1.0])
def test_amid_fkl_parity(sample_logits, div_order, alpha, lam):
    s, t, _, mask, no_model_batch = sample_logits

    ref_val = amid_ref(
        s,
        t,
        no_model_batch,
        alpha=alpha,
        lam=lam,
        div_name="fkl",
        div_order=div_order,
    )

    our_per_tok = amid(
        s,
        t,
        alpha=alpha,
        lam=lam,
        div_name="fkl",
        div_order=div_order,
    )
    our_val = (our_per_tok * mask).sum() / mask.sum()

    assert torch.allclose(our_val, ref_val, atol=1e-5), f"FKL order={div_order}: our={our_val} != ref={ref_val}"


@pytest.mark.parametrize("div_order", ["pr", "qr", "rp", "rq"])
@pytest.mark.parametrize("alpha", [0.5, 1.2])
def test_amid_ab_parity(sample_logits, div_order, alpha):
    s, t, _, mask, no_model_batch = sample_logits

    ref_val = amid_ref(
        s,
        t,
        no_model_batch,
        alpha=alpha,
        lam=0.6,
        div_name="ab",
        div_order=div_order,
        ab_alpha=0.3,
        ab_beta=0.6,
    )

    our_per_tok = amid(
        s,
        t,
        alpha=alpha,
        lam=0.6,
        div_name="ab",
        div_order=div_order,
        ab_alpha=0.3,
        ab_beta=0.6,
    )
    our_val = (our_per_tok * mask).sum() / mask.sum()

    assert torch.allclose(our_val, ref_val, atol=1e-5), f"AB order={div_order}: our={our_val} != ref={ref_val}"


def test_amid_gradient_parity(sample_logits):
    s, t, _, mask, no_model_batch = sample_logits

    s_ref = s.clone().detach().requires_grad_(True)
    ref_loss = amid_ref(
        s_ref,
        t,
        no_model_batch,
        alpha=0.7,
        lam=0.5,
        div_name="fkl",
        div_order="pr",
    )
    ref_grad = torch.autograd.grad(ref_loss, s_ref)[0]

    s_our = s.clone().detach().requires_grad_(True)
    our_per_tok = amid(
        s_our,
        t,
        alpha=0.7,
        lam=0.5,
        div_name="fkl",
        div_order="pr",
    )
    our_loss = (our_per_tok * mask).sum() / mask.sum()
    our_grad = torch.autograd.grad(our_loss, s_our)[0]

    assert torch.allclose(our_loss, ref_loss, atol=1e-5)
    assert torch.allclose(our_grad, ref_grad, atol=1e-5)


def test_amid_token_kd_integration():
    torch.manual_seed(42)
    B, T, V = 2, 6, 20
    s = torch.randn(B, T, V)
    t = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100

    batch = Batch({"student.labels": labels})
    s_view = LogitsView(logits=s)
    t_view = LogitsView(logits=t)
    ctx = LossContext(
        batch=batch,
        student=SimpleNamespace(logits=s_view),
        student_views=[],
        teacher=TeacherOutput(logits=t_view),
        modules=None,
        state=TrainState(),
        rt=None,
    )

    cfg = TokenKD.Config(
        divergence="amid",
        amid_alpha=0.8,
        amid_lam=0.4,
        amid_div_name="fkl",
        amid_div_order="pr",
    )
    crit = TokenKD(cfg)
    loss_out = crit(ctx)

    # Parity with reference on target-aligned logits
    s_target = s_view.full()
    t_target = t_view.full()
    no_model_batch = {"label": labels}
    mask = labels != -100
    ref_val = amid_ref(
        s_target,
        t_target,
        no_model_batch,
        alpha=0.8,
        lam=0.4,
        div_name="fkl",
        div_order="pr",
    )
    ref_sum = ref_val * mask.sum()

    torch.testing.assert_close(loss_out.terms["amid"].value, ref_sum, atol=1e-5, rtol=1e-5)
