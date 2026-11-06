"""Unit tests for Feature Dynamics Distillation (FDD) baseline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from baselines.same_tokenizer.fdd import (
    FDD,
    FDDCriterion,
    compute_fdd_loss,
    soft_label_distill_loss,
)
from lmkd.core.batch import Batch
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import LossContext
from lmkd.plugins import load_builtin_plugins
from lmkd.teachers.signals import TeacherOutput

load_builtin_plugins()


def test_fdd_registered_and_exported():
    assert "fdd" in CRITERIONS
    assert FDD is FDDCriterion
    crit_cls = CRITERIONS.get("fdd")
    assert crit_cls is FDDCriterion


def test_soft_label_distill_loss():
    torch.manual_seed(42)
    B, T, V = 2, 4, 8
    s_logits = torch.randn(B, T, V, requires_grad=True)
    t_logits = torch.randn(B, T, V)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])

    loss = soft_label_distill_loss(s_logits, t_logits, mask=mask, temperature=2.0)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    loss.backward()
    assert s_logits.grad is not None
    # Masked positions [0, 2] and [0, 3] should have 0 gradients
    assert torch.allclose(s_logits.grad[0, 2:], torch.zeros_like(s_logits.grad[0, 2:]))
    # Unmasked positions should have non-zero gradients
    assert (s_logits.grad[0, :2].abs().sum() > 0).item()


def test_compute_fdd_loss_single_and_multi_layer():
    torch.manual_seed(42)
    B, T, D_s, D_t, V = 2, 4, 16, 16, 32

    s_head = nn.Linear(D_s, V, bias=False)
    t_head = nn.Linear(D_t, V, bias=False)

    mask = torch.ones(B, T)

    # 1. Single layer mapping: der_loss should be 0
    s_h1 = torch.randn(B, T, D_s, requires_grad=True)
    t_h1 = torch.randn(B, T, D_t)

    total_1, traj_1, der_1 = compute_fdd_loss(
        s_hiddens=[s_h1],
        t_hiddens=[t_h1],
        mask=mask,
        s_head=s_head,
        t_head=t_head,
        temperature=1.0,
        w_traj=1.0,
        w_der=1.0,
    )
    assert total_1.item() > 0.0
    assert traj_1.item() > 0.0
    assert der_1.item() == 0.0

    # 2. Multi-layer mapping: der_loss should be positive
    s_h2 = torch.randn(B, T, D_s, requires_grad=True)
    t_h2 = torch.randn(B, T, D_t)

    total_2, traj_2, der_2 = compute_fdd_loss(
        s_hiddens=[s_h1, s_h2],
        t_hiddens=[t_h1, t_h2],
        mask=mask,
        s_head=s_head,
        t_head=t_head,
        temperature=1.0,
        w_traj=1.0,
        w_der=1.0,
    )
    assert total_2.item() > 0.0
    assert traj_2.item() > 0.0
    assert der_2.item() > 0.0

    total_2.backward()
    assert s_h1.grad is not None and s_h1.grad.abs().sum() > 0
    assert s_h2.grad is not None and s_h2.grad.abs().sum() > 0


def test_fdd_criterion_forward_and_backward():
    torch.manual_seed(42)
    B, T, D, V = 2, 6, 16, 32

    s_head = nn.Linear(D, V, bias=False)
    t_head = nn.Linear(D, V, bias=False)

    # Mock student & teacher models with adapter
    class MockAdapter:
        def lm_head(self, model):
            return model.head

        def num_layers(self, model):
            return 4

    student_model = SimpleNamespace(head=s_head)
    teacher_model = SimpleNamespace(head=t_head)
    adapter = MockAdapter()

    h_s0 = torch.randn(B, T, D, requires_grad=True)
    h_s1 = torch.randn(B, T, D, requires_grad=True)
    h_t0 = torch.randn(B, T, D)
    h_t1 = torch.randn(B, T, D)

    s_hiddens = {-2: h_s0, -1: h_s1}
    t_hiddens = {-2: h_t0, -1: h_t1}

    side_s = SimpleNamespace(
        model=student_model,
        adapter=adapter,
        logits=None,
        hidden=lambda l: s_hiddens[l],
        num_layers=4,
    )
    side_t = SimpleNamespace(
        model=teacher_model,
        adapter=adapter,
        logits=None,
        hidden=lambda l: t_hiddens[l],
        num_layers=4,
    )
    teacher_out = TeacherOutput(side=side_t)

    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([[1, 2, 3, 4, -100, -100], [5, 6, 7, -100, -100, -100]])
    batch = Batch({"student.labels": labels, "student.attention_mask": mask})

    ctx = LossContext(
        batch=batch,
        student=side_s,
        student_views=[side_s],
        teacher=teacher_out,
        modules=None,
        state=SimpleNamespace(step=0),
        rt=None,
    )

    crit = FDDCriterion(
        FDDCriterion.Config(
            student_layers=[-2, -1],
            teacher_layers=[-2, -1],
            temperature=1.0,
            w_traj=1.0,
            w_der=0.5,
        )
    )

    out = crit.forward(ctx)
    assert "fdd" in out.terms
    assert "fdd/traj" in out.metrics
    assert "fdd/der" in out.metrics

    loss_val = out.terms["fdd"].value
    assert loss_val.item() > 0.0

    loss_val.backward()
    assert h_s0.grad is not None and h_s0.grad.abs().sum() > 0
    assert h_s1.grad is not None and h_s1.grad.abs().sum() > 0


def test_fdd_criterion_layer_mismatch_raises():
    crit = FDDCriterion(
        FDDCriterion.Config(
            student_layers=[-2, -1],
            teacher_layers=[-1],
        )
    )
    ctx = SimpleNamespace(
        teacher=SimpleNamespace(side=SimpleNamespace()),
        student=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="must have the same length"):
        crit.forward(ctx)  # type: ignore[arg-type]

