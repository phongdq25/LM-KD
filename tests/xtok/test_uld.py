"""Unit tests for ULD criterion (T13)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext
from lmkd.losses.cross_tokenizer.uld import ULD, uld_loss
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


def test_uld_zero_when_identical_sorted():
    """teacher_rows = student_rows with columns permuted -> loss 0."""
    torch.manual_seed(42)
    N, V = 10, 64
    s_rows = torch.randn(N, V)
    perm = torch.randperm(V)
    t_rows = s_rows[:, perm].clone()

    loss = uld_loss(s_rows, t_rows, temperature=1.0)
    assert loss.shape == (N,)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_uld_criterion_no_inplace_and_dropped_metric():
    """Check that ULD forward does not modify input tensors and tracks dropped metric."""
    torch.manual_seed(42)
    B, T, V_s, V_t = 2, 6, 20, 25
    logits_s = torch.randn(B, T, V_s)
    logits_t = torch.randn(B, T, V_t)

    s_orig = logits_s.clone()
    t_orig = logits_t.clone()

    # prompt_len: student=2, teacher=2. Sequence length T=6
    # Target positions: 2, 3, 4, 5. Let's make student target at position 5 masked (-100).
    s_labels = torch.full((B, T), -100, dtype=torch.long)
    s_labels[:, 2:5] = torch.tensor([[1, 2, 3], [4, 5, 6]])

    t_labels = torch.full((B, T), -100, dtype=torch.long)
    t_labels[:, 2:6] = torch.tensor([[10, 11, 12, 13], [14, 15, 16, 17]])

    view_s = LogitsView(logits=logits_s)
    view_t = LogitsView(logits=logits_t)
    so_s = SimpleNamespace(logits=view_s)
    so_t = SimpleNamespace(logits=view_t)
    to_t = TeacherOutput(logits=view_t, side=so_t)

    batch = Batch(
        {
            "student.input_ids": torch.zeros((B, T), dtype=torch.long),
            "student.labels": s_labels,
            "student.prompt_len": torch.tensor([2, 2], dtype=torch.long),
            "teacher.input_ids": torch.zeros((B, T), dtype=torch.long),
            "teacher.labels": t_labels,
            "teacher.prompt_len": torch.tensor([2, 2], dtype=torch.long),
        }
    )

    ctx = LossContext(
        batch=batch,
        student=so_s,
        student_views=[so_s],
        teacher=to_t,
        modules=ModuleRegistry(),
        state=SimpleNamespace(step=1, progress=0.1, log=lambda *args: None),
        rt=None,
    )

    crit = ULD()
    out = crit.forward(ctx)
    assert "uld" in out.terms
    assert out.terms["uld"].reduction == "token_sum"
    assert out.terms["uld"].value > 0

    # Ensure inputs were NOT modified in-place
    assert torch.equal(logits_s, s_orig)
    assert torch.equal(logits_t, t_orig)
    assert "uld_dropped" in out.metrics
