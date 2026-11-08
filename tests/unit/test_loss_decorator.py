"""Tests for Loss Decorator API (@lmkd.loss, TokenLoss, SeqLoss, ScalarLoss, Inputs, chunked vs full)."""

import pytest
import torch

import lmkd
from lmkd import Inputs, ScalarLoss, SeqLoss, TokenLoss
from lmkd.core.batch import Batch
from lmkd.core.registry import CRITERIONS
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import SideOutput
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherOutput


def test_token_loss_as_loss_term():
    val = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    tl = TokenLoss(val, mask)
    term = tl.as_loss_term()
    assert term.reduction == "token_sum"
    assert torch.isclose(term.value, torch.tensor(1.0 + 3.0 + 4.0))


def test_seq_loss_as_loss_term():
    val = torch.tensor([2.5, 4.5])
    sl = SeqLoss(val)
    term = sl.as_loss_term()
    assert term.reduction == "seq_sum"
    assert torch.isclose(term.value, torch.tensor(7.0))


def test_scalar_loss_as_loss_term():
    val = torch.tensor(1.234)
    scl = ScalarLoss(val)
    term = scl.as_loss_term()
    assert term.reduction == "mean"
    assert torch.isclose(term.value, torch.tensor(1.234))

    with pytest.raises(ValueError, match="must be scalar"):
        ScalarLoss(torch.tensor([1.0, 2.0])).as_loss_term()


def test_unified_loss_decorator_registration():
    @lmkd.loss("test_custom_fkl", teacher=True)
    def custom_fkl(s: Inputs, t: Inputs, temperature: float = 2.0) -> TokenLoss:
        p = (t.logits / temperature).softmax(-1)
        per_tok = -(p * (s.logits / temperature).log_softmax(-1)).sum(-1)
        return TokenLoss(per_tok, s.mask)

    assert "test_custom_fkl" in CRITERIONS
    meta = CRITERIONS.meta("test_custom_fkl")
    assert meta.name == "test_custom_fkl"


def test_unified_loss_execution_chunked_and_full(tiny_models):
    # Register test loss
    @lmkd.loss("test_toy_distill", teacher=True)
    def toy_distill(s: Inputs, t: Inputs, tau: float = 1.0) -> TokenLoss:
        diff = (s.logits - t.logits) ** 2
        return TokenLoss(diff.mean(-1), s.mask)

    crit_cls = CRITERIONS.get("test_toy_distill")
    crit = crit_cls(crit_cls.Config(tau=2.0))

    # Mock batch
    B, T, V = 2, 8, 32
    head = torch.nn.Linear(16, V, bias=False)
    hidden_s = torch.randn(B, T, 16, requires_grad=True)
    hidden_t = torch.randn(B, T, 16)
    labels = torch.tensor([[0, 1, 2, 3, -100, -100, -100, -100], [0, 5, 6, -100, -100, -100, -100, -100]])
    batch = Batch({"student.labels": labels, "student.input_ids": labels.clamp(min=0)})

    s_view = LogitsView(hidden_s, head)
    t_view = LogitsView(hidden_t, head)

    adapter = HFAutoAdapter()
    dummy_model = torch.nn.Identity()
    s_side = SideOutput(dummy_model, adapter, None, None)
    s_side.logits = s_view
    s_side._out = type("MockOut", (), {"last_hidden_state": hidden_s})()

    t_side = SideOutput(dummy_model, adapter, None, None)
    t_side.logits = t_view
    t_side._out = type("MockOut", (), {"last_hidden_state": hidden_t})()

    teacher_out = TeacherOutput(side=t_side, logits=t_view)
    ctx = LossContext(
        batch=batch,
        student=s_side,
        student_views=[s_side],
        teacher=teacher_out,
        modules=None,
        state=TrainState(),
        rt=None,
    )

    out = crit.forward(ctx)
    assert "loss" in out.terms
    loss_term = out.terms["loss"]
    assert loss_term.reduction == "token_sum"
    assert loss_term.value.requires_grad
    assert torch.isfinite(loss_term.value)

    # Backward works
    loss_term.value.backward()
    assert hidden_s.grad is not None
