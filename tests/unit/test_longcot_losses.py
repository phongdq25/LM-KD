from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from baselines.longCoT.sgl_spectral import (
    PerSourceObjective,
    TransitionLoss,
    TransitionPredictor,
    compute_transition_hidden_index,
)
from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry


def _make_ctx(
    batch: Batch,
    student: Any,
    modules: ModuleRegistry | None = None,
    state: TrainState | None = None,
) -> LossContext:
    return LossContext(
        batch=batch,
        student=student,
        student_views=[student],
        teacher=None,
        modules=modules if modules is not None else ModuleRegistry(),
        state=state if state is not None else TrainState(),
        rt=None,
    )


def test_global_norm_equivalence() -> None:
    """Two micro-batches with different mask densities sum to exact full-batch loss."""
    torch.manual_seed(123)
    V = 11
    # Micro 0: length 6, only 3 supervised tokens
    logits0 = torch.randn(1, 6, V, dtype=torch.float32)
    labels0 = torch.tensor([[-100, 2, 5, -100, 3, -100]], dtype=torch.long)

    # Micro 1: length 8, 7 supervised tokens
    logits1 = torch.randn(1, 8, V, dtype=torch.float32)
    labels1 = torch.tensor([[-100, 1, 4, 2, 8, 6, 9, 3]], dtype=torch.long)

    crit = PerSourceObjective(PerSourceObjective.Config(default="nll"))

    # Forward micro 0
    v0 = LogitsView(logits=logits0)
    ctx0 = _make_ctx(
        batch=Batch({"student.labels": labels0}),
        student=SimpleNamespace(logits=v0),
    )
    out0 = crit.forward(ctx0)
    num0 = out0.terms["obj"].value

    # Forward micro 1
    v1 = LogitsView(logits=logits1)
    ctx1 = _make_ctx(
        batch=Batch({"student.labels": labels1}),
        student=SimpleNamespace(logits=v1),
    )
    out1 = crit.forward(ctx1)
    num1 = out1.terms["obj"].value

    z0 = (labels0[:, 1:] != -100).sum()
    z1 = (labels1[:, 1:] != -100).sum()
    z_tot = z0 + z1
    combined_loss = (num0 + num1) / z_tot

    # Full batch calculation (padded to max length 8)
    logits_full = torch.zeros(2, 8, V, dtype=torch.float32)
    logits_full[0, :6] = logits0[0]
    logits_full[1, :8] = logits1[0]
    labels_full = torch.full((2, 8), -100, dtype=torch.long)
    labels_full[0, :6] = labels0[0]
    labels_full[1, :8] = labels1[0]

    v_full = LogitsView(logits=logits_full)
    ctx_full = _make_ctx(
        batch=Batch({"student.labels": labels_full}),
        student=SimpleNamespace(logits=v_full),
    )
    out_full = crit.forward(ctx_full)
    full_loss = out_full.terms["obj"].value / z_tot

    assert torch.allclose(combined_loss, full_loss, atol=1e-6)


def test_dft_no_grad_through_coef() -> None:
    """grad wrt logits of DFT == p * grad(NLL) on that token position."""
    torch.manual_seed(99)
    B, T, V = 1, 4, 7
    raw_logits_nll = torch.randn(B, T, V, dtype=torch.float32, requires_grad=True)
    raw_logits_dft = raw_logits_nll.detach().clone().requires_grad_(True)
    labels = torch.tensor([[-100, 3, 2, -100]], dtype=torch.long)

    # NLL
    crit_nll = PerSourceObjective(PerSourceObjective.Config(default="nll"))
    ctx_nll = _make_ctx(
        batch=Batch({"student.labels": labels}),
        student=SimpleNamespace(logits=LogitsView(logits=raw_logits_nll)),
    )
    out_nll = crit_nll.forward(ctx_nll)
    out_nll.terms["obj"].value.backward()
    grad_nll = raw_logits_nll.grad

    # DFT
    crit_dft = PerSourceObjective(PerSourceObjective.Config(default="dft"))
    ctx_dft = _make_ctx(
        batch=Batch({"student.labels": labels}),
        student=SimpleNamespace(logits=LogitsView(logits=raw_logits_dft)),
    )
    out_dft = crit_dft.forward(ctx_dft)
    out_dft.terms["obj"].value.backward()
    grad_dft = raw_logits_dft.grad

    assert grad_nll is not None and grad_dft is not None
    # For target token 1 (pos 1, target id 3): raw_logits[:, 0] predicts token 1
    probs = torch.softmax(raw_logits_nll[:, 0], dim=-1)
    p_target1 = probs[0, 3].item()

    assert torch.allclose(grad_dft[:, 0], grad_nll[:, 0] * p_target1, atol=1e-6)


def test_transition_no_pairs_zero_grad() -> None:
    """Empty pair_src gives zero loss and zero gradients on predictor parameters."""
    d = 8
    pred = TransitionPredictor(d_model=d, hidden=16)
    modules = ModuleRegistry()
    modules.add("transition_loss__predictor", pred, group="trans_predictor", owner="transition_loss")

    cfg = TransitionLoss.Config(layer=1, predictor_hidden=16)
    crit = TransitionLoss(cfg)
    crit._hidden_index = 1

    T = 10
    H = torch.randn(1, T, d, requires_grad=True)
    batch = Batch(
        {
            "student.labels": torch.full((1, T), 0, dtype=torch.long),
            "seq.step_id": torch.full((1, T), -1, dtype=torch.long),
            "seq.step_end": [torch.empty(0, dtype=torch.long)],
            "seq.pair_src": [torch.empty(0, dtype=torch.long)],
        }
    )

    class DummyStudent:
        def hidden(self, layer: int) -> torch.Tensor:
            return H

    ctx = _make_ctx(
        batch=batch,
        student=DummyStudent(),
        modules=modules,
    )
    out = crit.forward(ctx)
    loss = out.terms["trans"].value
    assert loss.item() == 0.0

    loss.backward()
    assert pred.up.weight.grad is not None
    assert torch.all(pred.up.weight.grad == 0.0)


def test_layer_index_two_thirds() -> None:
    """L=36 -> hidden_index 25, matching 0-based block round(2L/3) + 1."""
    assert compute_transition_hidden_index("two_thirds", 36) == 25
    assert compute_transition_hidden_index(0.667, 36) == 25
    assert compute_transition_hidden_index(0.5, 20) == 11
    assert compute_transition_hidden_index(15, 36) == 16  # block 15 -> HF hidden index 16


def test_lambda_ramp() -> None:
    """warmup_ratio 0.1, total 100, step 5 -> lambda 0.5."""
    d = 8
    pred = TransitionPredictor(d_model=d, hidden=16)
    modules = ModuleRegistry()
    modules.add("transition_loss__predictor", pred, group="trans_predictor", owner="transition_loss")

    cfg = TransitionLoss.Config(layer=1, predictor_hidden=16, warmup_ratio=0.1)
    crit = TransitionLoss(cfg)
    crit._hidden_index = 1

    T = 6
    H = torch.randn(1, T, d)
    step_id = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    step_end = torch.tensor([1, 3, 5], dtype=torch.long)
    pair_src = torch.tensor([0, 1], dtype=torch.long)
    batch = Batch(
        {
            "student.labels": torch.full((1, T), 0, dtype=torch.long),
            "seq.step_id": step_id,
            "seq.step_end": [step_end],
            "seq.pair_src": [pair_src],
        }
    )

    class DummyStudent:
        def hidden(self, layer: int) -> torch.Tensor:
            return H

    # Step 5 of 100 with warmup_ratio 0.1 -> warmup_steps=10 -> lambda=0.5
    state5 = TrainState(total_steps=100)
    state5.step = 5
    ctx5 = _make_ctx(
        batch=batch,
        student=DummyStudent(),
        modules=modules,
        state=state5,
    )
    out5 = crit.forward(ctx5)
    assert abs(out5.metrics["lambda"].item() - 0.5) < 1e-6

    # Step 10 -> lambda 1.0
    state10 = TrainState(total_steps=100)
    state10.step = 10
    ctx10 = _make_ctx(
        batch=batch,
        student=DummyStudent(),
        modules=modules,
        state=state10,
    )
    out10 = crit.forward(ctx10)
    assert abs(out10.metrics["lambda"].item() - 1.0) < 1e-6


def test_config_validation() -> None:
    """Invalid default, objectives, or layer configs raise ValueError."""
    with pytest.raises(ValueError, match="default must be"):
        PerSourceObjective.Config(default="invalid")

    with pytest.raises(ValueError, match="unknown source"):
        PerSourceObjective.Config(objectives={"unknown_src": "nll"})

    with pytest.raises(ValueError, match="objective value must be"):
        PerSourceObjective.Config(objectives={"cont": "invalid"})

    with pytest.raises(ValueError, match="layer"):
        TransitionLoss.Config(layer="half")

    with pytest.raises(ValueError, match="layer"):
        TransitionLoss.Config(layer=1.5)

    with pytest.raises(ValueError, match="layer"):
        TransitionLoss.Config(layer=0)
