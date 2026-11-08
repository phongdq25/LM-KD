from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from baselines.longCoT.sgl_spectral import PerSourceObjective, TransitionLoss, TransitionPredictor
from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry

SGL_ROOT = Path(__file__).resolve().parents[3] / "Long-CoT-Disitllation" / "SpectralGuidedLearning"
SGL_SRC = SGL_ROOT / "src"

pytestmark = [
    pytest.mark.parity,
    pytest.mark.skipif(not SGL_SRC.exists(), reason="SGL repo not found"),
]

if SGL_SRC.exists() and str(SGL_SRC) not in sys.path:
    sys.path.insert(0, str(SGL_SRC))


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


@pytest.mark.parametrize("case", ["nll", "dft", "weights", "provenance"])
def test_masked_ce_parity(case: str) -> None:
    from sgl.training.losses import OBJECTIVE_DFT, OBJECTIVE_NLL, masked_cross_entropy

    torch.manual_seed(42)
    B, T, V = 2, 9, 13
    logits = torch.randn(B, T, V, dtype=torch.float32)
    labels = torch.randint(0, V, (B, T), dtype=torch.long)
    labels[:, 0] = -100
    labels[0, 6:] = -100

    token_weights = None
    token_objective = None
    batch_dict: dict[str, torch.Tensor] = {"student.labels": labels}

    if case == "nll":
        cfg = PerSourceObjective.Config(default="nll")
    elif case == "dft":
        cfg = PerSourceObjective.Config(default="dft")
        token_objective = torch.full_like(labels, OBJECTIVE_DFT)
    elif case == "weights":
        cfg = PerSourceObjective.Config(default="nll", use_token_weights=True)
        token_weights = torch.rand(B, T, dtype=torch.float32)
        batch_dict["seq.token_weights"] = token_weights
    elif case == "provenance":
        cfg = PerSourceObjective.Config(
            default="nll",
            objectives={"prefix": "nll", "cont": "dft", "junction": "nll"},
            use_token_weights=True,
        )
        token_source = torch.randint(0, 4, (B, T), dtype=torch.long)
        batch_dict["seq.token_source"] = token_source
        token_weights = torch.rand(B, T, dtype=torch.float32)
        batch_dict["seq.token_weights"] = token_weights
        token_objective = torch.full_like(labels, OBJECTIVE_NLL)
        token_objective[token_source == 2] = OBJECTIVE_DFT  # cont -> dft
    else:
        raise ValueError(f"Unknown case: {case}")

    Z = (labels[:, 1:] != -100).sum()
    sgl_loss = masked_cross_entropy(
        logits,
        labels,
        denominator=Z,
        token_weights=token_weights,
        token_objective=token_objective,
    )

    crit = PerSourceObjective(cfg)
    view = LogitsView(logits=logits)
    student = SimpleNamespace(logits=view)
    ctx = _make_ctx(
        batch=Batch(batch_dict),
        student=student,
    )
    out = crit.forward(ctx)
    fw_loss = out.terms["obj"].value / Z

    assert torch.allclose(fw_loss, sgl_loss, atol=1e-6)


def test_transition_parity() -> None:
    from sgl.training.transition_loss import (
        TransitionPredictor as SGLPredictor,
    )
    from sgl.training.transition_loss import (
        transition_loss as sgl_transition_loss,
    )

    torch.manual_seed(42)
    T, d = 20, 8
    H = torch.randn(T, d, dtype=torch.float32, requires_grad=True)

    step_id = torch.full((T,), -1, dtype=torch.long)
    step_id[2:8] = 0
    step_id[8:15] = 1
    step_id[15:20] = 2

    step_end = torch.tensor([7, 14, 19], dtype=torch.long)
    pair_src = torch.tensor([0, 1], dtype=torch.long)

    sgl_pred = SGLPredictor(d_model=d, hidden=16)
    fw_pred = TransitionPredictor(d_model=d, hidden=16)
    fw_pred.load_state_dict(sgl_pred.state_dict())

    sgl_loss, sgl_stats = sgl_transition_loss(H, step_id, step_end, pair_src, sgl_pred)

    modules = ModuleRegistry()
    modules.add("transition_loss__predictor", fw_pred, group="trans_predictor", owner="transition_loss")

    cfg = TransitionLoss.Config(layer=1, predictor_hidden=16)
    crit = TransitionLoss(cfg)
    crit._hidden_index = 1

    batch = Batch(
        {
            "student.labels": torch.full((1, T), 0, dtype=torch.long),
            "seq.step_id": step_id.unsqueeze(0),
            "seq.step_end": [step_end],
            "seq.pair_src": [pair_src],
        }
    )

    class DummyStudent:
        def hidden(self, layer: int) -> torch.Tensor:
            return H.unsqueeze(0)

    ctx = _make_ctx(
        batch=batch,
        student=DummyStudent(),
        modules=modules,
    )
    out = crit.forward(ctx)
    fw_loss = out.terms["trans"].value

    assert torch.allclose(fw_loss, sgl_loss, atol=1e-5)
    assert abs(out.metrics["trans_cos"].item() - sgl_stats.cos) < 1e-5
