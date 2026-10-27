"""Small CPU probes for stateless criteria; full lifecycle tests belong in experiments."""
from types import SimpleNamespace

import torch

from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import SideOutput
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherOutput


def synthetic_context() -> tuple[LossContext, torch.Tensor]:
    """Independent student/teacher tensors with ignored targets and a trainable head."""
    hidden = torch.randn(2, 4, 16, requires_grad=True)

    def side(h: torch.Tensor) -> SideOutput:
        out = SideOutput(torch.nn.Identity(), HFAutoAdapter(), None, None)
        out._out = SimpleNamespace(last_hidden_state=h, hidden_states=(h,))
        out.logits = LogitsView(h, torch.nn.Linear(16, 32, bias=False))
        return out

    student, teacher = side(hidden), side(torch.randn_like(hidden))
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, -100]])
    batch = Batch({"student.labels": labels, "student.input_ids": labels.clamp_min(0)})
    ctx = LossContext(batch, student, [student], TeacherOutput(side=teacher, logits=teacher.logits),
                      None, TrainState(), None)
    return ctx, hidden


def probe_gradient(criterion: Criterion) -> LossOutput:
    """Reject invalid values or gradients; this does not replace a Runtime smoke test."""
    ctx, hidden = synthetic_context()
    out = criterion(ctx)
    if not out.terms:
        raise ValueError("loss produced no terms")
    for name, term in out.terms.items():
        if term.value.dim() != 0 or not bool(torch.isfinite(term.value)):
            raise ValueError(f"term {name!r} must be a finite scalar")
        if not term.value.requires_grad:
            raise ValueError(f"term {name!r} is detached from the student")
    total = sum(term.value for term in out.terms.values())
    grad, = torch.autograd.grad(total, hidden, allow_unused=True)
    if grad is None or not bool(torch.isfinite(grad).all()):
        raise ValueError("loss must produce finite student gradients")
    return out
