"""Small CPU probes for stateless criteria; full lifecycle tests belong in experiments."""

from types import SimpleNamespace

import torch

from lmkd.core.batch import Batch
from lmkd.losses.base import Criterion, LossContext, LossOutput, ModelPairInfo
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import SideOutput
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


def synthetic_context(
    num_layers: int = 4, hidden_size: int = 16, vocab_size: int = 32
) -> tuple[LossContext, torch.Tensor]:
    """Independent student/teacher tensors with ignored targets, multi-layer hiddens, and a trainable head."""
    hidden = torch.randn(2, 4, hidden_size, requires_grad=True)

    def side(h: torch.Tensor) -> SideOutput:
        head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        dummy_model = SimpleNamespace(lm_head=head, get_output_embeddings=lambda: head)
        hs = tuple(h for _ in range(num_layers + 1))
        hf_out = SimpleNamespace(last_hidden_state=h, hidden_states=hs)
        return SideOutput(dummy_model, HFAutoAdapter(), hf_out, torch.ones(2, 4, dtype=torch.long))  # type: ignore[arg-type]

    student, teacher = side(hidden), side(torch.randn_like(hidden))
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, -100]])
    batch = Batch(
        {
            "student.labels": labels,
            "student.input_ids": labels.clamp_min(0),
            "student.attention_mask": torch.ones_like(labels),
        }
    )
    ctx = LossContext(batch, student, [student], TeacherOutput(side=teacher, logits=teacher.logits))
    return ctx, hidden


def probe_gradient(criterion: Criterion) -> LossOutput:
    """Reject invalid values or gradients; builds auxiliary modules if the criterion defines build_modules()."""
    ctx, hidden = synthetic_context()
    modules = ModuleRegistry()
    info = ModelPairInfo(
        student_hidden=16,
        student_layers=4,
        student_vocab=32,
        teacher_hidden=16,
        teacher_layers=4,
        teacher_vocab=32,
    )
    criterion.build_modules(info, modules)
    ctx.modules = modules
    out = criterion(ctx)
    if not out.terms:
        raise ValueError("loss produced no terms")
    for name, term in out.terms.items():
        if term.value.dim() != 0 or not bool(torch.isfinite(term.value)):
            raise ValueError(f"term {name!r} must be a finite scalar")
        if not term.value.requires_grad:
            raise ValueError(f"term {name!r} is detached from the student")
    total = sum(term.value for term in out.terms.values())
    (grad,) = torch.autograd.grad(total, hidden, allow_unused=True)
    if grad is None or not bool(torch.isfinite(grad).all()):
        raise ValueError("loss must produce finite student gradients")
    return out
