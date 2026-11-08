"""Unit tests for MinED criterion (T13)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext
from lmkd.losses.cross_tokenizer.mined import MinEDFKL
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


def test_unaligned_uses_onehot():
    """dtw_t_for_s all -1 -> KD = FKL against one-hot target."""
    torch.manual_seed(42)
    B, T, V_s, V_t = 2, 4, 10, 15
    logits_s = torch.randn(B, T, V_s)
    logits_t = torch.randn(B, T, V_t)

    # In target-aligned format: target positions 1, 2, 3
    s_labels = torch.tensor(
        [
            [-100, 2, 5, -100],
            [-100, 1, 3, 4],
        ],
        dtype=torch.long,
    )
    dtw_t = torch.full((B, T), -1, dtype=torch.long)

    view_s = LogitsView(logits=logits_s)
    view_t = LogitsView(logits=logits_t)
    so_s = SimpleNamespace(logits=view_s)
    so_t = SimpleNamespace(logits=view_t)
    to_t = TeacherOutput(logits=view_t, side=so_t)

    batch = Batch(
        {
            "student.input_ids": torch.zeros((B, T), dtype=torch.long),
            "student.labels": s_labels,
            "teacher.input_ids": torch.zeros((B, T), dtype=torch.long),
            "align.dtw_t_for_s": dtw_t,
        }
    )

    # Fake vocab map with stu2tea
    stu2tea = torch.full((V_s,), -1, dtype=torch.long)
    stu2tea[0] = 0
    vocab_map = SimpleNamespace(stu2tea=stu2tea)
    rt = SimpleNamespace(data_ctx=SimpleNamespace(vocab_map=vocab_map))

    ctx = LossContext(
        batch=batch,
        student=so_s,
        student_views=[so_s],
        teacher=to_t,
        modules=ModuleRegistry(),
        state=SimpleNamespace(step=1, progress=0.1, log=lambda *args: None),
        rt=rt,
    )

    crit = MinEDFKL(MinEDFKL.Config(temperature=1.0))
    out = crit.forward(ctx)

    assert "mined_fkl" in out.terms
    loss_val = out.terms["mined_fkl"].value

    # Manual FKL vs one-hot:
    # Target positions where s_labels != -100:
    # At temperature 1.0, one-hot target has teacher_probs=1 at label, so KL = -log_softmax(s_logits)[label]
    mask = s_labels != -100
    pos = mask.nonzero(as_tuple=False)
    # Target position t uses logits at t-1 in HF convention
    s_rows = logits_s[pos[:, 0], pos[:, 1] - 1]
    targets = s_labels[pos[:, 0], pos[:, 1]]
    lprobs = F.log_softmax(s_rows, dim=-1)
    expected = -lprobs[torch.arange(pos.shape[0]), targets].sum()

    assert torch.allclose(loss_val, expected, atol=1e-4)
    assert out.metrics["align_ratio"].item() == 0.0
