"""Offline teacher provider signals."""

import pytest
import torch

from lmkd.core.batch import Batch
from lmkd.teachers.providers import CacheTeacher
from lmkd.teachers.signals import TeacherSignalSpec


def test_signals_topk():
    teacher = CacheTeacher(CacheTeacher.Config(kind="topk"))
    batch = Batch(
        {
            "cache.topk_ids": torch.tensor([[[1, 2]]]),
            "cache.topk_p": torch.tensor([[[0.7, 0.2]]]),
            "cache.tail_p": torch.tensor([[0.1]]),
        }
    )
    out = teacher.signals(batch, TeacherSignalSpec(kind="topk", k=2), None, None)
    assert torch.equal(out.topk_ids, batch["cache.topk_ids"])
    torch.testing.assert_close(out.topk_logp.exp(), batch["cache.topk_p"])
    torch.testing.assert_close(out.tail_logp.exp(), batch["cache.tail_p"])
    with pytest.raises(ValueError, match="cannot produce"):
        teacher.check(TeacherSignalSpec(kind="full"), None)


def test_signals_token_and_sequence():
    teacher = CacheTeacher(CacheTeacher.Config(kind="token_logp"))
    batch = Batch(
        {"cache.token_logp": torch.tensor([[0.0, -0.5, -1.2]]), "student.labels": torch.tensor([[-100, 1, 2]])}
    )
    out = teacher.signals(batch, TeacherSignalSpec(kind="sampled_logp"), None, None)
    assert out.token_logp.tolist()[0] == pytest.approx([0.0, -0.5, -1.2])
    assert out.seq_logp.item() == pytest.approx(-1.7)
