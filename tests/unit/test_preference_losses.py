"""Unit tests for preference distillation criteria."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from baselines.preference import (
    ADPA,
    CTPD,
    DP3O,
    PBSD,
    PPD,
    TVKD,
    VPD,
    ChosenNLL,
    CompressedKL,
)
from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


def _make_context(
    batch_dict: dict,
    logits: torch.Tensor,
    teacher_token_logp: torch.Tensor | None = None,
) -> LossContext:
    batch = Batch(batch_dict)
    view = LogitsView(logits=logits)
    student_side = SimpleNamespace(logits=view)
    teacher_output = TeacherOutput(token_logp=teacher_token_logp) if teacher_token_logp is not None else None
    return LossContext(
        batch=batch,
        student=student_side,  # type: ignore[arg-type]
        student_views=[student_side],  # type: ignore[list-item]
        teacher=teacher_output,
        modules=ModuleRegistry(),
        state=SimpleNamespace(step=0),
        rt=SimpleNamespace(),
    )


def test_compressed_kl_no_full_vocab_alloc(monkeypatch):
    B, T, V, K = 2, 8, 50, 4
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    topk_ids = torch.randint(0, V, (B, T, K))
    topk_p = torch.softmax(torch.randn(B, T, K), dim=-1) * 0.8
    tail_p = 1.0 - topk_p.sum(dim=-1)

    batch_dict = {
        "student.labels": labels,
        "cache.topk_ids": topk_ids,
        "cache.topk_p": topk_p,
        "cache.tail_p": tail_p,
        "group.size": torch.tensor([2, 2]),
        "group.valid": torch.tensor([True, True]),
    }
    ctx = _make_context(batch_dict, logits)

    def _fail_full(*args, **kwargs):
        raise RuntimeError("LogitsView.full was called!")

    monkeypatch.setattr(LogitsView, "full", _fail_full)

    crit = CompressedKL(CompressedKL.Config(chunk_size=4))
    out = crit.forward(ctx)
    assert "compressed_kl" in out.terms
    assert torch.isfinite(out.terms["compressed_kl"].value)


def test_pairwise_requires_n2():
    B, T, V = 3, 5, 20
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100

    batch_dict = {
        "student.labels": labels,
        "group.size": torch.tensor([3, 3, 3]),
        "group.valid": torch.tensor([True, True, True]),
    }
    ctx = _make_context(batch_dict, logits, teacher_token_logp=torch.randn(B, T))

    crit = DP3O(DP3O.Config())
    with pytest.raises(ValueError, match="requires exactly chosen and rejected"):
        crit.forward(ctx)


def test_chosen_nll_only_chosen():
    B, T, V = 2, 6, 16
    logits = torch.randn(B, T, V)
    labels_1 = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 5, 4, 3, 2, 1]])
    labels_2 = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 9, 8, 7, -100, -100]])
    labels_1[:, 0] = -100
    labels_2[:, 0] = -100

    batch_1 = {
        "student.labels": labels_1,
        "group.size": torch.tensor([2, 2]),
        "group.index": torch.tensor([0, 1]),
    }
    batch_2 = {
        "student.labels": labels_2,
        "group.size": torch.tensor([2, 2]),
        "group.index": torch.tensor([0, 1]),
    }

    ctx_1 = _make_context(batch_1, logits)
    ctx_2 = _make_context(batch_2, logits)

    crit = ChosenNLL(ChosenNLL.Config())
    out_1 = crit.forward(ctx_1)
    out_2 = crit.forward(ctx_2)

    torch.testing.assert_close(out_1.terms["chosen_nll"].value, out_2.terms["chosen_nll"].value)


def test_vpd_and_ppd_criteria():
    B, T, V = 4, 5, 10
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    scores = torch.randn(B)

    batch_dict = {
        "student.labels": labels,
        "group.size": torch.tensor([2, 2, 2, 2]),
        "group.valid": torch.tensor([True, True, True, True]),
        "pref.teacher_score": scores,
    }
    ctx = _make_context(batch_dict, logits)

    vpd = VPD(VPD.Config(beta=2.0))
    vpd_out = vpd.forward(ctx)
    assert "vpd" in vpd_out.terms
    assert torch.isfinite(vpd_out.terms["vpd"].value)

    ppd = PPD(PPD.Config(beta=2.0))
    ppd_out = ppd.forward(ctx)
    assert "ppd" in ppd_out.terms
    assert torch.isfinite(ppd_out.terms["ppd"].value)


def test_tvkd_criterion():
    B, T, V, K = 2, 6, 12, 3
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    ids = torch.randint(0, V, (B, T, K))
    p = torch.softmax(torch.randn(B, T, K), dim=-1) * 0.8
    tail = 1.0 - p.sum(dim=-1)

    batch_dict = {
        "student.labels": labels,
        "group.size": torch.tensor([2, 2]),
        "group.valid": torch.tensor([True, True]),
        "cache.topk_ids": ids,
        "cache.topk_p": p,
        "cache.tail_p": tail,
    }
    ctx = _make_context(batch_dict, logits)
    tvkd = TVKD(TVKD.Config(value_function="entropy"))
    out = tvkd.forward(ctx)
    assert "tvkd" in out.terms
    assert torch.isfinite(out.terms["tvkd"].value)


def test_adpa_criterion():
    B, T, V, K = 2, 6, 12, 3
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    adv_ids = torch.randint(0, V, (B, T, K))
    adv_val = torch.randn(B, T, K)

    batch_dict = {
        "student.labels": labels,
        "group.size": torch.tensor([2, 2]),
        "group.valid": torch.tensor([True, True]),
        "cache.adv_ids": adv_ids,
        "cache.adv_val": adv_val,
    }
    ctx = _make_context(batch_dict, logits)
    adpa = ADPA(ADPA.Config())
    out = adpa.forward(ctx)
    assert "adpa" in out.terms
    assert torch.isfinite(out.terms["adpa"].value)


def test_dp3o_ctpd_pbsd_criteria():
    B, T, V = 2, 6, 12
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    ref_tok = torch.randn(B, T)

    batch_dict = {
        "student.labels": labels,
        "group.size": torch.tensor([2, 2]),
        "group.valid": torch.tensor([True, True]),
        "pref.reward": torch.tensor([1.5, 0.2]),
        "pref.gate": torch.tensor([1.0, 1.0]),
        "pref.ctpd_groups": torch.tensor([[[1, 2], [3, 4]], [[1, 3], [2, 4]]]),
        "pref.ctpd_w": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
    }
    ctx = _make_context(batch_dict, logits, teacher_token_logp=ref_tok)

    dp3o = DP3O(DP3O.Config(alpha=0.5))
    dp3o_out = dp3o.forward(ctx)
    assert "dp3o" in dp3o_out.terms
    assert torch.isfinite(dp3o_out.terms["dp3o"].value)

    ctpd = CTPD(CTPD.Config())
    ctpd_out = ctpd.forward(ctx)
    assert "ctpd" in ctpd_out.terms
    assert torch.isfinite(ctpd_out.terms["ctpd"].value)

    pbsd = PBSD(PBSD.Config())
    pbsd_out = pbsd.forward(ctx)
    assert "pbsd" in pbsd_out.terms
    assert torch.isfinite(pbsd_out.terms["pbsd"].value)
