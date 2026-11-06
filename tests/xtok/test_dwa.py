"""Unit and integration tests for DWA-KD cross-tokenizer baseline."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from baselines.cross_tokenizer.dwa import (
    DWAKD,
    SoftDTW,
    compute_dwa_band_penalty,
    soft_dtw_divergence,
)
from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext, ModelPairInfo
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


def _create_mock_context(
    B: int = 2,
    L_s: int = 8,
    L_t: int = 10,
    D_s: int = 16,
    D_t: int = 24,
    V_s: int = 32,
    V_t: int = 40,
):
    torch.manual_seed(101)
    # Input embeddings
    stu_emb = nn.Embedding(V_s, D_s)
    tea_emb = nn.Embedding(V_t, D_t)
    stu_head = nn.Linear(D_s, V_s, bias=False)
    tea_head = nn.Linear(D_t, V_t, bias=False)

    s_ids = torch.randint(1, V_s, (B, L_s))
    t_ids = torch.randint(1, V_t, (B, L_t))

    labels_s = s_ids.clone()
    labels_s[:, :2] = -100  # mask first 2 prompt tokens
    labels_t = t_ids.clone()
    labels_t[:, :3] = -100

    h_s_raw = torch.randn(B, L_s, D_s, requires_grad=True)
    h_t_raw = torch.randn(B, L_t, D_t)

    s_logits = torch.randn(B, L_s, V_s, requires_grad=True)
    t_logits = torch.randn(B, L_t, V_t)

    student = SimpleNamespace(
        logits=LogitsView(logits=s_logits, lm_head=stu_head),
        hidden=lambda idx: h_s_raw,
    )
    teacher_side = SimpleNamespace(
        logits=LogitsView(logits=t_logits, lm_head=tea_head),
        hidden=lambda idx: h_t_raw,
    )
    teacher = TeacherOutput(logits=teacher_side.logits, side=teacher_side)

    batch = Batch(
        {
            "student.input_ids": s_ids,
            "teacher.input_ids": t_ids,
            "student.labels": labels_s,
            "teacher.labels": labels_t,
            "student.attention_mask": torch.ones(B, L_s, dtype=torch.long),
            "teacher.attention_mask": torch.ones(B, L_t, dtype=torch.long),
        }
    )

    modules = ModuleRegistry()
    info = ModelPairInfo(
        student_hidden=D_s,
        student_layers=4,
        student_vocab=V_s,
        teacher_hidden=D_t,
        teacher_layers=6,
        teacher_vocab=V_t,
    )

    rt = SimpleNamespace(
        student_model=SimpleNamespace(),
        teacher_model=SimpleNamespace(),
        student_adapter=SimpleNamespace(embed_tokens=lambda m: stu_emb, lm_head=lambda m: stu_head),
        teacher_adapter=SimpleNamespace(embed_tokens=lambda m: tea_emb, lm_head=lambda m: tea_head),
    )

    ctx = LossContext(
        batch=batch,
        student=student,
        student_views=[student],
        teacher=teacher,
        modules=modules,
        state=TrainState(step=0, total_steps=100),
        rt=rt,
    )
    return ctx, info, modules


def test_soft_dtw_return_alignment():
    """Verify differentiable alignment matrix A properties (non-negative, row sum=1)."""
    sdtw = SoftDTW(gamma=1.0)
    C = torch.rand(2, 8, 12, requires_grad=True)
    vals, A = sdtw.forward_with_cost_matrix(C, return_alignment=True)

    assert vals.shape == (2,)
    assert A.shape == (2, 8, 12)
    assert (A >= 0).all()
    # Row normalization check
    row_sums = A.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)


def test_soft_dtw_divergence_properties():
    """Verify self-distance is near zero and cross-distance is non-negative."""
    sdtw = SoftDTW(gamma=1.0)
    x = torch.randn(10, 16)
    y = torch.randn(12, 16)

    self_div, _ = soft_dtw_divergence(sdtw, x, x, band_width=0.0)
    assert abs(self_div.item()) < 1e-4

    cross_div, _ = soft_dtw_divergence(sdtw, x, y, band_width=0.0)
    assert cross_div.item() >= -1e-4


def test_dwa_band_penalty_edge_cases():
    """Check zero length and zero width return zeros."""
    A = torch.softmax(torch.randn(10, 10), dim=-1)
    pen_zero_w = compute_dwa_band_penalty(A, 10, 10, band_width=0.0)
    assert (pen_zero_w == 0).all()

    pen_zero_len = compute_dwa_band_penalty(A, 0, 10, band_width=5.0)
    assert pen_zero_len.numel() == 0


def test_dwa_kd_build_modules():
    """Verify auxiliary projectors are registered in ModuleRegistry under group 'projector'."""
    _, info, modules = _create_mock_context()
    crit = DWAKD()
    crit.build_modules(info, modules)

    expected = {"query", "s2t", "t2s", "dtw_embed_t2s"}
    mod_names = {name for name, _ in modules.named_modules()}
    for exp in expected:
        assert any(f"dwa_kd__{exp}" in name for name in mod_names), f"Expected aux module {exp} not in {mod_names}"


def test_dwa_kd_forward_step():
    """Verify forward step produces finite loss and expected terms and metrics."""
    ctx, info, modules = _create_mock_context()
    crit = DWAKD(
        DWAKD.Config(
            temperature=2.0,
            kd_objective="forward_kl",
            dtw_rate=0.2,
            dtw_gamma=1.5,
            dtw_band_width=4.0,
        )
    )
    crit.build_modules(info, modules)

    output = crit(ctx)
    assert "t2s_ce" in output.terms
    assert "t2s_kd" in output.terms
    assert "s2t_kd" in output.terms
    assert "dtw" in output.terms

    for name, term in output.terms.items():
        assert torch.isfinite(term.value), f"Term {name} produced non-finite value: {term.value}"

    assert "t2s_acc" in output.metrics
    assert "max_t2s_prob" in output.metrics
    assert "dtw_loss" in output.metrics
    assert output.metrics["dtw_loss"] > 0


def test_dwa_kd_zero_dtw_rate():
    """When dtw_rate is 0.0, dtw term is not emitted."""
    ctx, info, modules = _create_mock_context()
    crit = DWAKD(DWAKD.Config(dtw_rate=0.0))
    crit.build_modules(info, modules)

    output = crit(ctx)
    assert "dtw" not in output.terms
    assert output.metrics["dtw_loss"] == 0.0
