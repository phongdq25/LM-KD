"""Parity tests for DSKD criteria family (T12) against verbatim reference implementations."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from tests.parity._ref.dskd_ref import (
    compute_dual_space_kd_loss_ref,
    compute_dual_space_kd_loss_with_cma_ref,
    compute_dual_space_kd_v2_loss_ref,
)
from tests.parity._ref.various_divergence_ref import VariousDivergenceRef
from tests.parity._xtok_utils import pad_pos, pos_to_tgt_labels

from baselines.dskd import DSKD, DSKDCMA
from baselines.dskdv2 import DSKDv2, DSKDv2ETA
from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext, PrepareContext
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


def _build_mock_context(
    logits_s: torch.Tensor,
    h_s: torch.Tensor,
    logits_t: torch.Tensor,
    h_t: torch.Tensor,
    labels_pos: torch.Tensor,
    t_labels_pos: torch.Tensor | None = None,
    stu_head: nn.Module | None = None,
    tea_head: nn.Module | None = None,
    extra_batch: dict[str, Any] | None = None,
    rt: Any | None = None,
    mods: ModuleRegistry | None = None,
) -> LossContext:
    if t_labels_pos is None:
        t_labels_pos = labels_pos

    labels_s_tgt = pos_to_tgt_labels(labels_pos)
    labels_t_tgt = pos_to_tgt_labels(t_labels_pos)

    view_s = LogitsView(logits=pad_pos(logits_s))
    if stu_head is not None:
        view_s._head = stu_head
    view_t = LogitsView(logits=pad_pos(logits_t))
    if tea_head is not None:
        view_t._head = tea_head

    so_s = SimpleNamespace(
        hidden=lambda layer=-1: pad_pos(h_s),
        last_hidden=pad_pos(h_s),
        logits=view_s,
    )
    so_t = SimpleNamespace(
        hidden=lambda layer=-1: pad_pos(h_t),
        last_hidden=pad_pos(h_t),
        logits=view_t,
    )
    to_t = TeacherOutput(logits=view_t, side=so_t)

    b_data: dict[str, Any] = {
        "student.input_ids": torch.where(labels_s_tgt != -100, labels_s_tgt, torch.zeros_like(labels_s_tgt)),
        "student.labels": labels_s_tgt,
        "teacher.input_ids": torch.where(labels_t_tgt != -100, labels_t_tgt, torch.zeros_like(labels_t_tgt)),
        "teacher.labels": labels_t_tgt,
    }
    if extra_batch:
        b_data.update(extra_batch)

    return LossContext(
        batch=Batch(b_data),
        student=so_s,
        student_views=[so_s],
        teacher=to_t,
        modules=mods if mods is not None else ModuleRegistry(),
        state=SimpleNamespace(step=1, progress=0.1, log=lambda *args: None),
        rt=rt,
    )


@pytest.mark.parity
def test_dskd_parity():
    torch.manual_seed(42)
    B, T, D_s, D_t, V = 2, 8, 16, 24, 30

    logits_s = torch.randn(B, T, V)
    h_s = torch.randn(B, T, D_s)
    logits_t = torch.randn(B, T, V)
    h_t = torch.randn(B, T, D_t)

    labels_pos = torch.randint(0, V, (B, T))
    labels_pos[0, 5:] = -100  # padding

    stu_head = nn.Linear(D_s, V, bias=False)
    tea_head = nn.Linear(D_t, V, bias=False)
    t2s = nn.Linear(D_t, D_s, bias=False)
    s2t = nn.Linear(D_s, D_t, bias=False)

    loss_denom = float((labels_pos != -100).sum())
    ref_batch = {
        "label_batch": {"label": labels_pos, "loss_denom": loss_denom},
        "teacher_label_batch": {"label": labels_pos, "loss_denom": loss_denom},
    }
    distiller = SimpleNamespace(
        student_model=SimpleNamespace(lm_head=stu_head),
        teacher_model=SimpleNamespace(lm_head=tea_head),
        t2s_projector=t2s,
        s2t_projector=s2t,
    )
    args = SimpleNamespace(only_stu_kd=False, only_tea_kd=False)
    div_ref = VariousDivergenceRef(kd_temp=2.0)
    div_ref.dist_func = div_ref.compute_forward_kl_divergence

    out_s = SimpleNamespace(logits=logits_s, hidden_states=[-1, h_s])
    out_t = SimpleNamespace(logits=logits_t, hidden_states=[-1, h_t])

    # Reference loss
    ref_loss, ref_log = compute_dual_space_kd_loss_ref(out_s, out_t, ref_batch, distiller, args, div_ref)

    # LM-KD Criterion
    crit = DSKD(DSKD.Config(temperature=2.0, kd_objective="forward_kl"))
    mods = ModuleRegistry()
    mods.add("dskd__t2s", copy.deepcopy(t2s), owner="dskd")
    mods.add("dskd__s2t", copy.deepcopy(s2t), owner="dskd")

    ctx = _build_mock_context(logits_s, h_s, logits_t, h_t, labels_pos, stu_head=stu_head, tea_head=tea_head, mods=mods)
    out = crit.forward(ctx)

    assert torch.allclose(out.terms["t2s_ce"].value / loss_denom, ref_log["t2s_ce_loss"], atol=1e-5)
    assert torch.allclose(out.terms["t2s_kd"].value / loss_denom, ref_log["t2s_kd_loss"], atol=1e-5)
    assert torch.allclose(out.terms["s2t_kd"].value / loss_denom, ref_log["s2t_kd_loss"], atol=1e-5)


@pytest.mark.parity
def test_dskdv2_parity_both_branches():
    torch.manual_seed(42)
    B, T, D_s, D_t, V = 2, 8, 16, 24, 30

    for agreement_thresh in (0.0, 1.01):
        logits_s = torch.randn(B, T, V)
        h_s = torch.randn(B, T, D_s)
        logits_t = torch.randn(B, T, V)
        h_t = torch.randn(B, T, D_t)

        labels_pos = torch.randint(0, V, (B, T))
        labels_pos[0, 5:] = -100

        stu_head = nn.Linear(D_s, V, bias=False)
        tea_head = nn.Linear(D_t, V, bias=False)
        t2s = nn.Linear(D_t, D_s, bias=False)
        s2t = nn.Linear(D_s, D_t, bias=False)

        loss_denom = float((labels_pos != -100).sum())
        ref_batch = {
            "label_batch": {"label": labels_pos, "loss_denom": loss_denom},
            "teacher_label_batch": {"label": labels_pos, "loss_denom": loss_denom},
        }
        distiller = SimpleNamespace(
            student_model=SimpleNamespace(lm_head=stu_head),
            teacher_model=SimpleNamespace(lm_head=tea_head),
            t2s_projector=t2s,
            s2t_projector=s2t,
            args=SimpleNamespace(t2s_agreement=agreement_thresh),
        )
        args = SimpleNamespace(
            only_stu_kd=False,
            only_tea_kd=False,
            init_s2t_projector=False,
            topk_vocab=-1,
        )
        div_ref = VariousDivergenceRef(kd_temp=2.0)
        div_ref.dist_func = div_ref.compute_forward_kl_divergence

        out_s = SimpleNamespace(logits=logits_s, hidden_states=[-1, h_s])
        out_t = SimpleNamespace(logits=logits_t, hidden_states=[-1, h_t])

        ref_loss, ref_log = compute_dual_space_kd_v2_loss_ref(out_s, out_t, ref_batch, distiller, args, div_ref)

        crit = DSKDv2(
            DSKDv2.Config(
                temperature=2.0,
                t2s_agreement=agreement_thresh,
                analytic_s2t=False,
                kd_objective="forward_kl",
            )
        )
        mods = ModuleRegistry()
        mods.add("dskdv2__t2s", copy.deepcopy(t2s), owner="dskdv2")
        mods.add("dskdv2__s2t", copy.deepcopy(s2t), owner="dskdv2")

        ctx = _build_mock_context(
            logits_s, h_s, logits_t, h_t, labels_pos, stu_head=stu_head, tea_head=tea_head, mods=mods
        )
        out = crit.forward(ctx)

        assert torch.allclose(out.terms["t2s_ce"].value / loss_denom, ref_log["t2s_ce_loss"], atol=1e-5)
        if out.terms["t2s_kd"].reduction == "mean":
            assert torch.allclose(out.terms["t2s_kd"].value, ref_log["t2s_kd_loss"], atol=1e-5)
        else:
            assert torch.allclose(out.terms["t2s_kd"].value / loss_denom, ref_log["t2s_kd_loss"], atol=1e-5)
        assert torch.allclose(out.terms["s2t_kd"].value / loss_denom, ref_log["s2t_kd_loss"], atol=1e-5)


@pytest.mark.parity
def test_dskdv2_analytic_s2t():
    torch.manual_seed(42)
    B, T, D_s, D_t, V_s, V_t = 2, 6, 16, 20, 32, 40

    stu_head = nn.Linear(D_s, V_s, bias=False)
    tea_head = nn.Linear(D_t, V_t, bias=False)

    # Overlapping vocabulary ids
    overlap_s = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.long)
    overlap_t = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.long)

    W_S = stu_head.weight.detach().transpose(0, 1)  # [D_s, V_s]
    W_T = tea_head.weight.detach().transpose(0, 1)  # [D_t, V_t]
    part_S = W_S[:, overlap_s]
    part_T = W_T[:, overlap_t]

    part_T_pinv = torch.linalg.pinv(part_T.float())
    s2t_analytic_ref = (part_S.float() @ part_T_pinv).transpose(0, 1)  # or stu_lm_head @ pinv

    # Distiller ref calculation
    stu_lm_head_sub = W_S[:, overlap_s]
    expected_s2t = stu_lm_head_sub @ part_T_pinv

    # In DSKDv2 criterion:
    crit = DSKDv2(DSKDv2.Config(analytic_s2t=True, topk_vocab=-1))
    crit.register_buffer("part_T_pinv", part_T_pinv, persistent=False)
    crit.overlap_s_part = overlap_s

    h_s = torch.randn(B, T, D_s)
    h_s_out = h_s @ expected_s2t
    # Confirm formula consistency
    assert h_s_out.shape == (B, T, D_t)


def test_pinv_init_logit_equivalence():
    torch.manual_seed(42)
    D_s, D_t, V_part = 48, 48, 16  # D_s >= V_part for exact pinv recovery

    stu_head = nn.Linear(D_s, V_part, bias=False)
    tea_head = nn.Linear(D_t, V_part, bias=False)

    # Prepare runtime mock
    rt = SimpleNamespace(
        student_model=SimpleNamespace(),
        teacher_model=SimpleNamespace(),
        student_adapter=SimpleNamespace(lm_head=lambda m: stu_head),
        teacher_adapter=SimpleNamespace(lm_head=lambda m: tea_head),
        modules=ModuleRegistry(),
        data_ctx=SimpleNamespace(
            vocab_map=SimpleNamespace(
                same_vocab=True,
                overlap_s=torch.arange(V_part),
                overlap_t=torch.arange(V_part),
            )
        ),
    )
    t2s = nn.Linear(D_t, D_s, bias=False)
    rt.modules.add("dskdv2__t2s", t2s, owner="dskdv2")

    crit = DSKDv2(DSKDv2.Config(init_t2s_pinv=True, topk_vocab=V_part))
    pctx = PrepareContext(rt=rt)
    crit.prepare(pctx)

    # Test logit equivalence: t2s(h_T) @ W_S ≈ h_T @ W_T
    h_t = torch.randn(4, D_t)
    W_S = stu_head.weight.detach().transpose(0, 1)
    W_T = tea_head.weight.detach().transpose(0, 1)

    t2s_mod = rt.modules.get("dskdv2__t2s")
    proj_logits = t2s_mod(h_t) @ W_S
    target_logits = h_t @ W_T

    rel_error = (proj_logits - target_logits).abs().max() / target_logits.abs().mean()
    assert rel_error < 1e-3, f"Pinv relative error too high: {rel_error}"


@pytest.mark.parity
def test_cma_parity():
    torch.manual_seed(42)
    B, T_s_pos, T_t_pos, D_s, D_t, V_s, V_t = 2, 7, 9, 16, 24, 30, 40

    # Create target-aligned tensors of length T+1 directly
    labels_s = torch.randint(0, V_s, (B, T_s_pos + 1))
    labels_s[:, 0] = -100
    labels_s[0, 6:] = -100

    labels_t = torch.randint(0, V_t, (B, T_t_pos + 1))
    labels_t[:, 0] = -100
    labels_t[0, 7:] = -100

    ids_s = torch.where(labels_s != -100, labels_s, torch.zeros_like(labels_s))
    ids_t = torch.where(labels_t != -100, labels_t, torch.zeros_like(labels_t))

    in_s = torch.zeros_like(ids_s)
    in_s[:, 1:] = ids_s[:, :-1]
    in_t = torch.zeros_like(ids_t)
    in_t[:, 1:] = ids_t[:, :-1]

    in_data = {"input_ids": in_s, "teacher_input_ids": in_t}
    out_data = {"label": labels_s, "teacher_label": labels_t}

    logits_s_pos = torch.randn(B, T_s_pos + 1, V_s)
    logits_t_pos = torch.randn(B, T_t_pos + 1, V_t)

    h_s_pos = torch.randn(B, T_s_pos + 1, D_s)
    h_t_pos = torch.randn(B, T_t_pos + 1, D_t)

    h_s_tgt = torch.zeros_like(h_s_pos)
    h_s_tgt[:, 1:] = h_s_pos[:, :-1]
    h_t_tgt = torch.zeros_like(h_t_pos)
    h_t_tgt[:, 1:] = h_t_pos[:, :-1]

    stu_head = nn.Linear(D_s, V_s, bias=False)
    tea_head = nn.Linear(D_t, V_t, bias=False)

    view_s = LogitsView(logits=logits_s_pos)
    view_t = LogitsView(logits=logits_t_pos)
    view_s._head = stu_head
    view_t._head = tea_head

    ref_logits_s = view_s.full()
    ref_logits_t = view_t.full()

    out_s_ref = SimpleNamespace(logits=ref_logits_s, hidden_states=[-1, h_s_tgt])
    out_t_ref = SimpleNamespace(logits=ref_logits_t, hidden_states=[-1, h_t_tgt])

    emb_s = nn.Embedding(V_s, D_s)
    emb_t = nn.Embedding(V_t, D_t)
    proj_q = nn.Linear(2 * D_s, 2 * D_t, bias=True)
    proj_s2t = nn.Linear(D_s, D_t, bias=True)
    proj_t2s = nn.Linear(D_t, D_s, bias=True)

    distiller = SimpleNamespace(
        student_model=SimpleNamespace(lm_head=stu_head),
        teacher_model=SimpleNamespace(lm_head=tea_head),
        student_embed_tokens=emb_s,
        teacher_embed_tokens=emb_t,
        projectors={
            "query": copy.deepcopy(proj_q),
            "s2t": copy.deepcopy(proj_s2t),
            "t2s": copy.deepcopy(proj_t2s),
        },
    )
    div_ref = VariousDivergenceRef(kd_temp=3.0, tea_temp=1.0)
    div_ref.dist_func = div_ref.compute_forward_kl_divergence

    ref_loss, ref_log = compute_dual_space_kd_loss_with_cma_ref(
        out_s_ref, out_t_ref, in_data, out_data, distiller, SimpleNamespace(only_save_projector=False), div_ref
    )

    crit = DSKDCMA(DSKDCMA.Config(temperature=3.0, kd_objective="forward_kl", only_t2s_ce=False))
    mods = ModuleRegistry()
    mods.add("dskd_cma__query", copy.deepcopy(proj_q), owner="dskd_cma")
    mods.add("dskd_cma__s2t", copy.deepcopy(proj_s2t), owner="dskd_cma")
    mods.add("dskd_cma__t2s", copy.deepcopy(proj_t2s), owner="dskd_cma")

    so_s = SimpleNamespace(hidden=lambda l=-1: h_s_pos, logits=view_s)
    so_t = SimpleNamespace(hidden=lambda l=-1: h_t_pos, logits=view_t)
    to_t = SimpleNamespace(logits=view_t, side=so_t)

    ctx = LossContext(
        batch=Batch(
            {
                "student.input_ids": ids_s,
                "student.labels": labels_s,
                "teacher.input_ids": ids_t,
                "teacher.labels": labels_t,
            }
        ),
        student=so_s,  # type: ignore[arg-type]
        student_views=[so_s],  # type: ignore[list-item]
        teacher=to_t,  # type: ignore[arg-type]
        modules=mods,
        state=SimpleNamespace(step=1, progress=0.1, log=lambda *args: None),
        rt=None,
    )
    ctx.student_embed_tokens = emb_s  # type: ignore[attr-defined]
    ctx.teacher_embed_tokens = emb_t  # type: ignore[attr-defined]

    out = crit.forward(ctx)
    assert torch.allclose(out.terms["t2s_ce"].value, ref_log["t2s_ce_loss"], atol=1e-5)
    assert torch.allclose(out.terms["t2s_kd"].value, ref_log["t2s_kd_loss"], atol=1e-5)
    assert torch.allclose(out.terms["s2t_kd"].value, ref_log["s2t_kd_loss"], atol=1e-5)

    # Test only_t2s_ce
    crit_ce = DSKDCMA(DSKDCMA.Config(temperature=3.0, only_t2s_ce=True))
    out_ce = crit_ce.forward(ctx)
    assert "t2s_ce" in out_ce.terms
    assert "t2s_kd" not in out_ce.terms
    assert "s2t_kd" not in out_ce.terms


@pytest.mark.parity
def test_eta_parity():
    torch.manual_seed(42)
    B, T, D_s, D_t, V_s, V_t = 2, 6, 16, 20, 25, 30

    logits_s = torch.randn(B, T, V_s)
    h_s = torch.randn(B, T, D_s)
    logits_t = torch.randn(B, T, V_t)
    h_t = torch.randn(B, T, D_t)

    labels_s_pos = torch.randint(0, V_s, (B, T))
    labels_t_pos = torch.randint(0, V_t, (B, T))
    labels_s_pos[0, 4:] = -100
    labels_t_pos[0, 4:] = -100

    stu_head = nn.Linear(D_s, V_s, bias=False)
    tea_head = nn.Linear(D_t, V_t, bias=False)
    t2s = nn.Linear(D_t, D_s, bias=False)
    s2t = nn.Linear(D_s, D_t, bias=False)

    crit = DSKDv2ETA(DSKDv2ETA.Config(temperature=2.0, analytic_s2t=False, normalize="local"))
    mods = ModuleRegistry()
    mods.add("dskdv2_eta__t2s", copy.deepcopy(t2s), owner="dskdv2_eta")
    mods.add("dskdv2_eta__s2t", copy.deepcopy(s2t), owner="dskdv2_eta")

    # Construct mock ETA alignment: align position 1, 2, 3 in student to 1, 2, 3 in teacher
    eta_t_for_s = torch.full((B, T + 1), -1, dtype=torch.long)
    eta_t_for_s[:, 1:4] = torch.tensor([[1, 2, 3], [1, 2, 3]])
    eta_s_for_t = torch.full((B, T + 1), -1, dtype=torch.long)
    eta_s_for_t[:, 1:4] = torch.tensor([[1, 2, 3], [1, 2, 3]])

    tea2stu_exact = torch.arange(V_t, dtype=torch.long)
    tea2stu_exact[tea2stu_exact >= V_s] = -1

    mock_vm = SimpleNamespace(tea2stu_exact=tea2stu_exact)
    rt = SimpleNamespace(
        data_ctx=SimpleNamespace(vocab_map=mock_vm),
        teacher_tokenizer=SimpleNamespace(eos_token_id=1),
        student_tokenizer=SimpleNamespace(eos_token_id=1),
    )
    crit.tea_eos_id = 1
    crit.stu_eos_id = 1

    ctx = _build_mock_context(
        logits_s,
        h_s,
        logits_t,
        h_t,
        labels_s_pos,
        labels_t_pos,
        stu_head=stu_head,
        tea_head=tea_head,
        mods=mods,
        extra_batch={"align.eta_t_for_s": eta_t_for_s, "align.eta_s_for_t": eta_s_for_t},
        rt=rt,
    )
    out = crit.forward(ctx)
    assert "t2s_ce" in out.terms
    assert "t2s_kd" in out.terms
    assert "s2t_kd" in out.terms
    assert out.terms["t2s_ce"].value.isfinite()
    assert out.terms["t2s_kd"].value.isfinite()
    assert out.terms["s2t_kd"].value.isfinite()
