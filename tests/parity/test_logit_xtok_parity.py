"""Parity tests for ULD, MinED, MultiLevelOT, and E-SCOT against reference implementations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tests.parity._ref.escot_ref import compute_escot_loss_ref
from tests.parity._ref.mined_ref import (
    compute_mined_loss_ref,
    get_aligned_teacher_logits_ref,
)
from tests.parity._ref.mlot_ref import compute_multi_level_ot_loss_ref
from tests.parity._ref.uld_ref import compute_universal_logit_distillation_loss_ref
from tests.parity._xtok_utils import pad_pos, pos_to_tgt_labels

from baselines.cross_tokenizer.escot import ESCOT
from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext
from lmkd.losses.cross_tokenizer.mined import MinEDFKL
from lmkd.losses.cross_tokenizer.mlot import MultiLevelOT
from lmkd.losses.cross_tokenizer.uld import ULD
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


def _build_context(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    labels_pos_s: torch.Tensor,
    labels_pos_t: torch.Tensor | None = None,
    extra_batch: dict[str, Any] | None = None,
    rt: Any | None = None,
) -> tuple[LossContext, torch.Tensor, torch.Tensor]:
    """Builds a LossContext where logits_s and logits_t are positional [B, T, V]."""
    if labels_pos_t is None:
        labels_pos_t = labels_pos_s

    # In target-aligned LM-KD, row 0 is dummy and target t corresponds to positional step t-1
    labels_s_tgt = pos_to_tgt_labels(labels_pos_s)
    labels_t_tgt = pos_to_tgt_labels(labels_pos_t)

    # pad_pos appends dummy step so LogitsView.at([[b, t]]) gets logits[:, t-1]
    view_s = LogitsView(logits=pad_pos(logits_s))
    view_t = LogitsView(logits=pad_pos(logits_t))
    so_s = SimpleNamespace(logits=view_s)
    so_t = SimpleNamespace(logits=view_t)
    to_t = TeacherOutput(logits=view_t, side=so_t)

    b_data: dict[str, Any] = {
        "student.input_ids": torch.where(labels_s_tgt != -100, labels_s_tgt, torch.zeros_like(labels_s_tgt)),
        "student.labels": labels_s_tgt,
        "teacher.input_ids": torch.where(labels_t_tgt != -100, labels_t_tgt, torch.zeros_like(labels_t_tgt)),
        "teacher.labels": labels_t_tgt,
    }
    if extra_batch:
        b_data.update(extra_batch)

    ctx = LossContext(
        batch=Batch(b_data),
        student=so_s,
        student_views=[so_s],
        teacher=to_t,
        modules=ModuleRegistry(),
        state=SimpleNamespace(step=1, progress=0.1, log=lambda *args: None),
        rt=rt,
    )
    return ctx, labels_s_tgt, labels_t_tgt


@pytest.mark.parity
def test_uld_parity():
    """Random batch with heterogeneous prompt lengths, -100 padding, different V:
    matches reference implementation (atol 1e-5) without mutating inputs.
    """
    torch.manual_seed(42)
    B, T, V_s, V_t = 3, 10, 32, 48

    logits_s = torch.randn(B, T, V_s)
    logits_t = torch.randn(B, T, V_t)
    s_clone = logits_s.clone()
    t_clone = logits_t.clone()

    # Positional labels
    labels_s = torch.randint(0, V_s, (B, T))
    labels_t = torch.randint(0, V_t, (B, T))

    # Mask prompts and padding
    # Sample 0: plen=2, valid=5 -> targets 2..6, pad 7..9
    # Sample 1: plen=3, valid=4 -> targets 3..6, pad 7..9
    # Sample 2: plen=1, valid=7 -> targets 1..7, pad 8..9
    plens_s = [2, 3, 1]
    plens_t = [2, 3, 1]
    for b in range(B):
        ps = plens_s[b]
        labels_s[b, :ps] = -100
        labels_s[b, ps + 4 :] = -100

        pt = plens_t[b]
        labels_t[b, :pt] = -100
        labels_t[b, pt + 4 :] = -100

    ctx, labels_s_tgt, labels_t_tgt = _build_context(
        logits_s,
        logits_t,
        labels_s,
        labels_t,
        extra_batch={
            "student.prompt_len": torch.tensor([p + 1 for p in plens_s], dtype=torch.long),
            "teacher.prompt_len": torch.tensor([p + 1 for p in plens_t], dtype=torch.long),
        },
    )

    crit = ULD(ULD.Config(temperature=1.0))
    out = crit.forward(ctx)
    our_loss = out.terms["uld"].value

    # Compute ref on positional inputs
    ref_loss = compute_universal_logit_distillation_loss_ref(
        logits_s,
        logits_t,
        labels_s,
        labels_t,
        padding_id=-100,
        clone_inputs=True,
    )

    assert torch.allclose(our_loss, ref_loss, atol=1e-5), f"ULD mismatch: {our_loss.item()} vs {ref_loss.item()}"
    # Verify inputs were NOT mutated in-place
    assert torch.equal(logits_s, s_clone)
    assert torch.equal(logits_t, t_clone)


@pytest.mark.parity
def test_mlot_parity_full_valid():
    """Batch without padding or prompt (all positions valid, equal lengths):
    matches reference (atol 1e-5); legacy_token_scale=True matches multi*count.
    """
    torch.manual_seed(42)
    B, T, V_s, V_t = 2, 8, 40, 60
    top_k = 25
    temperature = 2.0

    logits_s = torch.randn(B, T, V_s)
    logits_t = torch.randn(B, T, V_t)
    s_clone = logits_s.clone()
    t_clone = logits_t.clone()

    # All positions valid (no prompt, no padding)
    labels_s = torch.randint(0, V_s, (B, T))
    labels_t = torch.randint(0, V_t, (B, T))

    ctx, _, _ = _build_context(
        logits_s,
        logits_t,
        labels_s,
        labels_t,
        extra_batch={
            "student.prompt_len": torch.tensor([1, 1], dtype=torch.long),
            "teacher.prompt_len": torch.tensor([1, 1], dtype=torch.long),
        },
    )

    # 1. Test legacy_token_scale=True against ref
    crit_legacy = MultiLevelOT(
        MultiLevelOT.Config(
            temperature=temperature,
            top_k=top_k,
            legacy_token_scale=True,
        )
    )
    out_legacy = crit_legacy.forward(ctx)
    our_legacy_loss = out_legacy.terms["multilevel_ot"].value

    ref_token_scaled, ref_ot, ref_kl, ref_l1 = compute_multi_level_ot_loss_ref(
        logits_s,
        logits_t,
        labels_s,
        labels_t,
        student_temperature=temperature,
        teacher_temperature=temperature,
        top_k=top_k,
        padding_id=-100,
        clone_inputs=True,
    )

    assert torch.allclose(our_legacy_loss, ref_token_scaled, atol=1e-5), (
        f"MLOT legacy mismatch: {our_legacy_loss.item()} vs {ref_token_scaled.item()}"
    )

    # 2. Test default (scalar mean)
    crit_default = MultiLevelOT(
        MultiLevelOT.Config(
            temperature=temperature,
            top_k=top_k,
            legacy_token_scale=False,
        )
    )
    out_default = crit_default.forward(ctx)
    our_mean_loss = out_default.terms["multilevel_ot"].value
    expected_mean = ref_ot + ref_kl + ref_l1
    assert torch.allclose(our_mean_loss, expected_mean, atol=1e-5)

    assert torch.equal(logits_s, s_clone)
    assert torch.equal(logits_t, t_clone)


@pytest.mark.parity
def test_mined_parity():
    """Simulated tokenizer with manual id-map JSON; KD matches reference including one-hot positions."""
    torch.manual_seed(42)
    B, T, V_s, V_t = 1, 5, 12, 16
    logits_s = torch.randn(B, T, V_s)
    logits_t = torch.randn(B, T, V_t)

    labels_s = torch.tensor([[1, 3, 5, 7, -100]], dtype=torch.long)
    labels_t = torch.tensor([[2, 4, 6, 8, -100]], dtype=torch.long)

    # Let student tokens be:
    s_tokens = ["Ġthe", "Ġdog", "Ġbarks", "Ġloudly", "</s>"]
    t_tokens = ["Ġthe", "Ġhound", "Ġbarks", "Ġloud", "</s>"]
    s_input_ids = torch.tensor([[1, 3, 5, 7, 0]], dtype=torch.long)
    t_input_ids = torch.tensor([[2, 4, 6, 8, 0]], dtype=torch.long)

    # stu2tea mapping:
    # 1 -> 2 ("Ġthe"), 5 -> 6 ("Ġbarks"), others -1
    stu2tea = torch.full((V_s,), -1, dtype=torch.long)
    stu2tea[1] = 2
    stu2tea[5] = 6

    tea2stu = torch.full((V_t,), -1, dtype=torch.long)
    tea2stu[2] = 1
    tea2stu[6] = 5

    # dtw alignment:
    # pos 0 ('Ġthe'): maps to teacher 0 ('Ġthe', tea id 2 -> stu id 1) -> aligned
    # pos 1 ('Ġdog'): unaligned (tea id 4 != -1, not in tea2stu) -> -1
    # pos 2 ('Ġbarks'): maps to teacher 2 ('Ġbarks', tea id 6 -> stu id 5) -> aligned
    # pos 3 ('Ġloudly'): unaligned -> -1
    dtw_t_for_s_pos = torch.tensor([[0, -1, 2, -1, -1]], dtype=torch.long)

    # Call ref get_aligned_teacher_logits
    map_s = (stu2tea >= 0).nonzero(as_tuple=False).squeeze(-1)
    map_t = stu2tea[map_s]
    ref_aligned_logits = get_aligned_teacher_logits_ref(
        student_logits=logits_s,
        teacher_logits=logits_t,
        student_input_ids=s_input_ids,
        teacher_input_ids=t_input_ids,
        student_target=labels_s,
        teacher_target=labels_t,
        s_tokens=s_tokens,
        t_tokens=t_tokens,
        blending_to_base_mapping=tea2stu,
        map_s=map_s,
        map_t=map_t,
        padding_id=-100,
    )
    ref_kd = compute_mined_loss_ref(logits_s, ref_aligned_logits, labels_s, kd_temp=1.0, padding_id=-100)

    # In LM-KD target-aligned format:
    # Prepend dummy position 0 to dtw_t_for_s
    dtw_t_for_s_tgt = torch.cat([torch.full((B, 1), -1, dtype=torch.long), dtw_t_for_s_pos + 1], dim=1)
    # where dtw_t_for_s_pos was -1, keep -1
    dtw_t_for_s_tgt[dtw_t_for_s_tgt == 0] = -1

    vocab_map = SimpleNamespace(stu2tea=stu2tea)
    rt = SimpleNamespace(data_ctx=SimpleNamespace(vocab_map=vocab_map))

    ctx, _, _ = _build_context(
        logits_s,
        logits_t,
        labels_s,
        labels_t,
        extra_batch={"align.dtw_t_for_s": dtw_t_for_s_tgt},
        rt=rt,
    )

    crit = MinEDFKL(MinEDFKL.Config(temperature=1.0))
    crit.prepare(SimpleNamespace(rt=rt))
    out = crit.forward(ctx)
    our_kd = out.terms["mined_fkl"].value

    assert torch.allclose(our_kd, ref_kd, atol=1e-5), f"MinED mismatch: {our_kd.item()} vs {ref_kd.item()}"


@pytest.mark.parity
def test_escot_parity():
    """Random logits + offsets + span_pairs -> matches compute_escot_loss reference."""
    torch.manual_seed(42)
    B, T_s, T_t, V_s, V_t = 2, 6, 8, 20, 24
    tau = 2.0
    lam = 0.25
    alphas = [0.5, 1.0, 2.0]

    logits_s = torch.randn(B, T_s, V_s)
    logits_t = torch.randn(B, T_t, V_t)

    labels_s = torch.randint(0, V_s, (B, T_s))
    labels_t = torch.randint(0, V_t, (B, T_t))

    # Mask position 0 as prompt or pad
    labels_s[:, 0] = -100
    labels_s[:, 5:] = -100
    labels_t[:, 0] = -100
    labels_t[:, 6:] = -100

    # Character offsets for positional steps 0..T-1
    # Sample 0: s_offsets: [(0,3), (3,7), (7,11), (11,15), (15,20), (20,25)]
    #           t_offsets: [(0,2), (2,5), (5,8), (8,12), (12,16), (16,20), (20,23), (23,26)]
    s_offsets_b0 = [(0, 3), (3, 7), (7, 11), (11, 15), (15, 20), (20, 25)]
    t_offsets_b0 = [(0, 2), (2, 5), (5, 8), (8, 12), (12, 16), (16, 20), (20, 23), (23, 26)]
    s_offsets_b1 = [(0, 4), (4, 9), (9, 13), (13, 18), (18, 22), (22, 26)]
    t_offsets_b1 = [(0, 3), (3, 6), (6, 10), (10, 14), (14, 18), (18, 22), (22, 25), (25, 28)]

    s_offsets_batch = [s_offsets_b0, s_offsets_b1]
    t_offsets_batch = [t_offsets_b0, t_offsets_b1]

    # Compute reference loss
    ref_loss = compute_escot_loss_ref(
        logits_s,
        logits_t,
        labels_s,
        labels_t,
        s_offsets_batch,
        t_offsets_batch,
        kd_temperature=tau,
        lam=lam,
        alphas=alphas,
        padding_id=-100,
    )

    # Now construct span_pairs for LM-KD target-aligned format:
    # Remember: ref computes span alignment on s_spans, t_spans where position p is positional.
    # In LM-KD target-aligned format, target position t = p + 1.
    from tests.parity._ref.xtok_ref import _span_align

    span_pairs_batch: list[torch.Tensor] = []
    for b in range(B):
        s_valid = (labels_s[b] != -100).nonzero(as_tuple=False).flatten().tolist()
        t_valid = (labels_t[b] != -100).nonzero(as_tuple=False).flatten().tolist()
        s_spans = [tuple(s_offsets_batch[b][p]) for p in s_valid]
        t_spans = [tuple(t_offsets_batch[b][p]) for p in t_valid]
        hoods = _span_align(s_spans, t_spans)
        pairs_b = []
        for i_loc, N_i in enumerate(hoods):
            for j_loc in N_i:
                # target position in LM-KD is p + 1
                t_s = s_valid[i_loc] + 1
                t_t = t_valid[j_loc] + 1
                pairs_b.append([t_s, t_t])
        t_pairs = torch.tensor(pairs_b, dtype=torch.long) if pairs_b else torch.zeros((0, 2), dtype=torch.long)
        span_pairs_batch.append(t_pairs)

    ctx, _, _ = _build_context(
        logits_s,
        logits_t,
        labels_s,
        labels_t,
        extra_batch={"align.span_pairs": span_pairs_batch},
    )

    crit = ESCOT(ESCOT.Config(lam=lam, alphas=alphas, temperature=tau))
    out = crit.forward(ctx)
    our_loss = out.terms["escot"].value

    assert torch.allclose(our_loss, ref_loss, atol=1e-5), f"ESCOT mismatch: {our_loss.item()} vs {ref_loss.item()}"
