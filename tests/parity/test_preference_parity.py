"""Parity tests for preference losses against original Awesome-Human-AI-Alignment implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from baselines.preference import (
    advantage_expectation_loss,
    compressed_projection_kl,
    ctpd_loss,
    dp3o_loss,
    pbsd_loss,
    ppd_loss,
    tvkd_loss,
    vpd_loss,
)
from lmkd.models.logits_view import LogitsView

ROOT = Path(__file__).resolve().parents[3] / "Awesome-Human-AI-Alignment-main" / "src"
if not ROOT.exists():
    pytest.skip(f"Original source repository not found at {ROOT}", allow_module_level=True)

sys.path.insert(0, str(ROOT))
try:
    from human_alignment.mechanisms.training.distillation_losses import (  # noqa: E402
        advantage_expectation_loss as orig_adpa_loss,
    )
    from human_alignment.mechanisms.training.distillation_losses import (
        compressed_forward_kl as orig_compressed_kl,
    )
    from human_alignment.mechanisms.training.distillation_losses import (
        ctpd_loss as orig_ctpd_loss,
    )
    from human_alignment.mechanisms.training.distillation_losses import (
        pbsd_loss as orig_pbsd_loss,
    )
    from human_alignment.mechanisms.training.distillation_losses import (
        ppd_loss as orig_ppd_loss,
    )
    from human_alignment.mechanisms.training.distillation_losses import (
        tvkd_loss as orig_tvkd_loss,
    )
    from human_alignment.mechanisms.training.distillation_losses import (
        vpd_loss as orig_vpd_loss,
    )
    from human_alignment.mechanisms.training.preference.losses import (  # noqa: E402
        dp3o_loss as orig_dp3o_loss,
    )
except ImportError as e:
    pytest.skip(f"Failed to import original baseline: {e}", allow_module_level=True)


@pytest.mark.parity
def test_vpd_ppd_parity():
    torch.manual_seed(42)
    student_scores = torch.randn(3, 4)
    teacher_scores = torch.randn(3, 4)
    beta = 2.5

    our_vpd = vpd_loss(student_scores, teacher_scores, beta=beta)
    ref_vpd = orig_vpd_loss(student_scores, teacher_scores, beta=beta)
    torch.testing.assert_close(our_vpd, ref_vpd, atol=1e-6, rtol=0)

    our_ppd = ppd_loss(student_scores, teacher_scores, beta=beta, exact_ranking_limit=8)
    ref_ppd = orig_ppd_loss(student_scores, teacher_scores, beta=beta, exact_ranking_limit=8)
    torch.testing.assert_close(our_ppd, ref_ppd, atol=1e-6, rtol=0)


@pytest.mark.parity
def test_compressed_kl_parity():
    torch.manual_seed(42)
    B, T, V, K = 2, 7, 30, 5
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    labels[0, 4] = -100
    labels[1, 2] = -100

    t_logits = torch.randn(B, T, V)
    topk_p, topk_ids = t_logits.softmax(-1).topk(K, dim=-1)
    # inject 1 id >= V to test out-of-vocab handling
    topk_ids[0, 2, 1] = V + 5
    tail_p = (1.0 - topk_p.sum(dim=-1)).clamp_min(0.0)

    shifted_probs = logits[:, :-1, :].softmax(dim=-1)
    shifted_labels = labels[:, 1:].clone()

    compressed_batch = []
    for r in range(B):
        row_list = []
        positions = shifted_labels[r].ne(-100).nonzero(as_tuple=False).flatten().tolist()
        for p_idx in positions:
            t = p_idx + 1
            row_list.append(
                {
                    "indices": topk_ids[r, t].tolist(),
                    "values": topk_p[r, t].tolist(),
                    "remaining_probs_sum": float(tail_p[r, t]),
                }
            )
        compressed_batch.append(row_list)

    ref_loss = orig_compressed_kl(shifted_probs, shifted_labels, compressed_batch)

    view = LogitsView(logits=logits)
    target_probs = torch.cat([c.logits.softmax(-1) for c in view.chunks(size=4)], dim=1)

    mask = labels.ne(-100)
    mask[:, 0] = False
    sub_probs = target_probs[mask]
    sub_ids = topk_ids[mask]
    sub_p = topk_p[mask]
    sub_tail = tail_p[mask]

    our_kl = compressed_projection_kl(sub_probs, sub_ids, sub_p, sub_tail)
    our_loss = our_kl.mean()

    torch.testing.assert_close(our_loss, ref_loss, atol=1e-6, rtol=0)


@pytest.mark.parity
@pytest.mark.parametrize("value_function", ["entropy", "soft"])
def test_tvkd_parity(value_function: str):
    torch.manual_seed(42)
    B, T, V, K = 2, 6, 20, 4
    logits_c = torch.randn(B, T, V)
    logits_r = torch.randn(B, T, V)
    labels_c = torch.randint(0, V, (B, T))
    labels_r = torch.randint(0, V, (B, T))
    labels_c[:, 0] = -100
    labels_r[:, 0] = -100
    labels_c[0, 3] = -100
    labels_r[1, 4] = -100

    t_c = torch.randn(B, T, V)
    t_r = torch.randn(B, T, V)
    p_c, ids_c = t_c.softmax(-1).topk(K, dim=-1)
    p_r, ids_r = t_r.softmax(-1).topk(K, dim=-1)
    ids_c[0, 2, 0] = V + 2  # out of vocab
    tail_c = (1.0 - p_c.sum(dim=-1)).clamp_min(0.0)
    tail_r = (1.0 - p_r.sum(dim=-1)).clamp_min(0.0)

    view_c = LogitsView(logits=logits_c)
    view_r = LogitsView(logits=logits_r)
    probs_c = torch.cat([c.logits.softmax(-1) for c in view_c.chunks(size=4)], dim=1)
    probs_r = torch.cat([c.logits.softmax(-1) for c in view_r.chunks(size=4)], dim=1)

    teacher_chosen = []
    for r in range(B):
        row_list = []
        positions = labels_c[r, 1:].ne(-100).nonzero(as_tuple=False).flatten().tolist()
        for p_idx in positions:
            t = p_idx + 1
            row_list.append(
                {
                    "indices": ids_c[r, t].tolist(),
                    "values": p_c[r, t].tolist(),
                    "remaining_probs_sum": float(tail_c[r, t]),
                }
            )
        teacher_chosen.append(row_list)

    teacher_rejected = []
    for r in range(B):
        row_list = []
        positions = labels_r[r, 1:].ne(-100).nonzero(as_tuple=False).flatten().tolist()
        for p_idx in positions:
            t = p_idx + 1
            row_list.append(
                {
                    "indices": ids_r[r, t].tolist(),
                    "values": p_r[r, t].tolist(),
                    "remaining_probs_sum": float(tail_r[r, t]),
                }
            )
        teacher_rejected.append(row_list)

    ref_loss = orig_tvkd_loss(
        logits_c[:, :-1].softmax(dim=-1),
        logits_r[:, :-1].softmax(dim=-1),
        labels_c[:, 1:],
        labels_r[:, 1:],
        teacher_chosen,
        teacher_rejected,
        value_function=value_function,
        value_discount=0.9,
        value_temperature=1.0,
        student_value_weight=0.2,
        teacher_value_weight=0.7,
    )

    our_loss = tvkd_loss(
        probs_c,
        probs_r,
        labels_c,
        labels_r,
        ids_c,
        p_c,
        tail_c,
        ids_r,
        p_r,
        tail_r,
        value_function=value_function,
        value_discount=0.9,
        value_temperature=1.0,
        student_value_weight=0.2,
        teacher_value_weight=0.7,
    )

    torch.testing.assert_close(our_loss, ref_loss, atol=1e-6, rtol=0)


@pytest.mark.parity
def test_adpa_parity():
    torch.manual_seed(42)
    B, T, V, K = 2, 5, 16, 3
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    labels[0, 2] = -100

    adv_ids = torch.randint(-1, V + 2, (B, T, K))
    adv_val = torch.randn(B, T, K)

    shifted_probs = logits[:, :-1].softmax(dim=-1)
    shifted_labels = labels[:, 1:].clone()

    advantages = []
    for r in range(B):
        row_list = []
        positions = shifted_labels[r].ne(-100).nonzero(as_tuple=False).flatten().tolist()
        for p_idx in positions:
            t = p_idx + 1
            row_list.append(
                {
                    "indices": adv_ids[r, t].tolist(),
                    "values": adv_val[r, t].tolist(),
                }
            )
        advantages.append(row_list)

    ref_loss = orig_adpa_loss(shifted_probs, shifted_labels, advantages)

    view = LogitsView(logits=logits)
    target_probs = torch.cat([c.logits.softmax(-1) for c in view.chunks(size=4)], dim=1)
    our_loss = advantage_expectation_loss(target_probs, labels, adv_ids, adv_val)

    torch.testing.assert_close(our_loss, ref_loss, atol=1e-6, rtol=0)


@pytest.mark.parity
def test_ctpd_parity():
    torch.manual_seed(42)
    B, T_orig = 2, 5
    pol_c = torch.randn(B, T_orig)
    pol_r = torch.randn(B, T_orig)
    ref_c = torch.randn(B, T_orig)
    ref_r = torch.randn(B, T_orig)

    chosen_parents = [[[1, 2], [3, 4]], [[2, 3], [4]]]
    rejected_parents = [[[1, 3], [2, 4]], [[1, 2], [3]]]
    chosen_weights = torch.tensor([[0.8, 1.2], [1.0, 0.5]])
    rejected_weights = torch.tensor([[0.9, 1.1], [0.7, 0.6]])

    ref_loss = orig_ctpd_loss(
        pol_c, pol_r, ref_c, ref_r, chosen_parents, rejected_parents, chosen_weights, rejected_weights, beta=0.5
    )

    tok_c = torch.cat([torch.zeros(B, 1), pol_c], dim=-1)
    tok_r = torch.cat([torch.zeros(B, 1), pol_r], dim=-1)
    ref_tok_c = torch.cat([torch.zeros(B, 1), ref_c], dim=-1)
    ref_tok_r = torch.cat([torch.zeros(B, 1), ref_r], dim=-1)

    groups_c = torch.tensor([[[1, 2], [3, 4]], [[2, 3], [4, -1]]], dtype=torch.long)
    groups_r = torch.tensor([[[1, 3], [2, 4]], [[1, 2], [3, -1]]], dtype=torch.long)

    our_loss = ctpd_loss(
        tok_c, tok_r, ref_tok_c, ref_tok_r, groups_c, groups_r, chosen_weights, rejected_weights, beta=0.5
    )

    torch.testing.assert_close(our_loss, ref_loss, atol=1e-6, rtol=0)


@pytest.mark.parity
def test_dp3o_parity():
    torch.manual_seed(42)
    B = 4
    chosen = torch.randn(B)
    rejected = torch.randn(B)
    reward_diff = torch.randn(B)
    beta = 0.1
    alpha = 0.3

    our_dp3o = dp3o_loss(chosen, rejected, reward_diff, beta=beta, alpha=alpha)
    ref_dp3o = orig_dp3o_loss(chosen, rejected, reward_diff, beta=beta, alpha=alpha)
    torch.testing.assert_close(our_dp3o, ref_dp3o, atol=1e-6, rtol=0)

    our_dp3o_hard = dp3o_loss(chosen, rejected, None, beta=beta, alpha=1.0)
    ref_dp3o_hard = orig_dp3o_loss(chosen, rejected, None, beta=beta, alpha=1.0)
    torch.testing.assert_close(our_dp3o_hard, ref_dp3o_hard, atol=1e-6, rtol=0)


@pytest.mark.parity
@pytest.mark.parametrize("loss_type", ["logistic_gap", "mse", "mse_to_zero"])
def test_pbsd_parity(loss_type: str):
    torch.manual_seed(42)
    B = 4
    t_on_t = torch.randn(B)
    s_on_t = torch.randn(B)
    t_on_s = torch.randn(B)
    s_on_s = torch.randn(B)
    gate = torch.tensor([1.0, 0.0, 1.0, 1.0])
    beta = 0.2

    ref_loss = orig_pbsd_loss(t_on_t, s_on_t, t_on_s, s_on_s, beta=beta, loss_type=loss_type, gate=gate)
    our_loss = pbsd_loss(t_on_t, s_on_t, t_on_s, s_on_s, beta=beta, loss_type=loss_type, gate=gate)
    torch.testing.assert_close(our_loss, ref_loss, atol=1e-6, rtol=0)
