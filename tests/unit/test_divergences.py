from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.parity._ref.opsd_gjsd import generalized_jsd_loss as opsd_ref_gjsd

from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.losses.functional.divergences import (
    DIVERGENCES,
    get_divergence,
)
from lmkd.losses.token_kd import TokenKD
from lmkd.losses.topk_kd import TopKKD
from lmkd.models.logits_view import LogitsView, chunked_token_divergence
from lmkd.teachers.providers import _fill_from_view
from lmkd.teachers.signals import TeacherOutput, TeacherSignalSpec


def test_all_registered():
    required = {"fkl", "rkl", "symkl", "jsd", "tvd", "sfkl", "srkl", "csd", "gjsd"}
    assert required.issubset(set(DIVERGENCES.keys()))


def test_nonneg_and_zero_at_equal():
    torch.manual_seed(42)
    s = torch.randn(2, 4, 32)
    t = s.clone()

    rkl_fn = get_divergence("rkl")
    tvd_fn = get_divergence("tvd")
    jsd_fn = get_divergence("jsd", lam=0.5)
    gjsd_fn = get_divergence("gjsd", beta=0.5)

    assert torch.allclose(rkl_fn(s, t), torch.zeros(2, 4), atol=1e-6)
    assert torch.allclose(tvd_fn(s, t), torch.zeros(2, 4), atol=1e-6)
    assert torch.allclose(jsd_fn(s, t), torch.zeros(2, 4), atol=1e-6)
    assert torch.allclose(gjsd_fn(s, t), torch.zeros(2, 4), atol=1e-6)

    # When s != t
    t2 = torch.randn(2, 4, 32)
    tvd_val = tvd_fn(s, t2)
    assert (tvd_val >= 0.0).all() and (tvd_val <= 1.0 + 1e-6).all()
    jsd_val = jsd_fn(s, t2)
    assert (jsd_val >= -1e-6).all()


def test_jsd_equals_gjsd():
    torch.manual_seed(42)
    s = torch.randn(2, 4, 32)
    t = torch.randn(2, 4, 32)

    # jsd(lam=0.9) corresponds to teacher weight (1 - 0.9) = 0.1 -> gjsd(beta=0.1)
    jsd = get_divergence("jsd", lam=0.9)(s, t)
    gjsd = get_divergence("gjsd", beta=0.1)(s, t)
    assert torch.allclose(jsd, gjsd, atol=1e-5)


def test_gjsd_endpoints():
    torch.manual_seed(42)
    s = torch.randn(2, 4, 32)
    t = torch.randn(2, 4, 32)

    # beta=1 is reverse KL
    gjsd_1 = get_divergence("gjsd", beta=1.0)(s, t)
    rkl = get_divergence("rkl")(s, t)
    assert torch.allclose(gjsd_1, rkl, atol=1e-5)

    # beta=0 is full KL(T || S) = fkl + sum p * lp (teacher entropy)
    gjsd_0 = get_divergence("gjsd", beta=0.0)(s, t)
    fkl = get_divergence("fkl")(s, t)
    p = F.softmax(t, dim=-1)
    lp = F.log_softmax(t, dim=-1)
    ent_t = (p * lp).sum(-1)
    assert torch.allclose(gjsd_0, fkl + ent_t, atol=1e-5)


def test_csd_grad_equals_rkl_grad():
    torch.manual_seed(42)
    s1 = torch.randn(2, 4, 32, requires_grad=True)
    t = torch.randn(2, 4, 32)
    s2 = s1.clone().detach().requires_grad_(True)

    csd_val = get_divergence("csd")(s1, t).sum()
    rkl_val = get_divergence("rkl")(s2, t).sum()

    grad_csd = torch.autograd.grad(csd_val, s1)[0]
    grad_rkl = torch.autograd.grad(rkl_val, s2)[0]

    assert torch.allclose(grad_csd, grad_rkl, atol=1e-5)


def test_elementwise_vs_per_token_clip():
    torch.manual_seed(42)
    s = torch.randn(2, 4, 32)
    t = torch.randn(2, 4, 32)

    # Without clip, elementwise and per_token are identical
    gjsd_none_elem = get_divergence("gjsd", beta=0.5, token_clip=None, clip_mode="elementwise")(s, t)
    gjsd_none_tok = get_divergence("gjsd", beta=0.5, token_clip=None, clip_mode="per_token")(s, t)
    assert torch.allclose(gjsd_none_elem, gjsd_none_tok, atol=1e-6)

    # With clip 0.01, elementwise clip happens before sum over V, per_token happens after
    gjsd_elem = get_divergence("gjsd", beta=0.5, token_clip=0.01, clip_mode="elementwise")(s, t)
    gjsd_tok = get_divergence("gjsd", beta=0.5, token_clip=0.01, clip_mode="per_token")(s, t)
    assert not torch.allclose(gjsd_elem, gjsd_tok, atol=1e-4)


def test_token_kd_config_validation():
    with pytest.raises(ValueError, match="token_clip"):
        TokenKD.Config(divergence="fkl", token_clip=0.05)


def test_topk_renorm_equals_opsd_topk():
    torch.manual_seed(42)
    B, T, V, K = 2, 8, 32, 5
    s_logits = torch.randn(B, T, V)
    t_logits = torch.randn(B, T, V)

    # Target-aligned labels: row 0 is masked (-100), others have valid token ids
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    labels[:, 1:3] = -100  # prompt mask

    # LogitsView from raw logits
    # Since LogitsView target-aligns slice: pos t uses t-1.
    # To compare raw logits directly with OPSD ref, let's use LogitsView(logits=...)
    s_view = LogitsView(logits=s_logits)
    t_view = LogitsView(logits=t_logits)

    spec = TeacherSignalSpec(kind="topk", k=K, support="teacher", tail="none", temperature=1.0)
    batch = Batch({"student.labels": labels, "student.attention_mask": torch.ones(B, T, dtype=torch.long)})

    t_out = TeacherOutput(logits=t_view)
    _fill_from_view(t_out, batch, spec, "student", None)

    ctx = LossContext(
        batch=batch,
        student=SimpleNamespace(logits=s_view),
        student_views=[],
        teacher=t_out,
        modules=None,
        state=TrainState(),
        rt=None,
    )

    topk_kd = TopKKD(TopKKD.Config(k=K, support="teacher", tail="renorm", alpha=0.0, temperature=1.0))
    loss_out = topk_kd(ctx)

    # Target-aligned target slice in LogitsView: row t uses pos t-1.
    # So the logits evaluated at target t correspond to s_logits[:, t-1] (for t >= 1).
    # Let's extract the target-aligned slices from s_view and t_view to pass to OPSD ref:
    s_target_aligned = s_view.full()
    t_target_aligned = t_view.full()

    opsd_loss = opsd_ref_gjsd(
        s_target_aligned,
        t_target_aligned,
        labels=labels,
        beta=0.0,
        temperature=1.0,
        reduction="batchmean",
        top_k=K,
    )

    mask = labels != -100
    our_loss = loss_out.terms["topk_kd"].value / mask.sum()
    assert torch.allclose(our_loss, opsd_loss, atol=1e-5)


@pytest.mark.parametrize("name", ["fkl", "rkl", "symkl", "jsd", "tvd", "sfkl", "srkl", "csd", "gjsd"])
def test_chunked_autograd_matches_full(name):
    torch.manual_seed(42)
    B, T, D, V = 2, 8, 16, 20
    h_s = torch.randn(B, T, D, requires_grad=True)
    head_s = nn.Linear(D, V, bias=False)
    s_view = LogitsView(h_s, head_s)

    h_t = torch.randn(B, T, D)
    head_t = nn.Linear(D, V, bias=False)
    t_view = LogitsView(h_t, head_t)

    mask = torch.ones(B, T)
    mask[:, :2] = 0

    fn = get_divergence(name)

    # Chunked forward + backward (invoking custom ChunkedHeadFunction autograd)
    loss_chunk = chunked_token_divergence(s_view, t_view, fn, mask=mask, chunk_size=4).sum()
    loss_chunk.backward()
    grad_h_chunk = h_s.grad.clone()
    grad_w_chunk = head_s.weight.grad.clone()

    h_s.grad = None
    head_s.weight.grad = None

    # Full (un-chunked) forward + backward
    s_full = s_view.full()
    t_full = t_view.full().detach()
    loss_full = (fn(s_full, t_full) * mask).sum()
    loss_full.backward()
    grad_h_full = h_s.grad.clone()
    grad_w_full = head_s.weight.grad.clone()

    assert torch.allclose(loss_chunk, loss_full, atol=1e-5)
    assert torch.allclose(grad_h_chunk, grad_h_full, atol=1e-5)
    assert torch.allclose(grad_w_chunk, grad_w_full, atol=1e-5)


@pytest.mark.parametrize("name", ["fkl", "rkl", "symkl", "jsd", "tvd", "sfkl", "srkl", "csd", "gjsd"])
def test_divergences_temperature_scaling(name):
    torch.manual_seed(42)
    s = torch.randn(2, 4, 16)
    t = torch.randn(2, 4, 16)

    # 1. Default temperature is 1.0
    fn_default = get_divergence(name)
    val_default = fn_default(s, t)
    fn_temp_1 = get_divergence(name, temperature=1.0)
    torch.testing.assert_close(val_default, fn_temp_1(s, t))

    # 2. Temperature parameter scales logits
    tau = 2.5
    fn_temp = get_divergence(name, temperature=tau)
    val_temp = fn_temp(s, t)

    # 3. Direct scaling of input logits equals calling with temperature
    if name == "csd":
        sp = F.softmax(s / tau, dim=-1, dtype=torch.float32)
        expected = (s - t - torch.sum(sp * (s - t), dim=-1, keepdim=True)).detach() * sp.detach() * s
        torch.testing.assert_close(val_temp, expected.sum(-1), atol=1e-5, rtol=1e-5)
    else:
        val_scaled = fn_default(s / tau, t / tau)
        torch.testing.assert_close(val_temp, val_scaled, atol=1e-5, rtol=1e-5)


def test_invalid_temperature():
    with pytest.raises(ValueError, match="positive"):
        get_divergence("fkl", temperature=-1.0)

    with pytest.raises(ValueError, match="positive"):
        get_divergence("csd", temperature=0.0)


def test_csd_user_formula_parity():
    torch.manual_seed(42)
    s = torch.randn(3, 8, 32)
    t = torch.randn(3, 8, 32)
    labels = torch.randint(0, 32, (3, 8))
    labels[:, :2] = -100
    mask = (labels != -100).int()

    for temp in [0.5, 1.0, 1.5, 2.0]:
        # User original formula
        sp = F.softmax(s / temp, dim=-1, dtype=torch.float32)
        ref_loss = (s - t - torch.sum(sp * (s - t), dim=-1, keepdim=True)).detach() * sp.detach() * s
        ref_x = torch.sum(ref_loss, dim=-1).view(-1)
        ref_distil = torch.sum(ref_x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        # LM-KD csd divergence with temperature
        our_per_tok = get_divergence("csd", temperature=temp)(s, t)
        our_distil = (our_per_tok * mask).sum() / mask.sum()

        torch.testing.assert_close(our_distil, ref_distil, atol=1e-6, rtol=1e-5)
