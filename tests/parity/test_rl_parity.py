"""T25: compare against verbatim SDPO/RLSD source, including gradients."""

from types import SimpleNamespace

import pytest
import torch
from tests.parity._ref.rlsd_ref import _build_stgca_advantages
from tests.parity._ref.sdpo_ref import compute_self_distillation_loss

from baselines.self_distill import add_tail, rlsd_reweight, sdpo_per_token

pytestmark = pytest.mark.parity


@pytest.mark.parametrize("negative_only", [False, True])
@pytest.mark.parametrize("thought_only", [False, True])
def test_rlsd_reweight_parity(negative_only, thought_only):
    torch.manual_seed(25)
    adv, teacher, student = (torch.randn(4, 9) for _ in range(3))
    mask = torch.randint(2, (4, 9)).float()
    thought = torch.randint(2, (4, 9)).float() * mask if thought_only else None
    expected, _ = _build_stgca_advantages(
        teacher_log_probs=teacher,
        student_log_probs=student,
        advantages=adv,
        response_mask=mask,
        lam=0.5,
        clip_range=0.2,
        negative_only=negative_only,
        thought_mask=thought,
    )
    actual, _ = rlsd_reweight(adv, teacher, student, mask, 0.5, 0.2, negative_only, thought)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("tail", [False, True])
def test_sdpo_loss_parity(alpha, tail):
    torch.manual_seed(25)
    raw = torch.randn(3, 5, 11, requires_grad=True)
    s = raw.log_softmax(-1)
    t = torch.randn_like(raw).log_softmax(-1)
    ids = s.topk(4, -1).indices
    sk, tk = s.gather(-1, ids), t.gather(-1, ids)
    mask = torch.randint(2, (3, 5)).float()
    signal = torch.tensor([1.0, 0.0, 1.0])
    old = torch.randn(3, 5)
    weights = torch.rand(3, 5)
    cfg = SimpleNamespace(
        alpha=alpha, full_logit_distillation=True, distillation_topk=4, distillation_add_tail=tail, is_clip=2.0
    )
    expected, _ = compute_self_distillation_loss(
        s[..., 0],
        t[..., 0],
        mask,
        cfg,
        old,
        student_topk_log_probs=sk,
        teacher_topk_log_probs=tk,
        self_distillation_mask=signal,
        rollout_is_weights=weights,
    )
    a, b = (
        (add_tail(sk), add_tail(tk))
        if tail
        else (sk - sk.logsumexp(-1, keepdim=True), tk - tk.logsumexp(-1, keepdim=True))
    )
    per_token = sdpo_per_token(a, b, alpha)
    ratio = (s[..., 0].detach() - old).clamp(-20, 20).exp().clamp(max=2.0)
    loss_mask = mask * signal[:, None]
    actual = (per_token * ratio * weights * loss_mask).sum() / loss_mask.sum().clamp_min(1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    ga = torch.autograd.grad(actual, raw, retain_graph=True)[0]
    gb = torch.autograd.grad(expected, raw)[0]
    torch.testing.assert_close(ga, gb, atol=1e-6, rtol=0)
