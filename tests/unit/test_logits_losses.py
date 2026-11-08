import torch
import torch.nn as nn
import torch.nn.functional as F

from lmkd.losses.aggregator import KDMix, Norms, WeightedOutput, WeightedSum
from lmkd.losses.base import LossOutput, LossTerm
from lmkd.losses.functional.divergences import forward_kl, reverse_kl
from lmkd.models.logits_view import LogitsView, chunked_token_divergence


def _view(B=2, T=7, D=5, V=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, D, generator=g, requires_grad=True)
    head = nn.Linear(D, V, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.randn(V, D, generator=g))
    return h, head, LogitsView(h, head)


def test_target_alignment():
    h, head, v = _view()
    raw = head(h).float()
    full = v.full()
    assert torch.allclose(full[:, 1:], raw[:, :-1])  # target t <- position t-1
    labels = torch.randint(0, 11, (2, 7))
    labels[:, 0] = -100
    lp = v.target_logp(labels, chunk_size=3)
    ref = torch.log_softmax(raw[:, :-1], -1).gather(-1, labels[:, 1:, None]).squeeze(-1)
    assert torch.allclose(lp[:, 1:], ref, atol=1e-6) and (lp[:, 0] == 0).all()


def test_chunked_equals_full_and_grad():
    h, head, sv = _view(seed=1)
    _, _, tv = _view(seed=2)
    mask = torch.ones(2, 7)
    mask[:, :2] = 0
    a = chunked_token_divergence(sv, tv, reverse_kl, mask, chunk_size=3)
    b = reverse_kl(sv.full(), tv.full().detach()) * mask
    assert torch.allclose(a, b, atol=1e-5)
    (ga,) = torch.autograd.grad(a.sum(), h)
    (gb,) = torch.autograd.grad(b.sum(), h)
    assert torch.allclose(ga, gb, atol=1e-5)


def test_divergence_values():
    s, t = torch.randn(3, 4, 9), torch.randn(3, 4, 9)
    kl = F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1), reduction="none").sum(-1)
    ent = -(F.softmax(t, -1) * F.log_softmax(t, -1)).sum(-1)
    assert torch.allclose(forward_kl(s, t), kl + ent, atol=1e-5)  # CE form = KL + H(p)
    rkl = (F.softmax(s, -1) * (F.log_softmax(s, -1) - F.log_softmax(t, -1))).sum(-1)
    assert torch.allclose(reverse_kl(s, t), rkl, atol=1e-6)


def test_aggregator_global_norm():
    out = LossOutput({"a": LossTerm(torch.tensor(10.0), "token_sum"), "b": LossTerm(torch.tensor(4.0), "mean")})
    tot, m = WeightedSum().combine([WeightedOutput("x", 2.0, out)], Norms(z_tok=5, z_seq=2, n_micro=2))
    assert torch.isclose(tot, torch.tensor(2.0 * (10 / 5 + 4 / 2)))
    agg = KDMix(KDMix.Config(kd_ratio=0.25, ce_alias="nll"))
    o1 = LossOutput({"n": LossTerm(torch.tensor(1.0), "mean")})
    o2 = LossOutput({"k": LossTerm(torch.tensor(1.0), "mean")})
    tot, _ = agg.combine([WeightedOutput("nll", 1, o1), WeightedOutput("kd", 1, o2)], Norms(1, 1, 1))
    assert torch.isclose(tot, torch.tensor(1.0))
