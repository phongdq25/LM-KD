from __future__ import annotations

import pytest
import torch

from lmkd.losses.cross_tokenizer._common import (
    get_vocab_map,
    pad_ragged_2d,
    pairs_to_index,
    target_hidden,
    target_logits_full,
    vd_divergence,
)
from lmkd.models.logits_view import LogitsView

NAMES = ["forward_kl", "reverse_kl", "adaptive_kl", "skewed_forward_kl", "skewed_reverse_kl", "js_divergence"]


def test_target_hidden():
    h = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    out = target_hidden(h)
    assert torch.equal(out[:, 0], torch.zeros(2, 3))
    assert torch.equal(out[:, 1:], h[:, :-1])
    assert out is not h and h[0, 0].tolist() == [0.0, 1.0, 2.0]  # input untouched (C11)


def test_target_logits_full_is_view_full():
    logits = torch.randn(2, 5, 7)
    view = LogitsView(logits=logits)
    assert torch.equal(target_logits_full(view), view.full())


@pytest.mark.parity
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("use_tea_temp", [False, True])
def test_vd_divergence_parity(name, use_tea_temp):
    from tests.parity._ref.various_divergence_ref import VariousDivergenceRef

    torch.manual_seed(0)
    logits = torch.randn(3, 6, 11)
    t_logits = torch.randn(3, 6, 11)
    target = torch.randint(0, 11, (3, 6))
    target[:, 0] = -100
    ref = VariousDivergenceRef(kd_temp=2.0, tea_temp=1.5, skew_lambda=0.2, adaptive_kl_alpha=0.4)
    method = {
        "forward_kl": "compute_forward_kl_divergence",
        "reverse_kl": "compute_reverse_kl_divergence",
        "adaptive_kl": "compute_adaptive_kl_divergence",
        "skewed_forward_kl": "compute_skewed_forward_kl_divergence",
        "skewed_reverse_kl": "compute_skewed_reverse_kl_divergence",
        "js_divergence": "compute_js_divergence",
    }[name]
    fn = getattr(ref, method)
    expected = fn(logits.clone(), t_logits.clone(), target, reduction="none", use_tea_temp=use_tea_temp)
    got = vd_divergence(
        name,
        logits,
        t_logits,
        kd_temp=2.0,
        tea_temp=1.5,
        use_tea_temp=use_tea_temp,
        skew_lambda=0.2,
        adaptive_alpha=0.4,
    )
    assert got.shape == (3, 6)
    assert torch.allclose(got, expected, atol=1e-5)
    # the reduction='sum' path of the original masks padding the same way as our caller would
    summed = fn(logits.clone(), t_logits.clone(), target, reduction="sum", use_tea_temp=use_tea_temp)
    assert torch.allclose((got * (target != -100)).sum(), summed, atol=1e-4)


def test_vd_divergence_masks_inf_on_both_sides():
    logits = torch.randn(1, 2, 5)
    t_logits = torch.randn(1, 2, 5)
    logits[0, 0, 1] = float("-inf")
    out = vd_divergence("forward_kl", logits, t_logits)
    assert torch.isfinite(out).all()


def test_vd_divergence_unknown():
    with pytest.raises(ValueError, match="unknown divergence"):
        vd_divergence("nope", torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))


def test_pad_ragged_2d():
    items = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[7, 8]])]
    out = pad_ragged_2d(items)
    assert out.shape == (2, 2, 3)
    assert out[1].tolist() == [[7, 8, -1], [-1, -1, -1]]
    assert pad_ragged_2d(items, pad=0)[1, 0].tolist() == [7, 8, 0]


def test_pairs_to_index():
    items = [torch.tensor([[1, 2], [3, 4]]), torch.zeros(0, 2, dtype=torch.long), torch.tensor([[5, 6]])]
    b, c0, c1 = pairs_to_index(items, "cpu")
    assert b.tolist() == [0, 0, 2] and c0.tolist() == [1, 3, 5] and c1.tolist() == [2, 4, 6]
    b, c0, c1 = pairs_to_index([torch.zeros(0, dtype=torch.long)], "cpu")  # empty ragged -> reshape(-1, 2)
    assert b.numel() == c0.numel() == c1.numel() == 0


def test_get_vocab_map_error_and_ok():
    from types import SimpleNamespace

    ctx = SimpleNamespace(rt=SimpleNamespace(data_ctx=SimpleNamespace(vocab_map=None)))
    with pytest.raises(RuntimeError, match="load_vocab_map"):
        get_vocab_map(ctx)
    ctx.rt.data_ctx.vocab_map = "vm"
    assert get_vocab_map(ctx) == "vm"
