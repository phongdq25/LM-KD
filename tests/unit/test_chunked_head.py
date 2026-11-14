import pytest
import torch
from torch import nn

from lmkd.losses.functional.divergences import forward_kl, reverse_kl
from lmkd.models import logits_view as lv
from lmkd.models.chunked_head import ChunkedHeadFunction


@pytest.mark.parametrize("kind", ["logp", "fkl", "rkl"])
@pytest.mark.parametrize("bias,temperature,limit", [(False, 1.0, None), (True, 2.0, 29)])
def test_efficient_matches_naive(kind, bias, temperature, limit, monkeypatch):
    torch.manual_seed(3)
    h = torch.randn(2, 13, 8, requires_grad=True)
    head = nn.Linear(8, 31, bias=bias)
    teacher = lv.LogitsView(torch.randn_like(h), nn.Linear(8, 33))
    labels = torch.randint(0, 29, (2, 13))
    labels[:, :4] = -100
    labels[0] = -100
    mask = (labels != -100).float()
    view = lv.LogitsView(h, head, temperature=temperature, vocab_limit=limit)

    def run():
        if kind == "logp":
            return view.target_logp(labels, 4)
        return lv.chunked_token_divergence(view, teacher, forward_kl if kind == "fkl" else reverse_kl, mask, 4)

    monkeypatch.setattr(lv, "LOGITS_VIEW_EFFICIENT", False)
    ref = run()
    params = (h, head.weight) + ((head.bias,) if bias else ())
    grads = torch.autograd.grad(ref.sum(), params)
    monkeypatch.setattr(lv, "LOGITS_VIEW_EFFICIENT", True)
    actual = run()
    assert torch.allclose(actual, ref, atol=1e-6)
    for a, b in zip(torch.autograd.grad(actual.sum(), params), grads):
        assert torch.allclose(a, b, atol=1e-4)


def test_skips_masked_chunks():
    h = torch.randn(2, 12, 8, requires_grad=True)
    w = torch.randn(31, 8, requires_grad=True)
    mask = torch.zeros(2, 12)
    mask[:, 8:] = 1
    calls = []

    def fn(x, s, e):
        calls.append((s, e))
        return x.log_softmax(-1)[..., 0]

    ChunkedHeadFunction.apply(h, w, None, fn, 4, 1.0, None, mask).sum().backward()
    assert calls == [(8, 12), (8, 12)]


def test_no_grad_path_unchanged(monkeypatch):
    def fail(*args):
        raise AssertionError("efficient path under no_grad")

    monkeypatch.setattr(ChunkedHeadFunction, "apply", fail)
    with torch.no_grad():
        v = lv.LogitsView(torch.randn(2, 3, 8, requires_grad=True), nn.Linear(8, 31))
        assert v.target_logp(torch.zeros(2, 3, dtype=torch.long)).shape == (2, 3)


def test_tied_weights(monkeypatch):
    emb = nn.Embedding(31, 8)
    head = nn.Linear(8, 31, bias=False)
    head.weight = emb.weight
    ids = torch.randint(0, 31, (2, 13))

    def run(flag):
        monkeypatch.setattr(lv, "LOGITS_VIEW_EFFICIENT", flag)
        return lv.LogitsView(emb(ids), head).target_logp(ids, 4)

    ref = run(False)
    grad = torch.autograd.grad(ref.sum(), emb.weight)[0]
    actual = run(True)
    assert torch.allclose(actual, ref, atol=1e-6)
    assert torch.allclose(torch.autograd.grad(actual.sum(), emb.weight)[0], grad, atol=1e-4)


@pytest.mark.parametrize("mask_all", [False, True])
def test_zero_target_and_zero_mask(mask_all, monkeypatch):
    h = torch.randn(2, 5, 8, requires_grad=True)
    head = nn.Linear(8, 31)
    labels = torch.randint(0, 31, (2, 5))
    if mask_all:
        labels[:] = -100
    view = lv.LogitsView(h, head)
    monkeypatch.setattr(lv, "LOGITS_VIEW_EFFICIENT", False)
    ref = view.target_logp(labels, 2)
    grads = torch.autograd.grad(ref.sum(), (h, head.weight, head.bias))
    monkeypatch.setattr(lv, "LOGITS_VIEW_EFFICIENT", True)
    actual = view.target_logp(labels, 2)
    torch.testing.assert_close(actual, ref, atol=1e-6, rtol=0.0)
    for a, b in zip(torch.autograd.grad(actual.sum(), (h, head.weight, head.bias)), grads):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=0.0)


def test_no_logits_saved_for_backward():
    h = torch.randn(2, 13, 8, requires_grad=True)
    head = nn.Linear(8, 31)
    shapes = []

    def pack(t):
        shapes.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = lv.LogitsView(h, head).target_logp(torch.zeros(2, 13, dtype=torch.long), 4)
    assert not any(len(shape) == 3 and shape[-1] == 31 for shape in shapes)
    loss.sum().backward()


def test_sharded_head_fallback(monkeypatch, caplog):
    head = nn.Linear(8, 31)
    head.weight.ds_id = 1
    lv._WARNED_HEADS.clear()

    def fail(*args):
        raise AssertionError("sharded head used custom Function")

    monkeypatch.setattr(ChunkedHeadFunction, "apply", fail)
    view = lv.LogitsView(torch.randn(2, 5, 8, requires_grad=True), head)
    for _ in range(2):
        view.target_logp(torch.zeros(2, 5, dtype=torch.long), 2).sum().backward()
    assert sum("fallback" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("kind", ["logp", "fkl", "rkl"])
@pytest.mark.parametrize("state", ["weight_ds_id", "bias_ds_id", "empty_weight", "wrapped_head"])
def test_fallback_preserves_module_forward_and_gradients(kind, state, monkeypatch, caplog):
    """Simulate parameter availability; this does not initialize a DeepSpeed engine."""
    torch.manual_seed(42)
    h = torch.randn(2, 7, 8, requires_grad=True)
    linear = nn.Linear(8, 31)
    head = nn.Sequential(linear) if state == "wrapped_head" else linear
    view = lv.LogitsView(h, head, temperature=2.0, vocab_limit=29)
    teacher = lv.LogitsView(torch.randn_like(h), nn.Linear(8, 31))
    labels = torch.randint(0, 29, (2, 7))
    labels[:, :2] = -100

    def run():
        if kind == "logp":
            return view.target_logp(labels, 3)
        fn = forward_kl if kind == "fkl" else reverse_kl
        return lv.chunked_token_divergence(view, teacher, fn, (labels != -100).float(), 3)

    monkeypatch.setattr(lv, "LOGITS_VIEW_EFFICIENT", False)
    ref = run()
    params = (h, linear.weight, linear.bias)
    ref_grads = torch.autograd.grad(ref.sum(), params)
    if state == "weight_ds_id":
        linear.weight.ds_id = 0
    elif state == "bias_ds_id":
        linear.bias.ds_id = 0
    elif state == "empty_weight":
        full_weight = linear.weight.detach().clone()
        linear.weight.data = torch.empty(0)
    calls = []

    def before_forward(module, args):
        calls.append(1)
        if state == "empty_weight":
            # The engine owns parameter availability on the module fallback path.
            module.weight.data = full_weight.clone()

    handle = linear.register_forward_pre_hook(before_forward)

    def forbidden(*args):
        raise AssertionError("fallback must not invoke custom autograd")

    monkeypatch.setattr(ChunkedHeadFunction, "apply", forbidden)
    monkeypatch.setattr(lv, "LOGITS_VIEW_EFFICIENT", True)
    monkeypatch.setattr(lv, "_WARNED_HEADS", set())
    try:
        actual = run()
        torch.testing.assert_close(actual, ref, atol=1e-6, rtol=0.0)
        for a, b in zip(torch.autograd.grad(actual.sum(), params), ref_grads):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=0.0)
    finally:
        handle.remove()
    assert calls
    assert sum("fallback" in record.message for record in caplog.records) == 1
