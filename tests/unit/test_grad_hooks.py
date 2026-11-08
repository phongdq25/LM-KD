"""T24 activation hooks and PCGrad on CPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lmkd.core.config import recipe_from_dict
from lmkd.train.builder import build_runtime
from lmkd.train.grad import (
    GradProbe,
    HookContext,
    ParamWithGroup,
    PCGrad,
    ProjectOutTopK,
    StepContext,
    pcgrad_combine,
)


def _backward_once(rt, batch):
    norms = rt.step_strategy.global_norms([batch])
    outs = rt.algorithm.loss(batch, rt)
    loss, _ = rt.aggregator.combine(outs, norms)
    loss.backward()


def test_probe_logs_norm_equals_autograd(base_recipe):
    base_recipe["train"].update(batch_size=4, micro_batch_size=4)
    base_recipe["grad"] = {
        "clip_norm": None,
        "hooks": [{"type": "probe", "target": "student.hidden[1]", "name": "h1"}],
    }
    rt = build_runtime(recipe_from_dict(base_recipe))
    batch = rt.producer.next(rt.state)
    captured = {}

    def capture(_module, _args, output):
        captured["h"] = output[0]

    handle = rt.student_model.transformer.h[0].register_forward_hook(capture)
    try:
        norms = rt.step_strategy.global_norms([batch])
        outs = rt.algorithm.loss(batch, rt)
        loss, _ = rt.aggregator.combine(outs, norms)
    finally:
        handle.remove()
    expected = torch.autograd.grad(loss, captured["h"], retain_graph=True)[0]
    # autograd.grad also fires tensor hooks; measure only the actual backward.
    rt.state.extra["metrics"] = {}
    loss.backward()
    metrics = rt.state.extra["metrics"]
    assert metrics["gradprobe/h1/norm"] == pytest.approx(float(expected.norm()), rel=1e-4)
    assert metrics["gradprobe/h1/absmean"] == pytest.approx(float(expected.abs().mean()), rel=1e-4)
    assert metrics["gradprobe/h1/absmax"] == pytest.approx(float(expected.abs().max()), rel=1e-4)


def test_hook_with_gradient_checkpointing(base_recipe, monkeypatch):
    base_recipe["models"]["student"]["gradient_checkpointing"] = True
    base_recipe["train"].update(batch_size=4, micro_batch_size=2)
    base_recipe["grad"] = {
        "clip_norm": None,
        "hooks": [{"type": "probe", "target": "student.hidden[1]"}],
    }
    rt = build_runtime(recipe_from_dict(base_recipe))
    calls = []
    original = GradProbe.__call__

    def counted(self, grad, ctx):
        calls.append(grad.shape)
        return original(self, grad, ctx)

    monkeypatch.setattr(GradProbe, "__call__", counted)
    rt.step_strategy.run(rt.producer.next(rt.state), rt)
    assert len(calls) == 2  # once for each micro-batch, including with checkpoint recomputation


def _hook_context(mask, n_micro=1):
    rt = SimpleNamespace(
        config=SimpleNamespace(train=SimpleNamespace(batch_size=n_micro, micro_batch_size=1)),
        engine=SimpleNamespace(world_size=1),
    )
    state = SimpleNamespace(extra={})
    return HookContext(rt=rt, batch={"student.attention_mask": mask}, state=state)


@pytest.mark.parametrize("mask_padding", [False, True])
def test_project_out_removes_subspace(mask_padding):
    torch.manual_seed(3)
    grad = torch.randn(2, 5, 8, dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    ctx = _hook_context(mask)
    hook = ProjectOutTopK(
        ProjectOutTopK.Config(
            target="student.hidden[1]",
            k=3,
            method="full",
            mask_padding=mask_padding,
        )
    )
    projected = hook(grad, ctx)
    valid = mask.reshape(-1).bool() if mask_padding else torch.ones(10, dtype=torch.bool)
    _, _, vh = torch.linalg.svd(grad.reshape(-1, 8)[valid], full_matrices=False)
    p = vh[:3]
    assert projected.shape == grad.shape and projected.dtype == grad.dtype
    assert (projected.reshape(-1, 8)[valid] @ p.T).abs().max() < 1e-5
    if mask_padding:
        torch.testing.assert_close(projected.reshape(-1, 8)[~valid], grad.reshape(-1, 8)[~valid])
    assert 0 < ctx.state.extra["metrics"]["gradhook/student.hidden[1]/removed_frac"] <= 1


def test_project_out_changes_param_grad(base_recipe):
    base_recipe["train"].update(batch_size=4, micro_batch_size=4)
    base_recipe["grad"] = {"clip_norm": None}
    plain = build_runtime(recipe_from_dict(base_recipe))
    base_recipe["grad"]["hooks"] = [
        {
            "type": "project_out_topk_svd",
            "target": "student.hidden[1]",
            "k": 3,
            "method": "full",
        }
    ]
    projected = build_runtime(recipe_from_dict(base_recipe))
    plain.student_model.eval()
    projected.student_model.eval()  # remove dropout so both gradients use the same computation
    batch = plain.producer.next(plain.state)
    _backward_once(plain, batch)
    _backward_once(projected, batch)
    h0_plain = plain.student_model.transformer.h[0].mlp.c_fc.weight.grad
    h0_projected = projected.student_model.transformer.h[0].mlp.c_fc.weight.grad
    h1_plain = plain.student_model.transformer.h[1].mlp.c_fc.weight.grad
    h1_projected = projected.student_model.transformer.h[1].mlp.c_fc.weight.grad
    assert not torch.allclose(h0_plain, h0_projected, atol=1e-6)
    torch.testing.assert_close(h1_plain, h1_projected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_pcgrad_combine_conflict(seed):
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-1.0, 1.0])
    out = pcgrad_combine([[g1], [g2]], seed=seed)
    torch.testing.assert_close(out[0], torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(g1, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(g2, torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(pcgrad_combine([[g1], [torch.tensor([1.0, 1.0])]])[0], torch.tensor([2.0, 1.0]))


def test_pcgrad_transform_requires_term_grads():
    p = torch.nn.Parameter(torch.zeros(2))
    wrapped = [ParamWithGroup("p", p, "default", "default")]
    transform = PCGrad(PCGrad.Config(terms=["a", "b"]))
    with pytest.raises(RuntimeError, match=r"pcgrad needs step: multi_objective_grad"):
        transform(wrapped, StepContext(rt=None, metrics={}))
    ctx = StepContext(
        rt=None,
        metrics={},
        term_grads={
            "a": [torch.tensor([1.0, 0.0])],
            "b": [torch.tensor([-1.0, 1.0])],
        },
    )
    transform(wrapped, ctx)
    torch.testing.assert_close(p.grad, torch.tensor([0.5, 1.5]))


def test_input_embeds_frozen_warns_once():
    embedding = torch.nn.Embedding(4, 3)
    embedding.weight.requires_grad_(False)
    rt = SimpleNamespace(
        student_model=object(),
        student_adapter=SimpleNamespace(embed_tokens=lambda _model: embedding),
    )
    hook = GradProbe(GradProbe.Config(target="student.input_embeds"))
    with pytest.warns(RuntimeWarning, match="input_embeds") as warnings_seen:
        hook.setup(rt)
    assert len(warnings_seen) == 1
