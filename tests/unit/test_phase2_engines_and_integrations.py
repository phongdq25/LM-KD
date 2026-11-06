"""Tests for Phase 2: FSDP2 engine and DeepSpeed MultiOptimizer."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lmkd.core.registry import ENGINES
from lmkd.train.engine import FSDP2Engine, MultiOptimizer


def test_fsdp2_engine_registered_and_build():
    assert "fsdp2" in ENGINES
    eng = ENGINES.build({"type": "fsdp2", "mixed_precision": "no"})
    assert isinstance(eng, FSDP2Engine)
    assert eng.world_size >= 1
    assert eng.device.type in ("cuda", "cpu")


def test_fsdp2_engine_prepare_and_all_reduce():
    eng = FSDP2Engine(FSDP2Engine.Config(mixed_precision="no"))
    model = nn.Linear(4, 4)
    aux = nn.Module()
    opt = torch.optim.Adam(model.parameters())
    wrapped_model, wrapped_aux, opts = eng.prepare(model, aux, {"default": opt})
    assert wrapped_model is not None
    assert "default" in opts
    assert {id(p) for g in opt.param_groups for p in g["params"]} == {
        id(p) for p in wrapped_model.parameters() if p.requires_grad
    }

    t = torch.tensor([1.0, 2.0])
    reduced = eng.all_reduce_sum(t)
    assert torch.allclose(reduced, t)


def test_multi_optimizer_for_deepspeed():
    p1 = nn.Parameter(torch.randn(4, 4))
    p2 = nn.Parameter(torch.randn(4))
    opt1 = torch.optim.SGD([p1], lr=0.1)
    opt2 = torch.optim.AdamW([p2], lr=0.01)

    multi = MultiOptimizer({"sgd": opt1, "adam": opt2})
    assert len(multi.param_groups) == 2
    assert multi.defaults is not None

    p1.grad = torch.ones_like(p1)
    p2.grad = torch.ones_like(p2)
    multi.step()
    multi.zero_grad()
    assert p1.grad is None
    assert p2.grad is None


def test_fsdp2_rejects_adapter_only_checkpoint_lifecycle():
    eng = FSDP2Engine(FSDP2Engine.Config(mixed_precision="no"))
    rt = SimpleNamespace(student_model=SimpleNamespace(peft_config={}))
    with pytest.raises(ValueError, match="PEFT"):
        eng.setup(rt)
