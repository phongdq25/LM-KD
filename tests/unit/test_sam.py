"""T24 SAM/ASAM step strategy tests against pytorch_optimizer.SAM."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from lmkd.core.config import recipe_from_dict
from lmkd.train.builder import build_runtime
from lmkd.train.step_strategies import SAMStep


@pytest.fixture
def sam_recipe(base_recipe, tiny_models, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(tiny_models["student"])
    model_path = tmp_path / "no_dropout_gpt2"
    model_path.mkdir()
    torch.manual_seed(17)
    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    GPT2LMHeadModel(cfg).save_pretrained(model_path)
    base_recipe["models"]["student"].update(path=str(model_path), tokenizer=tiny_models["student"])
    base_recipe["train"].update(batch_size=8, micro_batch_size=8, max_steps=4)
    base_recipe["optim"] = {
        "lr": 0.1,
        "weight_decay": 0.0,
        "optimizers": {"default": {"type": "sgd"}},
        "schedulers": [{"type": "constant"}],
    }
    base_recipe["grad"] = {"clip_norm": None}
    base_recipe["step"] = {"type": "asam", "rho": 0.05, "clip_before_perturb": False}
    return base_recipe


def _ref_loss(rt, batch):
    norms = rt.step_strategy.global_norms([batch])
    outs = rt.algorithm.loss(batch, rt)
    return rt.aggregator.combine(outs, norms)[0]


@pytest.mark.parametrize("strategy,adaptive", [("asam", True), ("sam", False)])
def test_sam_matches_reference_3_steps(sam_recipe, strategy, adaptive):
    pytorch_optimizer = pytest.importorskip("pytorch_optimizer")
    sam_recipe["step"]["type"] = strategy
    actual = build_runtime(recipe_from_dict(sam_recipe))
    reference = build_runtime(recipe_from_dict(sam_recipe))
    batch = actual.producer.next(actual.state)
    ref_opt = pytorch_optimizer.SAM(
        reference.student_model.parameters(),
        torch.optim.SGD,
        rho=0.05,
        adaptive=adaptive,
        lr=0.1,
    )
    for _ in range(3):
        actual.step_strategy.run(batch, actual)
        ref_opt.zero_grad(set_to_none=True)
        _ref_loss(reference, batch).backward()
        ref_opt.first_step(zero_grad=True)
        _ref_loss(reference, batch).backward()
        ref_opt.second_step(zero_grad=True)
    for (name, got), (ref_name, expected) in zip(
        actual.student_model.named_parameters(), reference.student_model.named_parameters()
    ):
        assert name == ref_name
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=0)


def test_sam_grad_accum_invariance(sam_recipe):
    sam_recipe["train"]["micro_batch_size"] = 8
    full = build_runtime(recipe_from_dict(sam_recipe))
    sam_recipe["train"]["micro_batch_size"] = 2
    split = build_runtime(recipe_from_dict(sam_recipe))
    batch = full.producer.next(full.state)
    for _ in range(2):
        full.step_strategy.run(batch, full)
        split.step_strategy.run(batch, split)
    for got, expected in zip(full.student_model.parameters(), split.student_model.parameters()):
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)


def test_sam_restores_on_exception(sam_recipe, monkeypatch):
    rt = build_runtime(recipe_from_dict(sam_recipe))
    before = [p.detach().clone() for p in rt.student_model.parameters()]
    original = rt.step_strategy.accumulate
    calls = 0

    def fail_second(micros, norms):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("second pass failed")
        return original(micros, norms)

    monkeypatch.setattr(rt.step_strategy, "accumulate", fail_second)
    with pytest.raises(ValueError, match="second pass failed"):
        rt.step_strategy.run(rt.producer.next(rt.state), rt)
    for original_param, param in zip(before, rt.student_model.parameters()):
        # add_(delta); sub_(delta) can differ by one fp32 rounding unit.
        torch.testing.assert_close(param, original_param, rtol=0, atol=1e-6)


def test_scheduler_steps_once(sam_recipe):
    sam_recipe["optim"]["schedulers"] = [{"type": "linear"}]
    rt = build_runtime(recipe_from_dict(sam_recipe))
    batch = rt.producer.next(rt.state)
    rt.step_strategy.run(batch, rt)
    rt.step_strategy.run(batch, rt)
    assert rt.schedulers[0].last_epoch == 2
    assert rt.state.micro_step == 4


def test_sam_rejects_fp16(sam_recipe):
    sam_recipe["engine"]["mixed_precision"] = "fp16"
    with pytest.raises(NotImplementedError, match="fp16.*bf16"):
        build_runtime(recipe_from_dict(sam_recipe))


def test_sam_rejects_deepspeed():
    strategy = SAMStep(SAMStep.Config())
    engine = type(
        "FakeEngine",
        (),
        {"cfg": type("Cfg", (), {"backend": "deepspeed", "mixed_precision": "bf16"})(), "scaler": None},
    )()
    with pytest.raises(NotImplementedError, match="DeepSpeed"):
        strategy.setup(type("FakeRuntime", (), {"engine": engine})())
