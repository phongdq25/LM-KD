"""Reproductions for the 2026-10-26 audit correctness follow-up."""

import pytest
import torch
from tests.e2e_tiny.test_rl_tiny import rl_recipe

import lmkd
from lmkd.core.config import recipe_from_dict
from lmkd.core.validate import check_recipe, preflight
from lmkd.rl.rewards import MathVerify
from lmkd.train.checkpointing import load_checkpoint


@pytest.mark.parametrize("gold", [0, 0.0, "0"])
def test_math_numeric_zero(gold):
    assert MathVerify()(["p"], [r"\boxed{0}"], [{"answer": gold}])[0]["acc"] == 1.0


def test_math_case_sensitive_fallback(monkeypatch):
    def unavailable(*args):
        raise ImportError("unavailable")

    monkeypatch.setattr("lmkd.rl.rewards._math_equivalent", unavailable)
    result = MathVerify()(["p", "p"], [r"\boxed{x}", r"\boxed{A B}"], [{"answer": "X"}, {"answer": "AB"}])
    assert [r["acc"] for r in result] == [0.0, 1.0]


def test_math_verified_mismatch_has_no_fallback(monkeypatch):
    monkeypatch.setattr("lmkd.rl.rewards._math_equivalent", lambda *args: False)
    assert MathVerify()(["p"], [r"\boxed{x}"], [{"answer": "x"}])[0]["acc"] == 0.0


@pytest.mark.parametrize(
    "policy", ["ema", {"type": "periodic_sync", "every": 1}, {"type": "periodic_sync", "every": 0}, "trust_region"]
)
def test_self_teacher_resume(base_recipe, policy):
    base_recipe["teacher"] = {"provider": {"type": "self", "weight_policy": policy}, "input": "same"}
    base_recipe["loss"] = {"terms": [{"criterion": "token_kd"}]}
    rt = lmkd.build(base_recipe)
    rt.begin()
    rt.train_step()
    rt.train_step()
    path = rt.save("resume")
    expected = rt.train_step()
    restored = lmkd.build(base_recipe)
    restored.begin()
    load_checkpoint(restored, path)
    actual = restored.train_step()
    torch.testing.assert_close(actual.metrics["loss/total"], expected.metrics["loss/total"], atol=1e-6, rtol=1e-6)
    for p, q in zip(rt.student_model.parameters(), restored.student_model.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-6)
    assert (path / "teacher" / "rank0.pt").exists()


@pytest.mark.parametrize("successes", [[False] * 4, [False, True, False, True], [True] * 4])
def test_ppo_teacher_event_cadence(base_recipe, monkeypatch, successes):
    rt = lmkd.build(rl_recipe(base_recipe, "sdpo"))
    calls = []
    monkeypatch.setattr(rt.teacher, "on_optimizer_step", lambda state: calls.append(state.step))
    outcomes = iter(successes)
    original_step = rt.engine.step_optimizers
    monkeypatch.setattr(rt.engine, "step_optimizers", lambda opts: original_step(opts) if next(outcomes) else False)
    before = rt.schedulers[0].last_epoch
    rt.begin()
    out = rt.train_step()
    assert out.skipped == (not any(successes))
    assert calls == ([1] if any(successes) else [])
    assert rt.schedulers[0].last_epoch == before + int(any(successes))


@pytest.mark.parametrize(
    "field,value",
    [
        ("advantage", "missing_adv"),
        ("adv_transforms", [{"type": "missing_transform"}]),
        ("adv_transforms", [{"type": "rlsd_reweight", "lam": 2}]),
    ],
)
def test_nested_algorithm_validation(base_recipe, field, value):
    cfg = recipe_from_dict(rl_recipe(base_recipe))
    cfg.algorithm[field] = value
    report = check_recipe(cfg)
    assert not report.ok and any(field in e for e in report.errors)


@pytest.mark.parametrize("part", ["verifier", "weight_policy"])
def test_nested_other_validation(base_recipe, part):
    cfg = recipe_from_dict(rl_recipe(base_recipe, "sdpo"))
    if part == "verifier":
        cfg.producer[part] = "missing_verifier"
    else:
        cfg.teacher.provider[part] = "missing_policy"
    report = check_recipe(cfg)
    assert not report.ok and any(part in e for e in report.errors)


@pytest.mark.parametrize("n,method", [(2, "grpo"), (8, "grpo"), (2, "sdpo")])
def test_rl_preflight_preserves_groups(base_recipe, n, method):
    recipe = rl_recipe(base_recipe, method)
    recipe["producer"]["n"] = n
    recipe["train"]["batch_size"] = max(4, n)
    cfg = recipe_from_dict(recipe)
    before = cfg.model_dump()
    report = preflight(cfg)
    assert report.ok, report.render()
    assert cfg.model_dump() == before


def test_sampled_sdpo_bad_alpha_is_reported(base_recipe):
    recipe = rl_recipe(base_recipe, "sdpo")
    recipe["loss"]["terms"][0]["params"] = {"full_logit": False, "alpha": 0.5}
    report = check_recipe(recipe_from_dict(recipe))
    assert not report.ok and any("alpha" in e for e in report.errors)


@pytest.mark.parametrize("micro", [2, 4])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_pcgrad_accumulates_before_projection(base_recipe, monkeypatch, micro, cancel, scaled):
    from lmkd.losses.aggregator import WeightedOutput
    from lmkd.losses.base import LossOutput, LossTerm
    from lmkd.train.grad import pcgrad_combine

    base_recipe["step"] = {"type": "multi_objective_grad", "combine": "pcgrad"}
    base_recipe["grad"] = {"transforms": [], "clip_norm": None}
    base_recipe["optim"] = {"lr": 0.1, "weight_decay": 0.0, "optimizers": {"default": {"type": "sgd"}}}
    base_recipe["train"]["micro_batch_size"] = micro
    rt = lmkd.build(base_recipe)
    if scaled:
        rt.engine.scaler = torch.amp.GradScaler("cpu")
    p = rt.trainable_params[0].param
    before = p.detach().clone()

    def outputs(batch, runtime):
        x = p.flatten()
        pos = batch["audit.phase"].sum() if cancel else batch.batch_size
        neg = batch.batch_size - pos
        return [
            WeightedOutput("a", 0.5, LossOutput({"x": LossTerm(x[0] * pos + x[1] * neg, "seq_sum")})),
            WeightedOutput("b", 2.0, LossOutput({"x": LossTerm((-x[0] + x[1]) * (pos - neg), "seq_sum")})),
        ]

    monkeypatch.setattr(rt.algorithm, "loss", outputs)
    rt.begin()
    batch = rt.producer.next(rt.state)
    batch["audit.phase"] = torch.tensor([1.0] * 4 + [0.0] * 4)
    out = rt.train_step(batch)
    expected = (
        torch.tensor([0.25, 0.25])
        if cancel
        else pcgrad_combine([[torch.tensor([0.5, 0.0])], [torch.tensor([-2.0, 2.0])]])[0]
    )
    torch.testing.assert_close(p.flatten()[:2], before.flatten()[:2] - 0.1 * expected, atol=1e-7, rtol=1e-6)
    assert not out.skipped and torch.isfinite(out.metrics["loss/total"])


def test_unsupported_transition_probe_fails_early():
    from lmkd.core.registry import CRITERIONS

    with pytest.raises(ValueError, match="grad_probe_every"):
        CRITERIONS.build({"type": "transition_loss", "grad_probe_every": 10})


def test_unsupported_stage_interpreter_fails_early(base_recipe, monkeypatch):
    from lmkd.core.registry import STAGES
    from lmkd.pipeline.runner import plan
    from lmkd.pipeline.stage import Resources

    monkeypatch.setattr(STAGES.get("prepare_dataset"), "resources", Resources(python_env="/unsupported/python"))
    cfg = recipe_from_dict(base_recipe)
    cfg.pipeline = [{"stage": "prepare_dataset", "source": base_recipe["data"]["train"]}]
    with pytest.raises(NotImplementedError, match="python_env"):
        plan(cfg)


def test_listwise_preflight_keeps_complete_group(base_recipe, tmp_path):
    import json

    path = tmp_path / "listwise.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": str(i), "prompt": "w1", "responses": ["w2", "w3", "w4"]}) for i in range(4))
    )
    base_recipe["data"]["train"]["path"] = str(path)
    base_recipe["data"]["transforms"] = [{"type": "tokenize_listwise", "format_plain_prompts": False}]
    base_recipe["producer"] = {"type": "listwise"}
    base_recipe["train"].update(batch_size=6, micro_batch_size=3)
    report = preflight(recipe_from_dict(base_recipe))
    assert report.ok, report.render()
