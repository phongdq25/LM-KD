"""T25: rollout grouping, teacher re-prompt selection and PPO step accounting."""

from types import SimpleNamespace

import pytest
import torch
from tests.e2e_tiny.test_rl_tiny import ToyReward, rl_recipe

import lmkd
from lmkd.core.registry import REWARDS
from lmkd.rollout.base import Rollout
from lmkd.teachers.signals import TeacherSignalSpec


def build_rl(base_recipe, method="grpo"):
    REWARDS.register("toy_rl", force=True)(ToyReward)
    return lmkd.build(rl_recipe(base_recipe, method))


def test_rollout_group_shapes_uid(base_recipe):
    rt = build_rl(base_recipe)
    rt.rollout = SimpleNamespace(
        generate=lambda prompts, spec: [
            [Rollout(p, [3, 1], [-1.0, -1.0], text="a"), Rollout(p, [4, 1], [-2.0, -2.0], text="bb")] for p in prompts
        ]
    )
    batch = rt.producer.next(rt.state)
    assert batch.batch_size == 4
    assert batch.meta["group.uid"][0] == batch.meta["group.uid"][1]
    assert batch.meta["group.uid"][0] != batch.meta["group.uid"][2]
    assert batch["group.index"].tolist() == [0, 1, 0, 1]
    assert batch["rollout.reward"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert "rl/reward_mean" in rt.state.extra["metrics"]
    saved = rt.producer.state_dict()
    first = rt.producer.next(rt.state)
    rt.producer.load_state_dict(saved)
    assert rt.producer.next(rt.state).meta["group.uid"] == first.meta["group.uid"]


def test_sdpo_reprompt_selection(base_recipe):
    rt = build_rl(base_recipe, "sdpo")
    rows = [
        {
            "student.input_ids": [3, 4, 1],
            "student.attention_mask": [1, 1, 1],
            "student.labels": [-100, 4, 1],
            "student.prompt_len": 1,
            "group.uid": uid,
            "group.index": i % 3,
            "meta.prompt_text": "w0",
            "meta.response_text": text,
            "rollout.reward": reward,
            "meta.feedback": feedback,
        }
        for i, (uid, text, reward, feedback) in enumerate(
            [
                ("a", "w1", 1.0, ""),
                ("a", "w2", 0.0, ""),
                ("a", "w3", 1.0, ""),
                ("b", "w4", 0.0, ""),
                ("c", "w5", 0.0, "w6"),
            ]
        )
    ]
    batch = rt.teacher_input.build(rt.collator(rows), rt)
    prompts = [
        rt.student_tokenizer.decode(batch["teacher.input_ids"][i, : int(length)])
        for i, length in enumerate(batch["teacher.prompt_len"])
    ]
    assert prompts[:3] == ["w0 w3", "w0 w1", "w0 w1"]
    assert batch["rollout.has_signal"].tolist() == [1.0, 1.0, 1.0, 0.0, 1.0]
    assert batch["seq.loss_mask"][3].sum() == 0
    for i, length in enumerate(batch["teacher.prompt_len"]):
        assert batch["teacher.input_ids"][i, int(length) : int(length) + 2].tolist() == [4, 1]


def test_ppo_epochs_scheduler_once(base_recipe):
    rt = build_rl(base_recipe)
    calls = []
    handles = [opt.register_step_post_hook(lambda *args: calls.append(1)) for opt in rt.optimizers.values()]
    before = [s.last_epoch for s in rt.schedulers]
    rt.begin()
    try:
        rt.train_step()
        assert len(calls) == 4 * len(rt.optimizers)
        assert [s.last_epoch for s in rt.schedulers] == [i + 1 for i in before]
    finally:
        for handle in handles:
            handle.remove()
        rt.end()


@pytest.mark.parametrize("mix", [0.0, 1.0, 0.5])
def test_trust_region_lerp(base_recipe, mix):
    base_recipe["teacher"] = {
        "provider": {"type": "self", "weight_policy": {"type": "trust_region", "mix": mix}},
        "input": "same",
    }
    rt = lmkd.build(base_recipe)
    batch = rt.producer.next(rt.state)
    ref = rt.teacher.policy.ref
    with torch.no_grad():
        for p in rt.student_model.parameters():
            p.add_(0.02 * torch.randn_like(p))
        rt.student_model.eval()
        args = {"input_ids": batch["student.input_ids"], "attention_mask": batch["student.attention_mask"]}
        from lmkd.models.logits_view import LogitsView

        expected = LogitsView(logits=torch.lerp(ref(**args).logits, rt.student_model(**args).logits, mix))
        expected_logp = expected.target_logp(batch["student.labels"])
    result = rt.teacher.signals(batch, TeacherSignalSpec(kind="sampled_logp"), None, None)
    torch.testing.assert_close(result.token_logp, expected_logp, atol=1e-5, rtol=1e-5)


def test_ppo_epochs_overflow_keeps_scheduler(base_recipe, monkeypatch):
    rt = build_rl(base_recipe)
    monkeypatch.setattr(rt.engine, "step_optimizers", lambda opts: False)
    before = [s.last_epoch for s in rt.schedulers]
    rt.begin()
    try:
        result = rt.train_step()
        assert result.skipped
        assert [s.last_epoch for s in rt.schedulers] == before
        assert all(p.param.grad is None for p in rt.trainable_params)
    finally:
        rt.end()


def test_rl_old_logp_frozen_and_is_weight(base_recipe):
    base_recipe = rl_recipe(base_recipe)
    base_recipe["algorithm"]["rollout_is"] = "token"
    rt = lmkd.build(base_recipe)
    batch = rt.algorithm.augment(rt.producer.next(rt.state), rt)
    old = batch["rollout.old_logp"].clone()
    labels = batch["student.labels"]
    mask = labels.ne(-100)
    expected = (old - batch["rollout.rollout_logp"]).clamp(-20, 20).exp().clamp(max=2.0) * mask
    torch.testing.assert_close(batch["rollout.is_weight"], expected)
    assert not old.requires_grad and not batch["rollout.advantage"].requires_grad
    rt.step_strategy.run(batch, rt)
    torch.testing.assert_close(batch["rollout.old_logp"], old, atol=0, rtol=0)


def test_rl_on_policy_shortcut(base_recipe):
    base_recipe = rl_recipe(base_recipe)
    base_recipe["algorithm"]["recompute_old_logp"] = False
    base_recipe["step"].update(epochs=1, mini_batches=1)
    rt = lmkd.build(base_recipe)
    rt.begin()
    try:
        assert torch.isfinite(rt.train_step().metrics["loss/total"])
    finally:
        rt.end()


def test_trust_region_lora_hidden_path(base_recipe):
    base_recipe["models"]["student"]["lora"] = {"r": 2, "lora_alpha": 4, "target_modules": ["c_attn"]}
    base_recipe["teacher"] = {
        "provider": {"type": "self", "weight_policy": {"type": "trust_region", "mix": 0.5}},
        "input": "same",
    }
    rt = lmkd.build(base_recipe)
    rt.begin()
    try:
        rt.train_step()
        batch = rt.producer.next(rt.state)
        rt.student_model.eval()
        args = {"input_ids": batch["student.input_ids"], "attention_mask": batch["student.attention_mask"]}
        with torch.no_grad():
            with rt.student_adapter.adapters_disabled(rt.student_model):
                ref = rt.student_model(**args).logits
            live = rt.student_model(**args).logits
            from lmkd.models.logits_view import LogitsView

            expected = LogitsView(logits=torch.lerp(ref, live, 0.5)).target_logp(batch["student.labels"])
        actual = rt.teacher.signals(batch, TeacherSignalSpec(kind="sampled_logp"), None, None)
        assert rt.teacher.policy.ref is None and not rt.teacher.policy.warned_full
        torch.testing.assert_close(actual.token_logp, expected, atol=1e-5, rtol=1e-5)
    finally:
        rt.end()


def test_ema_teacher_keeps_ema_head_for_lazy_signals(base_recipe):
    import copy

    from lmkd.models.logits_view import LogitsView

    base_recipe["teacher"] = {"provider": {"type": "self", "weight_policy": "ema"}, "input": "same"}
    rt = lmkd.build(base_recipe)
    reference = copy.deepcopy(rt.student_model).eval()
    rt.teacher.policy.on_optimizer_step(rt)
    batch = rt.producer.next(rt.state)
    with torch.no_grad():
        for p in rt.student_model.parameters():
            p.add_(torch.randn_like(p) * 0.03)
        expected = LogitsView(
            logits=reference(
                input_ids=batch["student.input_ids"], attention_mask=batch["student.attention_mask"]
            ).logits
        )
        expected = expected.target_logp(batch["student.labels"])
    actual = rt.teacher.signals(batch, TeacherSignalSpec(kind="sampled_logp"), None, None)
    torch.testing.assert_close(actual.token_logp, expected, atol=1e-5, rtol=1e-5)
