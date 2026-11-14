"""T08: weight-sync name handling + rollout_weight_sync callback (CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM

import lmkd.plugins as plugins
from lmkd.core.config import recipe_from_dict
from lmkd.core.registry import CALLBACKS
from lmkd.rollout.weight_sync import clean_peft_name, iter_student_weights
from lmkd.train.builder import build_runtime


def test_clean_peft_name():
    assert clean_peft_name("base_model.model.h.0.attn.c_attn.base_layer.weight") == "h.0.attn.c_attn.weight"
    assert clean_peft_name("base_model.model.h.0.attn.c_attn.lora_A.default.weight") is None
    assert clean_peft_name("base_model.model.x.original_module.weight") is None
    assert clean_peft_name("base_model.model.lm_head.modules_to_save.default.weight") == "lm_head.weight"


def test_full_ft_names_match_state_dict(base_recipe):
    rt = build_runtime(recipe_from_dict(base_recipe))
    names = {n for n, _ in iter_student_weights(rt, "full")}
    assert names == {n for n, _ in rt.student_model.named_parameters()}


def test_zero3_uses_engine_stream(base_recipe):

    rt = build_runtime(recipe_from_dict(base_recipe))
    fake = SimpleNamespace(is_partitioned=True, iter_full_state_dict=lambda m: iter([("a.weight", torch.zeros(1))]))
    rt.engine = fake
    assert [n for n, _ in iter_student_weights(rt, "full")] == ["a.weight"]
    rt.engine = SimpleNamespace(is_partitioned=True)
    with pytest.raises(NotImplementedError, match="iter_full_state_dict"):
        list(iter_student_weights(rt, "full"))


def test_peft_merge_names(base_recipe):
    pytest.importorskip("peft")
    base_recipe["models"]["student"]["lora"] = {
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["c_attn"],
        "fan_in_fan_out": True,
    }
    rt = build_runtime(recipe_from_dict(base_recipe))
    m = rt.student_model
    # make the adapter non-trivial so merge actually changes weights
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.1)
    ids = torch.tensor([[3, 4, 5, 6]])
    m.eval()
    with torch.no_grad():
        before = m(input_ids=ids).logits.clone()

    sd_keys = set(AutoModelForCausalLM.from_pretrained(base_recipe["models"]["student"]["path"]).state_dict())
    merged_seen = []
    names = []
    for n, t in iter_student_weights(rt, "merge"):
        names.append(n)
        merged_seen.append(m.base_model.model.transformer.h[0].attn.c_attn.merged)
    assert names and all(merged_seen)
    for n in names:
        assert "lora_" not in n and "base_layer" not in n and not n.startswith("base_model.model.")
    assert set(names) <= sd_keys
    with torch.no_grad():
        after = m(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-6)
    assert not m.base_model.model.transformer.h[0].attn.c_attn.merged


def test_merge_rejects_zero3(base_recipe):
    pytest.importorskip("peft")
    base_recipe["models"]["student"]["lora"] = {"r": 4, "target_modules": ["c_attn"], "fan_in_fan_out": True}
    rt = build_runtime(recipe_from_dict(base_recipe))
    rt.engine.is_partitioned = True
    with pytest.raises(NotImplementedError, match="ZeRO-3"):
        list(iter_student_weights(rt, "merge"))
    assert not rt.student_model.base_model.model.transformer.h[0].attn.c_attn.merged


def test_callback_marks_stale(base_recipe):

    plugins.load_builtin_plugins()
    cb = CALLBACKS.build({"type": "rollout_weight_sync", "every": 2})

    class R:
        stale = 0

        def mark_stale(self):
            self.stale += 1

    class S:
        step = 1

    class RT:
        rollout = R()
        state = S()

    rt = RT()
    cb.on_step_end(rt, None)
    assert rt.rollout.stale == 0
    rt.state.step = 2
    cb.on_step_end(rt, None)
    assert rt.rollout.stale == 1
    rt.rollout = None
    cb.on_step_end(rt, None)  # no rollout: no-op
