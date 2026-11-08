"""Missing baseline paths train through the existing native runtime."""

from contextlib import contextmanager

import torch

import lmkd
from lmkd.rollout.base import Rollout


def test_pbsd_online_pair_gate_and_train(base_recipe, monkeypatch):
    base_recipe["data"]["transforms"] = [
        {
            "type": "tokenize_prompt",
            "template": "{prompt}",
            "keep_fields": ["prompt", "solution"],
            "use_chat_template": False,
        }
    ]
    base_recipe["models"]["student"]["lora"] = {"r": 2, "lora_alpha": 2, "target_modules": ["c_attn"]}
    base_recipe["producer"] = {"type": "pbsd_online", "sampling": {"max_new_tokens": 2}}
    base_recipe["rollout"] = {"type": "hf"}
    base_recipe["teacher"] = {
        "provider": {"type": "self", "weight_policy": "lora_disabled"},
        "input": {
            "type": "privileged_prompt",
            "template": "{prompt} {solution}",
            "fields": ["prompt", "solution"],
            "use_chat_template": False,
        },
    }
    base_recipe["loss"] = {"terms": [{"criterion": "pbsd"}]}
    rows = [{"id": str(i), "prompt": "Q: w1", "response": "w2", "solution": "w2"} for i in range(8)]
    rt = lmkd.build(base_recipe, train_records=rows)
    active = [False]

    @contextmanager
    def policy_context(_rt):
        assert not active[0]
        active[0] = True
        try:
            with rt.student_adapter.adapters_disabled(rt.student_model):
                yield rt.student_model
        finally:
            active[0] = False

    monkeypatch.setattr(rt.teacher.policy, "context", policy_context)

    def generate(prompts, spec, lora=None):
        ids = [4, 1] if active[0] else [5, 1]
        return [[Rollout(p, ids, text="correct" if active[0] else "wrong")] for p in prompts]

    monkeypatch.setattr(rt.rollout, "generate", generate)
    monkeypatch.setattr(rt.producer.teacher_rollout, "generate", generate)
    rt.producer.verifier = lambda prompts, texts, rows: [
        {"score": float(t == "correct"), "acc": float(t == "correct"), "feedback": ""} for t in texts
    ]
    batch = rt.producer.next(rt.state)
    assert batch["group.index"].tolist() == [0, 1] * 4
    assert batch["pref.gate"].tolist() == [1.0] * 8
    response_ids = batch["student.input_ids"].gather(1, batch["student.prompt_len"][:, None])
    assert response_ids.flatten().tolist() == [4, 5] * 4
    assert not active[0]
    rt.begin()
    try:
        rt.train_step(batch)
        assert rt.state.step == 1
        assert all(torch.isfinite(p).all() for p in rt.student_model.parameters())
    finally:
        rt.end()
