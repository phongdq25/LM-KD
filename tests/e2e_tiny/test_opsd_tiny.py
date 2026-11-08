"""E2E tiny tests for on-policy self-distillation (OPSD) with tinker_rkl and gjsd (T10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lmkd.core.config import recipe_from_dict
from lmkd.core.validate import check_recipe
from lmkd.losses.functional.divergences import DIVERGENCES
from lmkd.train.trainer import train


@pytest.fixture
def opsd_toy_jsonl(tmp_path):
    p = tmp_path / "opsd_toy.jsonl"
    rows = []
    for i in range(16):
        rows.append(
            {
                "id": f"math_{i}",
                "problem": f"What is w{i % 4} plus w{(i + 1) % 4}?",
                "solution": f"w{i % 4} + w{(i + 1) % 4} equals w{(i * 2) % 4}.",
            }
        )
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_opsd_tinker_tiny(base_recipe, tiny_models, opsd_toy_jsonl):
    """OPSD with tinker_rkl loss, self teacher, privileged_prompt, mixed producer runs 3 steps."""
    base_recipe["data"]["train"] = {"type": "jsonl", "path": opsd_toy_jsonl}
    base_recipe["data"]["transforms"] = [
        {
            "type": "tokenize_prompt",
            "template": "Problem: {problem}",
            "use_chat_template": False,
            "keep_fields": ["problem", "solution"],
        }
    ]
    base_recipe["rollout"] = {"type": "hf"}
    base_recipe["producer"] = {
        "type": "mixed",
        "lmbda": 1.0,
        "sampling": {"max_new_tokens": 4, "temperature": 0.0},
    }
    base_recipe["teacher"] = {
        "provider": {"type": "self", "weight_policy": "live", "position_map": "response_shift"},
        "input": {
            "type": "privileged_prompt",
            "template": "Problem: {problem}\nSolution: {solution}",
            "fields": ["problem", "solution"],
            "use_chat_template": False,
        },
    }
    base_recipe["loss"] = {
        "terms": [{"criterion": "tinker_rkl"}],
    }
    base_recipe["train"]["max_steps"] = 3
    base_recipe["train"]["log_every"] = 1

    cfg = recipe_from_dict(base_recipe)
    rep = check_recipe(cfg)
    assert rep.ok, rep.render()

    rt = train(cfg)
    metrics_path = Path(rt.config.train.output_dir) / "metrics.jsonl"
    assert metrics_path.exists()
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(records) >= 3
    for r in records:
        assert "loss/total" in r
        assert torch.isfinite(torch.tensor(r["loss/total"]))


def test_opsd_gjsd_tiny(base_recipe, tiny_models, opsd_toy_jsonl):
    """OPSD with token_kd (gjsd beta=0 token_clip=0.05), self teacher, privileged_prompt runs 3 steps."""
    if "gjsd" not in DIVERGENCES:
        pytest.skip("gjsd divergence not registered (T04)")

    base_recipe["data"]["train"] = {"type": "jsonl", "path": opsd_toy_jsonl}
    base_recipe["data"]["transforms"] = [
        {
            "type": "tokenize_prompt",
            "template": "Problem: {problem}",
            "use_chat_template": False,
            "keep_fields": ["problem", "solution"],
        }
    ]
    base_recipe["rollout"] = {"type": "hf"}
    base_recipe["producer"] = {
        "type": "mixed",
        "lmbda": 1.0,
        "sampling": {"max_new_tokens": 4, "temperature": 0.0},
    }
    base_recipe["teacher"] = {
        "provider": {"type": "self", "weight_policy": "live", "position_map": "response_shift"},
        "input": {
            "type": "privileged_prompt",
            "template": "Problem: {problem}\nSolution: {solution}",
            "fields": ["problem", "solution"],
            "use_chat_template": False,
        },
    }
    base_recipe["loss"] = {
        "terms": [
            {
                "criterion": "token_kd",
                "alias": "opsd",
                "params": {"divergence": "gjsd", "beta": 0.0, "token_clip": 0.05, "temperature": 1.0},
            }
        ],
    }
    base_recipe["train"]["max_steps"] = 3
    base_recipe["train"]["log_every"] = 1

    cfg = recipe_from_dict(base_recipe)
    rep = check_recipe(cfg)
    assert rep.ok, rep.render()

    rt = train(cfg)
    metrics_path = Path(rt.config.train.output_dir) / "metrics.jsonl"
    assert metrics_path.exists()
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(records) >= 3
    for r in records:
        assert "loss/total" in r
        assert torch.isfinite(torch.tensor(r["loss/total"]))
