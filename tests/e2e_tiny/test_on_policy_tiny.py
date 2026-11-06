"""E2E tiny tests for on-policy training with mixed and adaptive_sgo producers (T09)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from lmkd.core.config import recipe_from_dict
from lmkd.core.validate import check_recipe
from lmkd.train.trainer import train


def _read_metrics(run_dir: str | Path) -> list[dict]:
    metrics_file = Path(run_dir) / "metrics.jsonl"
    if not metrics_file.exists():
        return []
    return [json.loads(line) for line in metrics_file.read_text().splitlines()]


def test_mixed_on_policy_kd_runs(base_recipe, tiny_models):
    """MixedProducer with lmbda=1 always generates on-policy; logs producer/on_policy=1.0 and loss is finite."""
    base_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    base_recipe["rollout"] = {"type": "hf"}
    base_recipe["producer"] = {
        "type": "mixed",
        "lmbda": 1.0,
        "sampling": {"max_new_tokens": 4, "temperature": 0.0},
    }
    base_recipe["loss"] = {
        "aggregator": {"type": "kd_mix", "kd_ratio": 0.5},
        "terms": [
            {"criterion": "nll"},
            {"criterion": "token_kd", "alias": "kd", "params": {"divergence": "fkl", "chunk_size": 8}},
        ],
    }
    base_recipe["train"]["max_steps"] = 3
    base_recipe["train"]["log_every"] = 1

    cfg = recipe_from_dict(base_recipe)
    rep = check_recipe(cfg)
    assert rep.ok, rep.render()

    rt = train(cfg)
    assert rt.state.step == 3

    metrics = _read_metrics(cfg.train.output_dir)
    assert len(metrics) > 0
    for m in metrics:
        if "producer/on_policy" in m:
            assert m["producer/on_policy"] == 1.0
        if "loss/total" in m:
            assert torch.isfinite(torch.tensor(m["loss/total"]))


def test_adaptive_sgo_runs(base_recipe, tiny_models):
    """AdaptiveSGOProducer runs with init_threshold=1.0, generating samples and exercising replay."""
    base_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    base_recipe["rollout"] = {"type": "hf"}
    base_recipe["producer"] = {
        "type": "adaptive_sgo",
        "init_threshold": 1.0,
        "schedule": "decreasing",
        "capacity": 2,
        "sampling": {"max_new_tokens": 4, "temperature": 0.0},
    }
    base_recipe["loss"] = {
        "aggregator": {"type": "kd_mix", "kd_ratio": 0.5},
        "terms": [
            {"criterion": "nll"},
            {"criterion": "token_kd", "alias": "kd", "params": {"divergence": "fkl", "chunk_size": 8}},
        ],
    }
    base_recipe["train"]["max_steps"] = 4
    base_recipe["train"]["log_every"] = 1

    cfg = recipe_from_dict(base_recipe)
    rep = check_recipe(cfg)
    assert rep.ok, rep.render()

    rt = train(cfg)
    assert rt.state.step == 4

    metrics = _read_metrics(cfg.train.output_dir)
    assert len(metrics) > 0
    for m in metrics:
        if "loss/total" in m:
            assert torch.isfinite(torch.tensor(m["loss/total"]))
