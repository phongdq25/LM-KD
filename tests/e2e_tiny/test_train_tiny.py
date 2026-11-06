"""End-to-end on tiny random GPT-2 (CPU). These are the smoke tests every task must keep green."""

import json
from pathlib import Path

import pytest
import torch

from lmkd.core.config import recipe_from_dict
from lmkd.core.validate import check_recipe
from lmkd.train.trainer import train


def _metrics(run_dir):
    return [json.loads(l) for l in (Path(run_dir) / "metrics.jsonl").read_text().splitlines()]


def test_sft_runs_and_checkpoints(base_recipe):
    cfg = recipe_from_dict(base_recipe)
    assert check_recipe(cfg).ok
    rt = train(cfg)
    assert rt.state.step == 4
    m = _metrics(cfg.train.output_dir)
    assert "loss/nll/nll" in m[-1]
    assert (Path(cfg.train.output_dir) / "checkpoint-final" / "student" / "config.json").exists()


def test_sft_loss_decreases(base_recipe):
    base_recipe["train"]["max_steps"] = 30
    base_recipe["train"]["log_every"] = 10
    base_recipe["optim"]["lr"] = 5e-3
    cfg = recipe_from_dict(base_recipe)
    train(cfg)
    m = _metrics(cfg.train.output_dir)
    assert m[-1]["loss/total"] < m[0]["loss/total"]


@pytest.mark.parametrize("kd_type", ["fkl", "rkl", "jsd", "tvd", "sfkl", "srkl", "csd"])
def test_kd_same_tokenizer(base_recipe, tiny_models, kd_type):
    base_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    base_recipe["loss"] = {
        "aggregator": {"type": "kd_mix", "kd_ratio": 0.5},
        "terms": [
            {"criterion": "nll"},
            {"criterion": "token_kd", "alias": "kd", "params": {"divergence": kd_type, "chunk_size": 8}},
        ],
    }
    cfg = recipe_from_dict(base_recipe)
    assert check_recipe(cfg).ok, check_recipe(cfg).render()
    train(cfg)
    m = _metrics(cfg.train.output_dir)
    assert f"loss/kd/{kd_type}" in m[-1]
    assert torch.isfinite(torch.tensor(m[-1]["loss/total"]))


def test_check_catches_errors(base_recipe):
    base_recipe["loss"]["terms"] = [
        {"criterion": "token_kd", "params": {"divergence": "fkl", "bogus": 1}},
        {"criterion": "weighted_nll"},
        {"criterion": "impact"},
        {"criterion": "token_kd", "alias": "kd_ok"},
    ]
    rep = check_recipe(recipe_from_dict(base_recipe))
    text = rep.render()
    assert not rep.ok
    assert "bogus" in text  # strict params
    assert "seq.token_weights" in text  # requires graph
    assert "align.impact_lo" in text  # stable IMPACT requires offline boundary tables
    assert "needs a teacher" in text
