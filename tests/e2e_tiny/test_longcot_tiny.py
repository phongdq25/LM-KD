from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lmkd.core.config import load_recipe, recipe_from_dict
from lmkd.core.validate import check_recipe
from lmkd.train.trainer import train


def _metrics(run_dir: str | Path) -> list[dict]:
    p = Path(run_dir) / "metrics.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_trans_e2e(tiny_models: dict[str, str], tmp_path: Path) -> None:
    """3 steps of training with transition_loss; predictor in optim groups with lr 2e-4."""
    data_file = tmp_path / "train.jsonl"
    rows = []
    for _ in range(16):
        rows.append(
            {
                "problem": "w1 w2 w3",
                "response": "w4 w5 w6 w7 w8\n\nw9 w10 w11 w12 w13\n\nw14 w15 w16 w17 w18",
            }
        )
    data_file.write_text("\n".join(json.dumps(r) for r in rows))

    out_dir = tmp_path / "out_trans"
    recipe = {
        "models": {"student": {"path": tiny_models["student"], "dtype": "fp32"}},
        "data": {
            "train": {"type": "jsonl", "path": str(data_file)},
            "transforms": [
                {"type": "render_cot", "chat_template": False},
                {"type": "tokenize_cot", "verify": False},
                {"type": "step_transitions", "min_step_tokens": 4},
            ],
            "max_length": 128,
        },
        "loss": {
            "terms": [
                {"criterion": "nll", "weight": 1.0},
                {
                    "criterion": "transition_loss",
                    "weight": 0.3,
                    "params": {"layer": 1, "predictor_hidden": 16, "warmup_ratio": 0.1},
                },
            ]
        },
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {
            "lr": 1e-3,
            "groups": [
                {
                    "name": "trans_predictor",
                    "match": {"owner": "modules", "group": "trans_predictor"},
                    "lr": 2.0e-4,
                    "weight_decay": 0.0,
                },
                {"name": "no_decay", "match": {"ndim_le": 1}, "weight_decay": 0.0},
                {"name": "default", "match": {}},
            ],
        },
        "train": {
            "output_dir": str(out_dir),
            "max_steps": 3,
            "batch_size": 4,
            "micro_batch_size": 2,
            "log_every": 1,
        },
    }

    cfg = recipe_from_dict(recipe)
    rt = train(cfg)

    assert rt.state.step == 3
    # Check that predictor is in optimizer param groups with lr 2e-4
    opt = rt.optimizers["default"]
    predictor_groups = [g for g in opt.param_groups if g.get("group_name") == "trans_predictor"]
    assert len(predictor_groups) == 1
    assert predictor_groups[0]["initial_lr"] == 2.0e-4

    # Check metrics
    m = _metrics(out_dir)
    assert len(m) >= 3
    last = m[-1]
    assert "loss/transition_loss/trans" in last
    assert "metric/transition_loss/trans_cos" in last
    assert "metric/transition_loss/trans_pairs" in last
    assert "metric/transition_loss/lambda" in last

    # Check that predictor checkpoint is saved in modules.pt
    ckpt_dir = out_dir / "checkpoint-final"
    modules_pt = ckpt_dir / "modules.pt"
    assert modules_pt.exists()
    saved_modules = torch.load(modules_pt, map_location="cpu", weights_only=True)
    assert any("predictor" in k for k in saved_modules)


def test_per_source_objective_e2e(tiny_models: dict[str, str], tmp_path: Path) -> None:
    """3 steps of training with per_source_objective."""
    data_file = tmp_path / "train.jsonl"
    rows = []
    for _ in range(16):
        rows.append(
            {
                "problem": "w1 w2 w3",
                "response": "w4 w5 w6 <End_of_Prefix> w7 w8 w9 w10 w11",
            }
        )
    data_file.write_text("\n".join(json.dumps(r) for r in rows))

    out_dir = tmp_path / "out_pso"
    recipe = {
        "models": {"student": {"path": tiny_models["student"], "dtype": "fp32"}},
        "data": {
            "train": {"type": "jsonl", "path": str(data_file)},
            "transforms": [
                {"type": "render_cot", "chat_template": False},
                {"type": "tokenize_cot", "verify": False},
                {"type": "provenance", "junction_tokens": 2},
            ],
            "max_length": 128,
        },
        "loss": {
            "terms": [
                {
                    "criterion": "per_source_objective",
                    "params": {"objectives": {"prefix": "nll", "cont": "dft", "junction": "nll"}},
                }
            ]
        },
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {"lr": 1e-3},
        "train": {
            "output_dir": str(out_dir),
            "max_steps": 3,
            "batch_size": 4,
            "micro_batch_size": 2,
            "log_every": 1,
        },
    }

    cfg = recipe_from_dict(recipe)
    rt = train(cfg)

    assert rt.state.step == 3
    m = _metrics(out_dir)
    assert len(m) >= 3
    last = m[-1]
    assert "loss/per_source_objective/obj" in last
    assert "metric/per_source_objective/dft_frac" in last


RECIPES_LONG_COT = sorted(Path("recipes/long_cot").glob("*.yaml"))


@pytest.mark.parametrize("recipe_path", RECIPES_LONG_COT, ids=lambda p: p.name)
def test_recipes_check(recipe_path: Path) -> None:
    """Every long_cot recipe passes check_recipe or xfails due to T19 stub stages."""
    cfg = load_recipe(recipe_path)
    rep = check_recipe(cfg)
    if not rep.ok:
        stub_errors = [e for e in rep.errors if "stub" in e]
        if stub_errors:
            pytest.xfail(f"{recipe_path.name} depends on T19 stub stages: {stub_errors}")
        pytest.fail(f"Recipe {recipe_path.name} failed check with non-stub errors: {rep.errors}")
    assert rep.ok
