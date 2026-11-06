import json
from pathlib import Path

import pytest

from lmkd.core.config import recipe_from_dict
from lmkd.train.builder import build_runtime
from lmkd.train.schedules import schedule_value
from lmkd.train.trainer import train


def test_schedule_values():
    # linear(0->1, until=0.5) at progress 0.25 = 0.5, at 0.9 = 1.0
    spec_lin = {"type": "linear", "start": 0.0, "end": 1.0, "until": 0.5}
    assert pytest.approx(0.5) == schedule_value(spec_lin, step=25, total=100)
    assert pytest.approx(1.0) == schedule_value(spec_lin, step=90, total=100)

    # warmup_steps(steps=4) at step 2 = 0.5 (with start=0, end=1)
    spec_ws = {"type": "warmup_steps", "start": 0.0, "end": 1.0, "steps": 4}
    assert pytest.approx(0.5) == schedule_value(spec_ws, step=2, total=100)
    assert pytest.approx(1.0) == schedule_value(spec_ws, step=4, total=100)

    # step(values=[1, 2], boundaries=[0.5])
    spec_step = {"type": "step", "values": [1.0, 2.0], "boundaries": [0.5]}
    assert pytest.approx(1.0) == schedule_value(spec_step, step=25, total=100)
    assert pytest.approx(2.0) == schedule_value(spec_step, step=75, total=100)


def test_weight_schedule_e2e(base_recipe):
    base_recipe["loss"]["terms"] = [
        {"criterion": "nll", "alias": "a", "weight": 1.0},
        {"criterion": "nll", "alias": "b", "weight": 0.0},
    ]
    base_recipe["loss"]["schedules"] = {"b.weight": {"type": "linear", "start": 0.0, "end": 1.0}}
    base_recipe["train"]["max_steps"] = 4
    base_recipe["train"]["log_every"] = 1
    cfg = recipe_from_dict(base_recipe)
    rt = train(cfg)

    # rt.loss_weights["b"] final = 0.75 (value set at step 4 begin with progress 3/4)
    assert pytest.approx(0.75) == rt.loss_weights["b"]

    metrics = [json.loads(l) for l in (Path(cfg.train.output_dir) / "metrics.jsonl").read_text().splitlines()]
    b_weights = [m["sched/b.weight"] for m in metrics if "sched/b.weight" in m]
    assert len(b_weights) == 4
    # sched/b.weight increases monotonically
    for i in range(1, len(b_weights)):
        assert b_weights[i] > b_weights[i - 1]


def test_weight_from_zero_gets_features(base_recipe, tiny_models):
    base_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    base_recipe["loss"]["terms"] = [
        {"criterion": "nll", "alias": "nll", "weight": 1.0},
        {"criterion": "token_kd", "alias": "kd", "weight": 0.0, "params": {"chunk_size": 8}},
    ]
    base_recipe["loss"]["schedules"] = {"kd.weight": {"type": "step", "values": [0.0, 0.5], "boundaries": [0.5]}}
    base_recipe["train"]["max_steps"] = 4
    base_recipe["train"]["log_every"] = 1
    cfg = recipe_from_dict(base_recipe)
    train(cfg)

    metrics = [json.loads(l) for l in (Path(cfg.train.output_dir) / "metrics.jsonl").read_text().splitlines()]
    # steps 1-2 have progress 0/4=0.0 and 1/4=0.25 (< 0.5), so weight = 0.0 -> loss/kd/fkl NOT present
    # steps 3-4 have progress 2/4=0.5 and 3/4=0.75 (>= 0.5), so weight = 0.5 -> loss/kd/fkl present
    assert "loss/kd/fkl" not in metrics[0]
    assert "loss/kd/fkl" not in metrics[1]
    assert "loss/kd/fkl" in metrics[2]
    assert "loss/kd/fkl" in metrics[3]


def test_unknown_alias_or_field_errors(base_recipe):
    base_recipe["loss"]["terms"] = [{"criterion": "nll", "alias": "a"}]
    # Unknown alias
    base_recipe["loss"]["schedules"] = {"non_existent.weight": {"type": "constant", "value": 1.0}}
    with pytest.raises(ValueError, match="unknown loss alias"):
        build_runtime(recipe_from_dict(base_recipe))

    # Non-existent field
    base_recipe["loss"]["schedules"] = {"a.non_existent_field": {"type": "constant", "value": 1.0}}
    with pytest.raises(ValueError, match="field .* not in"):
        build_runtime(recipe_from_dict(base_recipe))


def test_cached_field_not_schedulable(base_recipe, tiny_models):
    base_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    base_recipe["loss"]["terms"] = [{"criterion": "token_kd", "alias": "kd", "params": {"divergence": "sfkl"}}]
    base_recipe["loss"]["schedules"] = {"kd.skew_alpha": {"type": "linear", "start": 0.1, "end": 0.5}}
    with pytest.raises(ValueError, match="not schedulable"):
        build_runtime(recipe_from_dict(base_recipe))


def test_schedule_value_every_type_and_errors():
    cos = {"type": "cosine", "start": 1.0, "end": 0.0, "begin": 0.2, "until": 0.8}
    assert schedule_value(cos, step=10, total=100) == 1.0
    assert schedule_value(cos, step=50, total=100) == pytest.approx(0.5)
    assert schedule_value(cos, step=90, total=100) == 0.0
    warm = {"type": "linear_warmup", "start": 0.0, "end": 2.0, "until": 0.5}
    assert schedule_value(warm, step=25, total=100) == pytest.approx(1.0)
    assert schedule_value(warm, step=60, total=100) == 2.0
    piece = {"type": "step", "values": [0.1, 0.2, 0.3], "boundaries": [0.3, 0.6]}
    assert [schedule_value(piece, step=s, total=100) for s in (0, 40, 80)] == [0.1, 0.2, 0.3]
    assert schedule_value({"type": "constant", "value": 3}, step=7, total=100) == 3.0
    with pytest.raises(ValueError, match="unknown schedule type 'nope'.*available"):
        schedule_value({"type": "nope"}, step=1, total=10)
