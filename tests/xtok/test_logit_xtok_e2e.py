"""E2E training tests for ULD, MinED, MultiLevelOT, and E-SCOT (T13)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from lmkd.core.config import recipe_from_dict
from lmkd.train.builder import build_runtime


@pytest.mark.parametrize("crit_name", ["uld", "mined_fkl", "multilevel_ot", "escot"])
def test_logit_xtok_runs(crit_name, base_recipe, xtok_models, tmp_path):
    recipe = copy.deepcopy(base_recipe)
    recipe["train"]["output_dir"] = str(tmp_path / crit_name)
    recipe["train"]["max_steps"] = 3
    recipe["train"]["batch_size"] = 2
    recipe["train"]["micro_batch_size"] = 2
    recipe["train"]["log_every"] = 1

    recipe["models"]["student"] = {"path": xtok_models["student"], "dtype": "fp32"}
    recipe["models"]["teacher"] = {"path": xtok_models["teacher"], "dtype": "fp32"}
    recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}

    if crit_name in ("uld", "multilevel_ot"):
        recipe["data"]["transforms"] = [
            {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"},
        ]
        recipe["loss"] = {
            "aggregator": {"type": "weighted_sum"},
            "terms": [
                {"criterion": "nll", "weight": 1.0},
                {"criterion": crit_name, "alias": crit_name, "weight": 1.0},
            ],
        }
    elif crit_name == "mined_fkl":
        recipe["data"]["transforms"] = [
            {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"},
            {"type": "load_vocab_map", "cache_path": str(tmp_path / "vm.pt")},
            {"type": "align", "aligner": {"type": "dtw_edit", "require_id_map": False}},
        ]
        recipe["loss"] = {
            "aggregator": {"type": "kd_mix", "kd_ratio": 0.5},
            "terms": [
                {"criterion": "nll"},
                {"criterion": "mined_fkl", "alias": "mined_fkl", "params": {"temperature": 3.0}},
            ],
        }
    elif crit_name == "escot":
        recipe["data"]["transforms"] = [
            {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"},
            {"type": "token_offsets", "teacher": True},
            {"type": "align", "aligner": {"type": "char_span"}},
        ]
        recipe["loss"] = {
            "aggregator": {"type": "kd_mix", "kd_ratio": 0.5},
            "terms": [
                {"criterion": "nll"},
                {"criterion": "escot", "alias": "escot"},
            ],
        }

    rt = build_runtime(recipe_from_dict(recipe))
    rt.begin()
    for _ in range(3):
        rt.train_step()
    rt.end()

    metrics_path = Path(recipe["train"]["output_dir"]) / "metrics.jsonl"
    assert metrics_path.exists()
    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) >= 3

    last_log = json.loads(lines[-1])
    assert f"loss/{crit_name}/{crit_name}" in last_log
    assert math.isfinite(last_log[f"loss/{crit_name}/{crit_name}"])
    assert math.isfinite(last_log["loss/total"])
