"""E2E training tests for DSKD family (T12)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from lmkd.core.config import recipe_from_dict
from lmkd.train.builder import build_runtime


@pytest.mark.parametrize("crit_name", ["dskd", "dskdv2", "dskd_cma", "dskdv2_eta"])
def test_dskd_family_runs(crit_name, base_recipe, tiny_models, xtok_models, tmp_path):
    recipe = copy.deepcopy(base_recipe)
    recipe["train"]["output_dir"] = str(tmp_path / crit_name)
    recipe["train"]["max_steps"] = 3
    recipe["train"]["batch_size"] = 2
    recipe["train"]["micro_batch_size"] = 2
    recipe["train"]["log_every"] = 1

    if crit_name in ("dskd", "dskdv2"):
        recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
        recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
        recipe["loss"] = {
            "terms": [
                {"criterion": "nll", "weight": 0.5},
                {"criterion": crit_name, "alias": crit_name, "weight": 0.5},
            ]
        }
    elif crit_name == "dskd_cma":
        recipe["models"]["student"] = {"path": xtok_models["student"], "dtype": "fp32"}
        recipe["models"]["teacher"] = {"path": xtok_models["teacher"], "dtype": "fp32"}
        recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
        recipe["data"]["transforms"] = [
            {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"},
        ]
        recipe["loss"] = {
            "terms": [
                {"criterion": "nll", "weight": 0.5},
                {"criterion": crit_name, "alias": crit_name, "weight": 0.5},
            ]
        }
    elif crit_name == "dskdv2_eta":
        recipe["models"]["student"] = {"path": xtok_models["student"], "dtype": "fp32"}
        recipe["models"]["teacher"] = {"path": xtok_models["teacher"], "dtype": "fp32"}
        recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
        recipe["data"]["transforms"] = [
            {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"},
            {"type": "load_vocab_map", "cache_path": str(tmp_path / "vm.pt")},
            {"type": "token_offsets", "teacher": True},
            {"type": "align", "aligner": {"type": "exact_token"}},
        ]
        recipe["loss"] = {
            "terms": [
                {"criterion": "nll", "weight": 0.5},
                {"criterion": crit_name, "alias": crit_name, "weight": 0.5},
            ]
        }

    rt = build_runtime(recipe_from_dict(recipe))
    rt.begin()

    # Capture gradient during backward
    t2s_proj = rt.modules.get(f"{crit_name}__t2s")
    first_linear = next(m for m in t2s_proj.modules() if isinstance(m, torch.nn.Linear))
    grad_holder: list[torch.Tensor] = []
    first_linear.weight.register_hook(lambda g: grad_holder.append(g.clone()))

    # Step 1
    rt.train_step()
    assert len(grad_holder) > 0
    assert (grad_holder[0].abs().sum() > 0).item(), "projector should have non-zero grad after step 1"

    # Step 2 and 3
    rt.train_step()
    rt.train_step()
    rt.end()

    # Verify metrics logged
    metrics_path = Path(recipe["train"]["output_dir"]) / "metrics.jsonl"
    assert metrics_path.exists()
    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) >= 3

    # Check loss is finite and t2s_ce is present
    last_log = eval(lines[-1]) if lines[-1].startswith("{") else None
    if last_log is None:
        last_log = json.loads(lines[-1])
    assert f"loss/{crit_name}/t2s_ce" in last_log
    assert math.isfinite(last_log[f"loss/{crit_name}/t2s_ce"])


def test_train_s2t_flag(base_recipe, tiny_models, tmp_path):
    recipe = copy.deepcopy(base_recipe)
    recipe["train"]["output_dir"] = str(tmp_path / "dskd_frozen_s2t")
    recipe["train"]["max_steps"] = 3
    recipe["train"]["batch_size"] = 2
    recipe["train"]["micro_batch_size"] = 2
    recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    recipe["loss"] = {
        "terms": [
            {"criterion": "nll", "weight": 0.5},
            {"criterion": "dskd", "alias": "dskd", "weight": 0.5, "params": {"train_s2t": False}},
        ]
    }

    rt = build_runtime(recipe_from_dict(recipe))
    s2t_proj = rt.modules.get("dskd__s2t")
    initial_weight = s2t_proj.weight.detach().clone()

    rt.begin()
    for _ in range(3):
        rt.train_step()
    rt.end()

    assert torch.equal(s2t_proj.weight, initial_weight), "s2t parameters should not change when train_s2t=False"
