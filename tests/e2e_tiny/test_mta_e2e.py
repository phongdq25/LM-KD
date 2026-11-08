"""End-to-end training verification for MTA on tiny models."""

import copy
import json
import math
from pathlib import Path

from lmkd.core.config import recipe_from_dict
from lmkd.train.builder import build_runtime


def test_mta_e2e_training(base_recipe, tiny_models, tmp_path):
    recipe = copy.deepcopy(base_recipe)
    recipe["models"] = {
        "student": {"path": tiny_models["student"], "dtype": "fp32", "attn_implementation": "eager"},
        "teacher": {"path": tiny_models["teacher"], "dtype": "fp32", "attn_implementation": "eager"},
    }
    recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    recipe["data"]["transforms"] = [
        {"type": "tokenize_sft", "use_chat_template": False, "keep_text": True},
        {"type": "token_offsets", "teacher": False},
        {"type": "align", "aligner": {"type": "mta_span"}},
    ]
    recipe["loss"] = {
        "aggregator": {"type": "kd_mix", "kd_ratio": 0.5, "ce_alias": "nll"},
        "terms": [
            {"criterion": "nll"},
            {
                "criterion": "mta",
                "alias": "mta",
                "params": {
                    "student_layers": [-1],
                    "teacher_layers": [-1],
                    "split_layer_mapping": [0, 0, 1],
                    "token_loss_weight": 0.1,
                },
            },
        ],
    }
    recipe["optim"] = {
        "groups": [
            {"name": "projector", "match": {"group": "projector"}, "lr": 0.0005},
            {"name": "default", "match": {}},
        ],
        "optimizers": {"default": {"type": "adamw"}},
        "lr": 0.001,
    }
    recipe["train"].update(
        output_dir=str(tmp_path / "mta_run"),
        max_steps=3,
        batch_size=4,
        micro_batch_size=2,
        log_every=1,
    )

    rt = build_runtime(recipe_from_dict(recipe))
    captured = []
    param = next(rt.modules.get("mta__proj_0").parameters())
    hook = param.register_hook(lambda grad: captured.append(grad.detach().norm().item()))

    rt.begin()
    for _ in range(3):
        rt.train_step()
    rt.end()
    hook.remove()

    metrics_file = Path(recipe["train"]["output_dir"]) / "metrics.jsonl"
    assert metrics_file.exists()
    lines = metrics_file.read_text().splitlines()
    assert len(lines) >= 3
    last_record = json.loads(lines[-1])
    assert "loss/mta/mta" in last_record
    assert math.isfinite(last_record["loss/mta/mta"])
    assert captured and max(captured) > 0


def test_dskdv2_mta_e2e_training(base_recipe, tiny_models, tmp_path):
    recipe = copy.deepcopy(base_recipe)
    recipe["models"] = {
        "student": {"path": tiny_models["student"], "dtype": "fp32", "attn_implementation": "eager"},
        "teacher": {"path": tiny_models["teacher"], "dtype": "fp32", "attn_implementation": "eager"},
    }
    recipe["teacher"] = {"provider": "hf_frozen", "input": "pretokenized"}
    recipe["data"]["transforms"] = [
        {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "keep_text": True},
        {"type": "token_offsets", "teacher": True},
        {"type": "align", "aligner": {"type": "mta_span"}},
    ]
    recipe["loss"] = {
        "aggregator": "weighted_sum",
        "terms": [
            {"criterion": "nll", "alias": "nll", "weight": 0.2},
            {"criterion": "dskdv2", "alias": "kd", "weight": 0.8},
            {
                "criterion": "mta",
                "alias": "mta",
                "weight": 1.0,
                "params": {
                    "student_layers": [-1],
                    "teacher_layers": [-1],
                    "split_layer_mapping": [0, 0, 1],
                    "token_loss_weight": 0.0,
                    "mode": "cross_tokenizer",
                },
            },
        ],
    }
    recipe["optim"] = {
        "groups": [
            {"name": "projector", "match": {"group": "projector"}, "lr": 0.0005},
            {"name": "default", "match": {}},
        ],
        "optimizers": {"default": {"type": "adamw"}},
        "lr": 0.001,
    }
    recipe["train"].update(
        output_dir=str(tmp_path / "dskdv2_mta_run"),
        max_steps=2,
        batch_size=4,
        micro_batch_size=2,
        log_every=1,
    )

    rt = build_runtime(recipe_from_dict(recipe))
    rt.begin()
    for _ in range(2):
        rt.train_step()
    rt.end()

    metrics_file = Path(recipe["train"]["output_dir"]) / "metrics.jsonl"
    assert metrics_file.exists()
    lines = metrics_file.read_text().splitlines()
    assert len(lines) >= 2
    last_record = json.loads(lines[-1])
    assert "loss/mta/mta" in last_record
    assert math.isfinite(last_record["loss/mta/mta"])

