"""E2E tiny tests for Self-Play Fine-Tuning baselines:

SPIN, T-SPIN, SPACE, SPPO, S-SPPO, and RSPO.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import lmkd


@pytest.fixture
def self_play_data(tmp_path: Path):
    source = tmp_path / "self_play.jsonl"
    rows = [
        {"id": str(i), "prompt": f"Q: prompt {i}", "responses": [f" A: chosen {i}", f" A: rollout {i}"]}
        for i in range(4)
    ]
    source.write_text("\n".join(json.dumps(r) for r in rows))
    return str(source)


@pytest.fixture
def triplet_data(tmp_path: Path):
    source = tmp_path / "triplet.jsonl"
    rows = [
        {
            "id": str(i),
            "prompt": f"Q: prompt {i}",
            "responses": [f" A: chosen {i}", f" A: rollout {i}", f" A: proto {i}"],
        }
        for i in range(3)
    ]
    source.write_text("\n".join(json.dumps(r) for r in rows))
    return str(source)


@pytest.mark.parametrize("method", ["spin", "space", "sppo", "s_sppo", "rspo"])
def test_self_play_tiny_pairwise(tiny_models, tmp_path, self_play_data, method):
    recipe = {
        "models": {
            "student": {"path": tiny_models["student"], "dtype": "fp32"},
            "teacher": {"path": tiny_models["teacher"], "dtype": "fp32"},
        },
        "data": {"max_length": 32, "pad_to_multiple_of": None},
        "pipeline": [
            {
                "stage": "prepare_dataset",
                "name": "prepare",
                "source": {"type": "jsonl", "path": self_play_data},
                "transforms": [
                    {
                        "type": "tokenize_listwise",
                        "format_plain_prompts": False,
                        "max_prompt_length": 16,
                        "max_length": 32,
                    }
                ],
                "dataset": True,
            },
        ],
        "producer": {"type": "listwise", "shuffle": False, "pad_groups_to": 2},
        "teacher": {"provider": "hf_frozen", "input": "same"},
        "loss": {"terms": [{"criterion": method}]},
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {"lr": 1e-3},
        "train": {
            "output_dir": str(tmp_path / f"run_{method}"),
            "max_steps": 2,
            "epochs": None,
            "batch_size": 4,
            "micro_batch_size": 2,
        },
    }
    rt = lmkd.build(recipe)
    rt.begin()
    last_out = None
    for _ in range(2):
        last_out = rt.train_step()
        assert torch.isfinite(last_out.metrics["loss/total"])
    rt.end()
    assert rt.state.step == 2
    assert last_out is not None


def test_tspin_tiny_triplet(tiny_models, tmp_path, triplet_data):
    recipe = {
        "models": {
            "student": {"path": tiny_models["student"], "dtype": "fp32"},
        },
        "data": {"max_length": 32, "pad_to_multiple_of": None},
        "pipeline": [
            {
                "stage": "prepare_dataset",
                "name": "prepare",
                "source": {"type": "jsonl", "path": triplet_data},
                "transforms": [
                    {
                        "type": "tokenize_listwise",
                        "format_plain_prompts": False,
                        "max_prompt_length": 16,
                        "max_length": 32,
                    }
                ],
                "dataset": True,
            },
        ],
        "producer": {"type": "listwise", "shuffle": False, "pad_groups_to": 3},
        "teacher": {"provider": "none", "input": "same"},
        "loss": {"terms": [{"criterion": "tspin", "params": {"alpha": 1.0, "beta": 0.1}}]},
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {"lr": 1e-3},
        "train": {
            "output_dir": str(tmp_path / "run_tspin"),
            "max_steps": 2,
            "epochs": None,
            "batch_size": 3,
            "micro_batch_size": 3,
        },
    }
    rt = lmkd.build(recipe)
    rt.begin()
    last_out = None
    for _ in range(2):
        last_out = rt.train_step()
        assert torch.isfinite(last_out.metrics["loss/total"])
    rt.end()
    assert rt.state.step == 2
    assert last_out is not None
