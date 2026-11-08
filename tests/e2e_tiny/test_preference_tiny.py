"""E2E tiny tests for preference distillation recipes."""

from __future__ import annotations

import json

import torch

import lmkd


def test_vpd_tiny(tiny_models, tmp_path):
    source = tmp_path / "prefs.jsonl"
    source.write_text(
        "\n".join(json.dumps({"id": str(i), "prompt": f"Q: w{i}", "responses": [" A: w2", " A: w3"]}) for i in range(4))
    )
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
                "source": {"type": "jsonl", "path": str(source)},
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
        "loss": {"terms": [{"criterion": "vpd", "params": {"beta": 1.0, "fill_missing_from_teacher": True}}]},
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {"lr": 1e-3},
        "train": {
            "output_dir": str(tmp_path / "run_vpd"),
            "max_steps": 3,
            "epochs": None,
            "batch_size": 4,
            "micro_batch_size": 2,
        },
    }
    rt = lmkd.build(recipe)
    rt.begin()
    last_out = None
    for _ in range(3):
        last_out = rt.train_step()
        assert torch.isfinite(last_out.metrics["loss/total"])
    rt.end()
    assert rt.state.step == 3
    assert last_out is not None
    assert "metric/vpd/loss" in last_out.metrics


def test_dckd_tiny_with_cache(tiny_models, tmp_path):
    source = tmp_path / "prefs.jsonl"
    source.write_text(
        "\n".join(json.dumps({"id": str(i), "prompt": f"Q: w{i}", "responses": [" A: w2", " A: w3"]}) for i in range(4))
    )
    recipe = {
        "models": {"student": {"path": tiny_models["student"], "dtype": "fp32"}},
        "data": {"max_length": 32, "pad_to_multiple_of": None},
        "pipeline": [
            {
                "stage": "prepare_dataset",
                "name": "prepare",
                "source": {"type": "jsonl", "path": str(source)},
                "transforms": [
                    {
                        "type": "tokenize_listwise",
                        "format_plain_prompts": False,
                        "max_prompt_length": 16,
                        "max_length": 32,
                    }
                ],
            },
            {
                "stage": "teacher_cache",
                "inputs": {"data": "prepare"},
                "dataset": True,
                "model": tiny_models["teacher"],
                "dtype": "fp32",
                "kind": "topk",
                "k": 4,
                "batch_size": 2,
            },
        ],
        "producer": {"type": "listwise", "shuffle": False, "pad_groups_to": 2},
        "teacher": {"provider": {"type": "cache", "kind": "topk"}},
        "loss": {"terms": [{"criterion": "compressed_kl", "params": {"chosen_weight": 0.5, "rejected_weight": 0.5}}]},
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {"lr": 1e-3},
        "train": {
            "output_dir": str(tmp_path / "run_dckd"),
            "max_steps": 2,
            "epochs": None,
            "batch_size": 4,
            "micro_batch_size": 2,
        },
    }
    rt = lmkd.build(recipe)
    rt.begin()
    for _ in range(2):
        out = rt.train_step()
        assert torch.isfinite(out.metrics["loss/total"])
    rt.end()
    assert rt.state.step == 2
