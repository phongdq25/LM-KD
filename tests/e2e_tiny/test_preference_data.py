"""Tiny listwise pipeline and optimizer steps through Runtime.train_step."""

import json

import torch

import lmkd


def test_listwise_cache_pipeline_train_step(tiny_models, tmp_path):
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
        "loss": {"terms": [{"criterion": "nll"}]},
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {"lr": 1e-3},
        "train": {
            "output_dir": str(tmp_path / "run"),
            "max_steps": 3,
            "epochs": None,
            "batch_size": 4,
            "micro_batch_size": 2,
        },
    }
    rt = lmkd.build(recipe)
    rt.begin()
    for _ in range(3):
        out = rt.train_step()
        assert torch.isfinite(out.metrics["loss/total"])
    rt.end()
    assert rt.state.step == 3
