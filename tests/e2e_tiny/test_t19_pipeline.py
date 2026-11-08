"""Tiny CPU Long-CoT signal pipeline through two weighted training steps."""

from __future__ import annotations

import json
from pathlib import Path

from lmkd.cli import main
from lmkd.core.config import recipe_from_dict
from lmkd.pipeline.io import read_jsonl
from lmkd.train.trainer import Trainer


def test_spectral_weights_then_train(base_recipe, tiny_models, tmp_path) -> None:
    raw = tmp_path / "cot.jsonl"
    raw.write_text("\n".join(json.dumps({"id": str(i), "problem": "w1 w2", "response": "w3. w4."}) for i in range(8)))
    base_recipe["pipeline"] = [
        {
            "stage": "prepare_dataset",
            "name": "prepare",
            "source": {"type": "jsonl", "path": str(raw)},
            "transforms": [
                {"type": "render_cot", "chat_template": False},
                {"type": "tokenize_cot", "verify": False},
                {"type": "segment"},
            ],
        },
        {
            "stage": "spectral_capture",
            "name": "capture",
            "inputs": {"data": "prepare"},
            "model": tiny_models["student"],
            "dtype": "fp32",
            "device": "cpu",
            "chunk": 2,
        },
        {
            "stage": "build_weights",
            "name": "weights",
            "inputs": {"data": "prepare", "signals": "capture"},
            "chain": [{"type": "select_energy_top_p", "p": 1.0}, {"type": "tail_boost", "boost": 2.0}],
            "dataset": True,
        },
    ]
    base_recipe["data"]["transforms"] = []
    base_recipe["loss"]["terms"] = [{"criterion": "weighted_nll"}]
    base_recipe["train"]["max_steps"] = 2
    base_recipe["train"]["batch_size"] = 4
    base_recipe["train"]["micro_batch_size"] = 2
    base_recipe["artifacts_dir"] = str(tmp_path / "artifacts")
    rt = Trainer.from_config(recipe_from_dict(base_recipe)).fit()
    assert rt.state.step == 2
    art = rt.extra["pipeline_artifacts"]["weights"]
    samples = read_jsonl(art.path / "samples.jsonl")
    assert len(samples) == 8
    assert all("seq.loss_mask" in row and "seq.token_weights" in row for row in samples)


def test_sample_recipe_pipeline_dry_run(tiny_models, toy_jsonl, capsys) -> None:
    recipe = Path("recipes/long_cot/sgl_spectral.yaml")
    assert (
        main(
            [
                "pipeline",
                "-c",
                str(recipe),
                "--set",
                f"models.student.path={tiny_models['student']}",
                f"pipeline.0.source.path={toy_jsonl}",
                f"pipeline.0.model={tiny_models['student']}",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "sgl_spectral" in capsys.readouterr().out
