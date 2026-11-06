from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lmkd.cli import main
from lmkd.core.config import recipe_from_dict
from lmkd.pipeline.stages.prepare import PrepareDataset
from lmkd.train.trainer import Trainer


def test_prepare_dataset_then_train(base_recipe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base_recipe["pipeline"] = [
        {
            "stage": "prepare_dataset",
            "name": "data",
            "source": base_recipe["data"]["train"],
            "transforms": [{"type": "tokenize_sft", "use_chat_template": False}],
        }
    ]
    base_recipe["data"]["transforms"] = []
    base_recipe["train"]["max_steps"] = 2
    base_recipe["train"]["output_dir"] = str(tmp_path / "run_prep")
    base_recipe["artifacts_dir"] = str(tmp_path / "art_prep")

    runs = {"count": 0}
    orig_run = PrepareDataset.run

    def tracked_run(self, ctx):
        runs["count"] += 1
        return orig_run(self, ctx)

    monkeypatch.setattr(PrepareDataset, "run", tracked_run)

    # First run: should execute prepare_dataset.run and train 2 steps
    cfg1 = recipe_from_dict(base_recipe)
    rt1 = Trainer.from_config(cfg1).fit()
    assert runs["count"] == 1
    assert rt1.state.step == 2

    # Second run: cached! prepare_dataset.run must not be called again
    base_recipe["train"]["output_dir"] = str(tmp_path / "run_prep_2")
    cfg2 = recipe_from_dict(base_recipe)
    rt2 = Trainer.from_config(cfg2).fit()
    assert runs["count"] == 1
    assert rt2.state.step == 2


def test_cli_pipeline_dry_run(base_recipe, tmp_path: Path, capsys: pytest.CaptureFixture):
    base_recipe["pipeline"] = [
        {
            "stage": "prepare_dataset",
            "name": "prep",
            "source": base_recipe["data"]["train"],
            "transforms": [{"type": "tokenize_sft", "use_chat_template": False}],
        }
    ]
    base_recipe["data"]["transforms"] = []
    recipe_file = tmp_path / "recipe.yaml"
    recipe_file.write_text(yaml.safe_dump(base_recipe))

    code = main(["pipeline", "-c", str(recipe_file), "--dry-run"])
    assert code == 0
    captured = capsys.readouterr()
    assert "prep" in captured.out
    assert "TO RUN" in captured.out
