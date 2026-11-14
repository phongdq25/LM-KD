"""Pipeline stage that executes a child training recipe."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from lmkd.core.config import recipe_from_dict
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.stage import StageContext
from lmkd.pipeline.stages.train import TrainRecipe
from lmkd.train.builder import build_runtime


def test_train_recipe_fingerprint_attach(base_recipe, tmp_path, monkeypatch):
    output = tmp_path / "child"
    recipe = tmp_path / "child.yaml"
    base_recipe["train"]["output_dir"] = str(output)
    recipe.write_text(yaml.safe_dump(base_recipe))
    stage = TrainRecipe(TrainRecipe.Config(recipe=str(recipe)))
    first = stage.fingerprint()
    recipe.write_text(recipe.read_text() + "# changed\n")
    assert stage.fingerprint() != first

    def fake_run(command, *, check):
        assert "train" in command
        assert check
        (output / "checkpoint-final").mkdir(parents=True)

    monkeypatch.setattr("subprocess.run", fake_run)
    artifact = Artifact(node="stage1", stage="train_recipe", key="abc", path=tmp_path / "artifact")
    runtime = SimpleNamespace(extra={})
    info = stage.run(StageContext(rt=runtime, output=artifact, node="stage1"))
    assert Path(info["output_dir"]) == output
    assert (artifact.path / "ckpt").resolve() == output / "checkpoint-final"
    stage.attach(artifact, runtime)
    assert runtime.extra["train_recipe:stage1"] == str(output)


def test_train_recipe_pipeline_tiny(base_recipe, tiny_models, tmp_path):
    stage1 = yaml.safe_load(yaml.safe_dump(base_recipe))
    stage1["models"]["student"] = {"path": tiny_models["teacher"], "dtype": "fp32", "trainable": False}
    stage1["loss"] = {"terms": [{"criterion": "residual_pretrain"}]}
    stage1["train"].update(output_dir=str(tmp_path / "stage1"), max_steps=2, batch_size=2, micro_batch_size=2)
    child_path = tmp_path / "child.yaml"
    child_path.write_text(yaml.safe_dump(stage1))

    main = yaml.safe_load(yaml.safe_dump(base_recipe))
    main["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    main["teacher"] = {"provider": "hf_frozen", "input": "same"}
    main["loss"] = {"terms": [{"criterion": "residualkd", "params": {"projector_ckpt": str(tmp_path / "stage1")}}]}
    main["pipeline"] = [{"stage": "train_recipe", "name": "stage1", "recipe": str(child_path)}]
    main["train"].update(output_dir=str(tmp_path / "main"), max_steps=1, batch_size=2, micro_batch_size=2)
    rt = build_runtime(recipe_from_dict(main))
    assert rt.extra["train_recipe:stage1"] == str(tmp_path / "stage1")
    rt.begin()
    rt.train_step()
    rt.end()
    assert (tmp_path / "main" / "checkpoint-final" / "COMPLETE").exists()
