import json
from pathlib import Path

import yaml

from lmkd.cli import main


def _write_minimal_recipe(path: Path) -> Path:
    # A self-contained minimal recipe without requiring real model paths if validate doesn't load model
    data = {
        "models": {"student": {"path": "dummy-gpt2"}},
        "data": {"train": {"type": "jsonl", "path": "train.jsonl"}},
        "loss": {"terms": [{"criterion": "nll"}]},
        "train": {"max_steps": 5},
    }
    path.write_text(yaml.safe_dump(data))
    return path


def test_cli_check_json(tmp_path, capsys):
    recipe_path = _write_minimal_recipe(tmp_path / "recipe.yaml")
    code = main(["check", "-c", str(recipe_path), "--json"])
    captured = capsys.readouterr()
    assert code == 0
    res = json.loads(captured.out)
    assert res.get("ok") is True
    assert "errors" in res
    assert "warnings" in res


def test_cli_config_dump(tmp_path, capsys):
    recipe_path = _write_minimal_recipe(tmp_path / "recipe.yaml")
    code = main(["config", "-c", str(recipe_path)])
    captured = capsys.readouterr()
    assert code == 0
    assert "max_steps" in captured.out


def test_cli_new_criterion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lmkd" / "losses").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "unit").mkdir(parents=True, exist_ok=True)
    (tmp_path / "recipes").mkdir(parents=True, exist_ok=True)

    code = main(["new", "criterion", "foo_kd"])
    assert code == 0

    assert (tmp_path / "src" / "lmkd" / "losses" / "foo_kd.py").exists()
    assert (tmp_path / "tests" / "unit" / "test_foo_kd.py").exists()
    assert (tmp_path / "recipes" / "experimental" / "foo_kd.yaml").exists()

    # Second run should fail with exit code 1
    code2 = main(["new", "criterion", "foo_kd"])
    assert code2 == 1


def test_cli_list_verbose(capsys):
    code = main(["list", "criterions", "--verbose"])
    captured = capsys.readouterr()
    assert code == 0
    assert "source" in captured.out
    assert "chunk_size" in captured.out
