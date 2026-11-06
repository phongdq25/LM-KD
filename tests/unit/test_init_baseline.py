"""Tests for ``lmkd init`` baseline/method template generator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

import lmkd
from lmkd.cli import main
from lmkd.core.config import load_recipe
from lmkd.core.registry import (
    ALGORITHMS,
    CALLBACKS,
    CRITERIONS,
    EVALUATORS,
    PRODUCERS,
    STAGES,
    STEP_STRATEGIES,
    TEACHER_INPUTS,
    TEACHERS,
    TRANSFORMS,
)
from lmkd.core.validate import check_recipe
from lmkd.plugins import load_plugins
from lmkd.scaffold.init_baseline import parse_init_target
from lmkd.scaffold.probe import probe_gradient


def test_parse_init_target():
    assert parse_init_target("my-kd") == ("my_kd", "dev", None)
    assert parse_init_target("my_kd", target="baseline") == ("my_kd", "baseline", None)
    assert parse_init_target("baselines/self_distill/my_kd") == ("my_kd", "baseline", "self_distill")
    assert parse_init_target("src/baselines/my_kd") == ("my_kd", "baseline", None)
    assert parse_init_target("dev/exp_kd") == ("exp_kd", "dev", None)


def test_cli_init_dev_and_default_execution(tmp_path: Path, monkeypatch, tiny_models, toy_jsonl):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "examples" / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "examples" / "data" / "train_sample.jsonl").write_text(
        Path(toy_jsonl).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    code = main(["init", "alpha_dev_kd"])
    assert code == 0

    pkg_dir = tmp_path / "dev" / "alpha_dev_kd"
    for fname in ("__init__.py", "config.py", "loss.py", "data.py", "teacher.py", "train.py", "eval.py", "recipe.yaml", "README.md"):
        assert (pkg_dir / fname).is_file(), f"Missing {fname}"
    assert (tmp_path / "recipes" / "dev" / "alpha_dev_kd.yaml").is_file()
    assert (tmp_path / "tests" / "unit" / "test_alpha_dev_kd.py").is_file()

    # Second run without --force should fail
    assert main(["init", "alpha_dev_kd"]) == 1
    # With --force should succeed
    assert main(["init", "alpha_dev_kd", "--force"]) == 0

    # Load generated plugins and verify all registries
    load_plugins([str(pkg_dir / "__init__.py")])
    assert CRITERIONS.build("alpha_dev_kd") is not None
    assert TRANSFORMS.build("alpha_dev_kd_transform") is not None
    assert PRODUCERS.build("alpha_dev_kd_producer") is not None
    assert STAGES.build("alpha_dev_kd_stage") is not None
    assert TEACHER_INPUTS.build("alpha_dev_kd_teacher_input") is not None
    assert TEACHERS.build("alpha_dev_kd_teacher") is not None
    assert ALGORITHMS.build("alpha_dev_kd_algorithm") is not None
    assert STEP_STRATEGIES.build("alpha_dev_kd_step") is not None
    assert CALLBACKS.build("alpha_dev_kd_callback") is not None
    assert EVALUATORS.build({"type": "alpha_dev_kd_eval", "data": {"type": "jsonl", "path": str(toy_jsonl)}}) is not None

    # Gradient probe on unmodified loss
    probe_gradient(CRITERIONS.build("alpha_dev_kd"))

    # Static check on both recipes
    for rpath in (pkg_dir / "recipe.yaml", tmp_path / "recipes" / "dev" / "alpha_dev_kd.yaml"):
        cfg = load_recipe(rpath)
        rep = check_recipe(cfg)
        assert rep.ok, rep.render()

    # End-to-end training run using unmodified template (falls back to framework defaults)
    rt = lmkd.run(
        pkg_dir / "recipe.yaml",
        overrides=[
            f"models.student.path={tiny_models['student']}",
            f"data.train.path={toy_jsonl}",
            f"train.output_dir={tmp_path / 'out'}",
            "train.max_steps=2",
        ],
    )
    assert rt.state.step == 2


def test_cli_init_baseline_and_interactive_prompt(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "examples" / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "examples" / "data" / "train_sample.jsonl").write_text(
        '{"prompt": "hello", "response": "world"}\n',
        encoding="utf-8",
    )

    # Test interactive prompt when name is omitted
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "beta_base_kd")

    code = main(["init", "--baseline", "--domain", "same_tokenizer"])
    assert code == 0

    pkg_dir = tmp_path / "src" / "baselines" / "same_tokenizer" / "beta_base_kd"
    assert (pkg_dir / "__init__.py").is_file()
    assert (pkg_dir / "loss.py").is_file()
    assert (tmp_path / "recipes" / "same_tokenizer" / "beta_base_kd.yaml").is_file()

    cfg = load_recipe(tmp_path / "recipes" / "same_tokenizer" / "beta_base_kd.yaml")
    rep = check_recipe(cfg)
    assert rep.ok, rep.render()


def test_cli_init_custom_override_loss_and_loop(tmp_path: Path, monkeypatch, tiny_models, toy_jsonl):
    """Verify user can override only loss.py (or train.py) while keeping all other parts default."""
    monkeypatch.chdir(tmp_path)
    assert main(["init", "custom_override_kd"]) == 0

    pkg_dir = tmp_path / "dev" / "custom_override_kd"
    loss_file = pkg_dir / "loss.py"

    # User customizes forward() in loss.py while leaving data.py, teacher.py, train.py untouched
    custom_loss_code = loss_file.read_text(encoding="utf-8").replace(
        "return super().forward(ctx)",
        """mask = ctx.supervised_mask()
        labels = ctx.batch["student.labels"]
        logp = ctx.student_logits().with_temperature(self.cfg.temperature).target_logp(labels)
        custom_val = -(logp * mask).sum() * self.cfg.alpha
        return LossOutput(terms={"custom_override_kd": LossTerm(custom_val, "token_sum")})""",
    )
    loss_file.write_text(custom_loss_code, encoding="utf-8")

    # Force reload of modified loss.py
    from lmkd.plugins import _USER_LOADED

    _USER_LOADED.clear()
    load_plugins([str(loss_file)])

    crit = CRITERIONS.build({"type": "custom_override_kd", "alpha": 0.25})
    out = probe_gradient(crit)
    assert "custom_override_kd" in out.terms
    assert torch.isfinite(out.terms["custom_override_kd"].value)

    # Run via the generated train.py main() entrypoint
    sys.path.insert(0, str(pkg_dir))
    try:
        from lmkd.plugins import _import_file

        _USER_LOADED.clear()
        _import_file(pkg_dir / "train.py")
        train_mod = next(m for k, m in sys.modules.items() if k.startswith("lmkd_user_train_"))
        rc = train_mod.main(
            [
                "-c",
                str(pkg_dir / "recipe.yaml"),
                "--set",
                f"models.student.path={tiny_models['student']}",
                f"data.train.path={toy_jsonl}",
                f"train.output_dir={tmp_path / 'out_custom'}",
                "train.max_steps=1",
            ]
        )
        assert rc == 0
    finally:
        sys.path.pop(0)


def test_generated_template_passes_ruff(tmp_path: Path, monkeypatch):
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    monkeypatch.chdir(tmp_path)
    assert main(["init", "lint_check_kd"]) == 0
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            str(pyproject),
            str(tmp_path / "dev" / "lint_check_kd"),
            str(tmp_path / "tests"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    m = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--ignore-missing-imports",
            "--check-untyped-defs",
            str(tmp_path / "dev" / "lint_check_kd"),
        ],
        capture_output=True,
        text=True,
    )
    assert m.returncode == 0, m.stdout + m.stderr
