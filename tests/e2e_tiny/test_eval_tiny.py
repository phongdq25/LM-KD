"""End-to-end tests for evaluation (T07) on tiny random GPT-2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from lmkd.cli import main
from lmkd.core.config import recipe_from_dict
from lmkd.core.registry import ROLLOUT
from lmkd.eval.evaluators import MathPassK
from lmkd.train.trainer import train


def test_dev_loss_eval(base_recipe, toy_jsonl):
    base_recipe["train"]["max_steps"] = 4
    base_recipe["eval"] = [{"type": "dev_loss", "data": {"type": "jsonl", "path": toy_jsonl}, "batch_size": 4}]
    base_recipe["callbacks"] = [{"type": "evaluate", "every": 2, "at_start": False, "at_end": True}]
    cfg = recipe_from_dict(base_recipe)
    rt = train(cfg)

    # eval.jsonl check
    eval_file = Path(cfg.train.output_dir) / "eval.jsonl"
    assert eval_file.exists()
    lines = [json.loads(line) for line in eval_file.read_text().splitlines()]
    # At step 2 and at end (step 4)
    assert len(lines) == 2
    assert "dev_loss/loss" in lines[-1]
    assert torch.isfinite(torch.tensor(lines[-1]["dev_loss/loss"]))

    # rt.state.extra["eval"] check
    assert "eval" in rt.state.extra
    assert "dev_loss/loss" in rt.state.extra["eval"]


def test_math_passk_rescore_only(tmp_path):
    # Prepare dummy output_dir with raw/math500.jsonl
    eval_dir = tmp_path / "eval" / "math_passk" / "step-0"
    raw_dir = eval_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    bench_file = raw_dir / "math500.jsonl"

    row1 = {
        "id": 0,
        "gold": "42",
        "task_type": "math",
        "generations": [
            {"text": r"Therefore \boxed{42}", "finish_reason": "stop", "n_tokens": 10},
            {"text": r"Therefore \boxed{0}", "finish_reason": "stop", "n_tokens": 10},
        ],
    }
    row2 = {
        "id": 1,
        "gold": "100",
        "task_type": "math",
        "generations": [
            {"text": r"\boxed{100}", "finish_reason": "stop", "n_tokens": 8},
            {"text": r"\boxed{100}", "finish_reason": "stop", "n_tokens": 8},
        ],
    }
    with open(bench_file, "w") as f:
        f.write(json.dumps(row1) + "\n" + json.dumps(row2) + "\n")

    cfg = MathPassK.Config(
        benchmarks=["math500"],
        grader="boxed_builtin",
        ks=[1],
        rescore_only=True,
    )
    evaluator = MathPassK(cfg)

    # Mock rt with output_dir
    class MockEngine:
        is_main = True

        def all_reduce_sum(self, x):
            return x

    class MockTrainCfg:
        output_dir = str(tmp_path)

    class MockCfg:
        train = MockTrainCfg()

    class MockState:
        step = 0

    class MockRuntime:
        config = MockCfg()
        state = MockState()
        engine = MockEngine()

    rt = MockRuntime()
    metrics = evaluator.evaluate(rt)

    assert "math_passk/math500/pass@1" in metrics
    # Row 1: 1 correct out of 2. Row 2: 2 correct out of 2. Total = 3/4 = 0.75
    assert metrics["math_passk/math500/pass@1"] == 0.75
    assert metrics["math_passk/math500/any_correct"] == 1.0


def test_generation_rouge_hf(base_recipe, toy_jsonl):
    if "hf" not in ROLLOUT or ROLLOUT.meta("hf").status == "stub":
        pytest.skip("ROLLOUT hf is stub or not registered")

    base_recipe["train"]["max_steps"] = 1
    base_recipe["eval"] = [
        {
            "type": "generation_rouge",
            "data": {"type": "jsonl", "path": toy_jsonl},
            "sampling": {"max_new_tokens": 4, "temperature": 0.0},
            "engine": "hf",
            "max_samples": 4,
            "seeds": [42],
        }
    ]
    base_recipe["callbacks"] = [{"type": "evaluate", "every": "end"}]
    cfg = recipe_from_dict(base_recipe)
    rt = train(cfg)

    eval_file = Path(cfg.train.output_dir) / "eval.jsonl"
    assert eval_file.exists()
    assert "generation_rouge/rougeL" in rt.state.extra["eval"]

    raw_file = Path(cfg.train.output_dir) / "eval" / "generation_rouge" / f"step-{rt.state.step}" / "raw.jsonl"
    assert raw_file.exists()
    lines = raw_file.read_text().splitlines()
    assert len(lines) == 4


def test_cli_eval(base_recipe, toy_jsonl, tmp_path, capsys):
    base_recipe["eval"] = [{"type": "dev_loss", "data": {"type": "jsonl", "path": toy_jsonl}, "batch_size": 4}]
    recipe_path = tmp_path / "eval_recipe.yaml"
    with open(recipe_path, "w") as f:
        yaml.dump(base_recipe, f)

    code = main(["eval", "-c", str(recipe_path)])
    assert code == 0
    captured = capsys.readouterr()
    assert "dev_loss/loss" in captured.out
