"""Test selective evaluation: eval dev every epoch, eval test once on best ckpt at train end."""

import json

from lmkd.core.config import recipe_from_dict
from lmkd.core.registry import EVALUATORS
from lmkd.eval.base import Evaluator
from lmkd.train.checkpointing import find_best_checkpoint
from lmkd.train.trainer import Trainer


@EVALUATORS.register("_test_dev_eval")
class _TestDevEval(Evaluator):
    class Config(Evaluator.Config):
        when: str = "epoch"

    def evaluate(self, rt):
        step = getattr(rt.state, "step", 0)
        # loss decreases then increases (step 1: 3.0, step 2: 1.0, step 3: 2.0)
        losses = {1: 3.0, 2: 1.0, 3: 2.0}
        val = losses.get(step, 4.0)
        return {"dev_loss/loss": val}


@EVALUATORS.register("_test_heavy_test_eval")
class _TestHeavyTestEval(Evaluator):
    class Config(Evaluator.Config):
        when: str = "end"

    def evaluate(self, rt):
        # Record the model weight or indicator to confirm it evaluated on best ckpt
        p_val = float(next(rt.student_model.parameters()).data.mean())
        return {"test_bench/accuracy": 0.95, "test_bench/weight_mean": p_val}


def test_selective_eval_flow(base_recipe, tmp_path):
    out = tmp_path / "run"
    base_recipe["train"].update(output_dir=str(out), max_steps=3, log_every=1)
    base_recipe["callbacks"] = [
        {"type": "evaluate", "every": 1, "load_best_at_end": True},
        {"type": "checkpoint", "every": 1, "keep_best": 1, "metric": "dev_loss/loss", "mode": "min"},
    ]
    base_recipe["eval"] = [
        {"type": "_test_dev_eval"},
        {"type": "_test_heavy_test_eval"},
    ]

    trainer = Trainer.from_config(recipe_from_dict(base_recipe)).fit()

    # Verify best checkpoint discovery
    best = find_best_checkpoint(out)
    assert best is not None
    assert best.name == "checkpoint-2"  # step 2 had loss 1.0 (best)
    assert (out / "best_checkpoint.txt").read_text().strip() == "checkpoint-2"

    # Verify eval.jsonl records
    lines = [json.loads(l) for l in (out / "eval.jsonl").read_text().splitlines()]
    # steps 1, 2, 3 should have dev_loss and phase="epoch"
    epoch_evals = [l for l in lines if l.get("phase") == "epoch"]
    assert len(epoch_evals) == 3
    for l in epoch_evals:
        assert "dev_loss/loss" in l
        assert "test_bench/accuracy" not in l  # heavy test was skipped during steps!

    # final eval should have phase="end" and contain test_bench metrics
    end_evals = [l for l in lines if l.get("phase") == "end"]
    assert len(end_evals) == 1
    assert "test_bench/accuracy" in end_evals[0]


def test_cli_multi_config_and_ckpt_resolution(base_recipe, tmp_path, monkeypatch, capsys):
    import yaml

    from lmkd.cli import main

    out = tmp_path / "run"
    base_recipe["train"].update(output_dir=str(out), max_steps=2, log_every=1)
    base_recipe["callbacks"] = [
        {"type": "evaluate", "every": 1},
        {"type": "checkpoint", "every": 1, "keep_best": 1, "metric": "dev_loss/loss", "mode": "min"},
    ]
    base_recipe["eval"] = [{"type": "_test_dev_eval"}]
    Trainer.from_config(recipe_from_dict(base_recipe)).fit()

    train_yaml = tmp_path / "train.yaml"
    eval_yaml = tmp_path / "eval.yaml"
    with open(train_yaml, "w") as f:
        yaml.safe_dump(base_recipe, f)
    with open(eval_yaml, "w") as f:
        yaml.safe_dump({"eval": [{"type": "_test_heavy_test_eval"}]}, f)

    # Run lmkd eval with multiple -c and --ckpt pointing to output dir (auto resolves best ckpt)
    ret = main(["eval", "-c", str(train_yaml), "-c", str(eval_yaml), "--ckpt", str(out)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "test_bench/accuracy" in captured.out

