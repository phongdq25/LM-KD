"""Tests for Phase 3: lmkd.quick API and CLI commands explain & try."""

import lmkd
from lmkd.cli import main


def test_quick_api_builds_runtime(tiny_models, toy_jsonl):
    # Test building a runtime without YAML via lmkd.quick
    rt = lmkd.quick(
        student=tiny_models["student"],
        teacher=tiny_models["teacher"],
        data=str(toy_jsonl),
        loss="nll",
        max_steps=2,
        batch_size=2,
        micro_batch_size=1,
        strict=False,
    )
    assert rt is not None
    assert rt.student_model is not None
    assert rt.config.train.max_steps == 2

    # Step executes cleanly
    rt.begin()
    out = rt.train_step()
    assert out is not None
    assert not out.skipped
    rt.end()


def test_cli_explain(capsys):
    ret = main(["explain", "-c", "recipes/same_tokenizer/fkl.yaml"])
    assert ret == 0
    captured = capsys.readouterr().out
    assert "Recipe:" in captured
    assert "Student:" in captured
    assert "Loss Terms:" in captured


def test_cli_try(capsys):
    ret = main(["try", "nll"])
    assert ret == 0
    captured = capsys.readouterr().out
    assert "Testing loss 'nll'" in captured
    assert "Success! Loss terms produced:" in captured
