"""Unit tests for self-contained (closed-loop) Long-CoT stages: SSFT, SGL-Spectral, and P-ALIGN."""

from __future__ import annotations

import json
from pathlib import Path

from baselines.longCoT import PAlignStage, SGLSpectralStage, SSFTStage
from lmkd.core.config import load_recipe
from lmkd.core.validate import check_recipe
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl
from lmkd.pipeline.stage import StageContext


def _make_dummy_records(tmp_path: Path, filename: str = "train.jsonl") -> Path:
    data_file = tmp_path / filename
    rows = []
    for i in range(4):
        rows.append(
            {
                "id": f"q_{i}",
                "problem": f"w1 w2 w{i}",
                "response": f"w4 w5 w6 w7 w8 w9 w{i}",
                "answer": f"w{i}",
                "deepseek_thinking_trajectory": "w4 w5 w6. w7 w8 w9.",
            }
        )
    data_file.write_text("\n".join(json.dumps(r) for r in rows))
    return data_file


def test_ssft_closed_stage(tmp_path: Path, tiny_models: dict[str, str]) -> None:
    data_file = _make_dummy_records(tmp_path)
    output_dir = tmp_path / "out_ssft"
    output_art = Artifact("ssft", "ssft", output_dir)

    stage = SSFTStage(
        SSFTStage.Config(
            source={"type": "jsonl", "path": str(data_file)},
            model=tiny_models["student"],
            dtype="fp32",
            steps=2,
            batch_size=2,
            cumulative_ratio=0.7,
            chat_template=False,
            tokenize_verify=False,
            gradient_checkpointing=False,
        )
    )

    # Test provided_keys
    p_keys = stage.provided_keys(stage.cfg)
    assert "student.input_ids" in p_keys
    assert "seq.loss_mask" in p_keys
    assert "meta.selected_steps" in p_keys

    ctx = StageContext(rt=None, output=output_art)
    info = stage.run(ctx)
    assert info["n"] > 0

    # Test outputs
    assert (output_dir / "samples.jsonl").exists()
    assert (output_dir / "signals.jsonl").exists()
    samples = stage.load_samples(output_art)
    assert len(samples) > 0
    assert "seq.loss_mask" in samples[0]
    assert "meta.selected_steps" in samples[0]


def test_sgl_spectral_closed_stage(tmp_path: Path, tiny_models: dict[str, str]) -> None:
    data_file = _make_dummy_records(tmp_path)
    output_dir = tmp_path / "out_spectral"
    output_art = Artifact("spectral", "spectral", output_dir)

    stage = SGLSpectralStage(
        SGLSpectralStage.Config(
            source={"type": "jsonl", "path": str(data_file)},
            model=tiny_models["student"],
            dtype="fp32",
            device="cpu",
            energy_cutoff=0.95,
            top_p=0.95,
            chat_template=False,
            tokenize_verify=False,
        )
    )

    p_keys = stage.provided_keys(stage.cfg)
    assert "student.input_ids" in p_keys
    assert "seq.loss_mask" in p_keys

    ctx = StageContext(rt=None, output=output_art)
    info = stage.run(ctx)
    assert info["computed"] > 0

    assert (output_dir / "samples.jsonl").exists()
    assert (output_dir / "signals.jsonl").exists()
    samples = stage.load_samples(output_art)
    assert len(samples) > 0
    assert "seq.loss_mask" in samples[0]
    assert "student.input_ids" in samples[0]


def test_palign_closed_stage(tmp_path: Path, tiny_models: dict[str, str]) -> None:
    data_file = _make_dummy_records(tmp_path)
    output_dir = tmp_path / "out_palign"
    output_art = Artifact("palign", "palign", output_dir)

    stage = PAlignStage(
        PAlignStage.Config(
            source={"type": "jsonl", "path": str(data_file)},
            model=tiny_models["student"],
            engine="hf",
            judge_template="Q: {question} {reasoning_part}",
            prompt_template="Q: {question} P: {prefix}",
            max_rounds=2,
            max_new_tokens=4,
            max_generate_tokens=8,
            keep_unverified=True,
            chat_template=False,
            tokenize_verify=False,
        )
    )

    p_keys = stage.provided_keys(stage.cfg)
    assert "student.input_ids" in p_keys
    assert "seq.loss_mask" in p_keys

    ctx = StageContext(rt=None, output=output_art)
    info = stage.run(ctx)
    assert info["n"] > 0

    assert (output_dir / "records.jsonl").exists()
    assert (output_dir / "samples.jsonl").exists()
    records = read_jsonl(output_dir / "records.jsonl")
    assert "<Begin_of_Prefix>" in records[0]["output"]

    samples = stage.load_samples(output_art)
    assert len(samples) > 0
    assert "student.input_ids" in samples[0]
    assert "seq.loss_mask" in samples[0]


def test_closed_recipes_validation() -> None:
    """Verify that all refactored recipes pass static schema and dependency checks."""
    root = Path(__file__).resolve().parents[2]
    recipes = [
        root / "recipes/long_cot/ssft.yaml",
        root / "recipes/long_cot/palign.yaml",
        root / "recipes/long_cot/sgl_spectral.yaml",
    ]
    for rpath in recipes:
        assert rpath.exists(), f"Recipe file does not exist: {rpath}"
        cfg = load_recipe(rpath)
        report = check_recipe(cfg)
        assert report.ok, f"Recipe {rpath} failed validation: {report.errors}"
