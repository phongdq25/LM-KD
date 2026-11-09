"""Executable baseline catalogue and recipe resolution regressions."""

from pathlib import Path

import pytest

from lmkd.reproduce import BASELINES, ROOT, resolve_jobs, validate_jobs


def test_catalogue_covers_six_domains_and_named_gaps():
    assert {b.domain for b in BASELINES.values()} == {
        "same_tokenizer",
        "cross_tokenizer",
        "long_cot",
        "self_distill",
        "preference",
        "embedding",
        "self_play",
    }
    for name in (
        "vpd",
        "ppd",
        "dckd",
        "tvkd",
        "adpa",
        "ctpd",
        "dp3o",
        "pbsd",
        "sgl_spectral",
        "residualkd",
    ):
        assert name in BASELINES
    assert all((ROOT / b.recipe).is_file() for b in BASELINES.values())


def test_every_catalogue_recipe_is_valid(tmp_path):
    for name in BASELINES:
        validate_jobs(name, resolve_jobs(name, tmp_path / name))


def test_residual_jobs_propagate_models_data_and_projector(tmp_path):
    jobs = resolve_jobs("residualkd", tmp_path, student="S", teacher="T", data="train.jsonl")
    assert len(jobs) == 2
    first, second = jobs
    assert first.models.student.path == "T"
    assert second.models.student.path == "S"
    assert second.models.teacher.path == "T"
    assert first.data.train["path"] == second.data.train["path"] == str(Path("train.jsonl").resolve())
    residual = next(t for t in second.loss.terms if t.criterion == "residualkd")
    assert residual.params["projector_ckpt"] == str(tmp_path / "stage1/checkpoint-final/modules.pt")


def test_jasper_jobs_propagate_2_stage_pipeline(tmp_path):
    jobs = resolve_jobs("jasper", tmp_path, student="S", embedding_teacher="ET", data="train.csv")
    assert len(jobs) == 2
    first, second = jobs
    # Stage 1: Student backbone is frozen, trains projector
    assert first.models.student.path == "S"
    assert not first.models.student.trainable
    assert first.data.train["path"] == second.data.train["path"] == str(Path("train.csv").resolve())
    # Stage 2: Student backbone trainable, loads projector_ckpt from Stage 1
    assert second.models.student.path == "S"
    assert second.models.student.trainable
    jasper_term = next(t for t in second.loss.terms if t.criterion == "jasper")
    assert jasper_term.params["projector_ckpt"] == str(tmp_path / "stage1/checkpoint-final/modules.pt")


def test_residual_phase1_respects_final_teacher_overrides(tmp_path):
    first, second = resolve_jobs(
        "residualkd",
        tmp_path,
        overrides=["models.teacher.path=final-teacher", "models.teacher.tokenizer=final-tokenizer"],
    )
    assert first.models.student.path == second.models.teacher.path == "final-teacher"
    assert first.models.student.tokenizer == "final-tokenizer"
    assert not first.models.student.trainable
    with pytest.raises(ValueError, match="distinct output"):
        resolve_jobs("residualkd", tmp_path, overrides=["train.output_dir=outputs/collision"])


def test_stage_models_and_sources_are_overridden_without_changing_artifact_sources(tmp_path):
    (cfg,) = resolve_jobs("dckd", tmp_path, student="S", teacher="T", data="pairs.jsonl")
    assert cfg.pipeline[0]["source"]["path"] == str(Path("pairs.jsonl").resolve())
    assert cfg.pipeline[1]["model"] == "T"
    (cfg,) = resolve_jobs("vpd", tmp_path, teacher="T", data="pairs.jsonl")
    assert cfg.pipeline[2]["model"] == "T"
    assert cfg.pipeline[3]["source"]["node"] == "scores"


def test_ctpd_uses_reference_for_online_scoring(tmp_path):
    (cfg,) = resolve_jobs("ctpd", tmp_path, teacher="T", reference="R")
    assert cfg.models.teacher.path == "R"
    assert cfg.pipeline[1]["model"] == "T"
    assert cfg.pipeline[1]["reference_model"] == "R"


def test_dckd_sft_only_supervises_chosen_response(tmp_path):
    (cfg,) = resolve_jobs("dckd", tmp_path)
    assert "chosen_nll" in [term.criterion for term in cfg.loss.terms]
    assert "nll" not in [term.criterion for term in cfg.loss.terms]


def test_distillm_initial_dev_loss_is_measured(tmp_path):
    (cfg,) = resolve_jobs("distillm", tmp_path)
    evaluator = next(callback for callback in cfg.callbacks if callback["type"] == "evaluate")
    assert evaluator["at_start"] is True


def test_unknown_baseline_fails_explicitly(tmp_path):
    with pytest.raises(ValueError, match="Unknown baseline"):
        resolve_jobs("orpo", tmp_path)


def test_longcot_model_override_reaches_offline_capture(tmp_path):
    (cfg,) = resolve_jobs("sgl_spectral", tmp_path, student="new-model")
    stage = cfg.pipeline[0]
    assert stage["stage"] == "sgl_spectral"
    assert stage["model"] == "new-model"


def test_embedding_teacher_receives_original_text(tmp_path):
    from lmkd.core.registry import STAGES, TRANSFORMS
    from lmkd.data.context import DataContext

    (cfg,) = resolve_jobs("talas_m", tmp_path)
    transform = TRANSFORMS.build(cfg.pipeline[0]["transforms"][0])

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2]}

    sample = transform(
        {"text": "A real sentence.", "meta.id": "row"}, DataContext(student_tokenizer=Tokenizer(), max_length=32)
    )
    teacher = STAGES.build(
        {key: value for key, value in cfg.pipeline[1].items() if key not in {"stage", "name", "inputs"}}
        | {"type": "embed_teacher"}
    )
    assert teacher._ids_and_texts([sample]) == (["row"], ["A real sentence."])


def test_failed_launch_records_failure(tmp_path, monkeypatch):
    import json
    import subprocess

    from lmkd.reproduce import execute_plan, write_plan

    jobs = resolve_jobs("sft", tmp_path)
    plan = write_plan("sft", jobs, tmp_path)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(7, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        execute_plan(plan, tmp_path)
    result = json.loads((tmp_path / "reproduction.json").read_text())
    assert result["status"] == "failed" and result["returncode"] == 7


def test_reproduce_cli_set_action_extend():
    from lmkd.reproduce import main

    res = main(["sft", "--check", "--set", "train.seed=123", "--set", "train.log_every=5"])
    assert res == 0 or res is None
