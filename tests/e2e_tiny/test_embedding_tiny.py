import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tests.conftest import _make_tokenizer
from transformers import BertConfig, BertModel

from lmkd.core.config import load_recipe, recipe_from_dict
from lmkd.core.validate import check_recipe
from lmkd.train.builder import build_runtime
from lmkd.train.trainer import Trainer


@pytest.fixture
def tiny_bert_setup(tmp_path):
    root = tmp_path / "bert_setup"
    root.mkdir()
    tok_dir = root / "student_tok"
    tok = _make_tokenizer(tok_dir)

    cfg = BertConfig(
        vocab_size=len(tok),
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    student = BertModel(cfg)
    student_dir = root / "student_model"
    student.save_pretrained(student_dir)
    tok.save_pretrained(student_dir)

    # Toy CSV dataset
    csv_file = root / "train.csv"
    lines = [",text,source\n"]
    for i in range(16):
        lines.append(f"{i},w{i % 10} w{(i * 2) % 10},src\n")
    csv_file.write_text("".join(lines))

    # Pre-computed teacher embeddings
    teacher_emb_file = root / "teacher_emb.safetensors"
    teacher_ids_file = root / "ids.json"
    D_t = 24
    emb = torch.randn(16, D_t)
    save_file({"emb": emb}, str(teacher_emb_file))
    ids = [str(i) for i in range(16)]
    teacher_ids_file.write_text(json.dumps(ids))

    cfg_t = BertConfig(
        vocab_size=len(tok),
        hidden_size=D_t,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=48,
        max_position_embeddings=128,
    )
    teacher = BertModel(cfg_t)
    teacher_dir = root / "teacher_model"
    teacher.save_pretrained(teacher_dir)
    tok.save_pretrained(teacher_dir)

    return {
        "student_path": str(student_dir),
        "teacher_model_path": str(teacher_dir),
        "csv_path": str(csv_file),
        "teacher_emb_path": str(teacher_emb_file),
        "teacher_dim": D_t,
        "run_dir": str(root / "run"),
    }


def test_talas_tiny(tiny_bert_setup):
    recipe_dict = {
        "models": {
            "student": {
                "path": tiny_bert_setup["student_path"],
                "task": "encoder",
                "dtype": "fp32",
            }
        },
        "data": {
            "train": {"type": "csv", "path": tiny_bert_setup["csv_path"]},
            "transforms": [{"type": "tokenize_text"}],
            "max_length": 32,
        },
        "teacher": {
            "provider": {
                "type": "embedding_store",
                "path": tiny_bert_setup["teacher_emb_path"],
            },
            "input": "same",
        },
        "loss": {
            "terms": [
                {"criterion": "simcse_unsupervised", "weight": 0.001, "params": {"temperature": 0.1}},
                {
                    "criterion": "cosine_to_teacher",
                    "weight": 0.75,
                    "params": {"teacher_dim": tiny_bert_setup["teacher_dim"]},
                },
                {"criterion": "lasd", "weight": 1.0},
            ]
        },
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {"lr": 1e-4},
        "train": {
            "output_dir": tiny_bert_setup["run_dir"],
            "max_steps": 3,
            "batch_size": 4,
            "micro_batch_size": 2,
            "log_every": 1,
        },
    }

    cfg = recipe_from_dict(recipe_dict)
    report = check_recipe(cfg)
    assert not report.errors, f"Recipe check failed: {report.errors}"

    rt = build_runtime(cfg)
    trainer = Trainer(rt)
    trainer.fit()

    assert rt.state.step == 3
    metrics_path = Path(tiny_bert_setup["run_dir"]) / "metrics.jsonl"
    assert metrics_path.exists()
    lines = [json.loads(l) for l in metrics_path.read_text().splitlines()]
    last_metric = lines[-1]
    assert any("simcse_unsupervised" in k for k in last_metric)
    assert any("cosine_to_teacher" in k for k in last_metric)
    assert any("lasd" in k for k in last_metric)


def test_distillcse_tiny(tiny_bert_setup):
    run_dir = Path(tiny_bert_setup["run_dir"]).parent / "distillcse_run"
    recipe_dict = {
        "models": {
            "student": {
                "path": tiny_bert_setup["student_path"],
                "task": "encoder",
                "dtype": "fp32",
            }
        },
        "data": {
            "train": {"type": "csv", "path": tiny_bert_setup["csv_path"]},
            "transforms": [{"type": "tokenize_text"}],
            "max_length": 32,
        },
        "teacher": {
            "provider": {
                "type": "embedding_store",
                "path": tiny_bert_setup["teacher_emb_path"],
            },
            "input": "same",
        },
        "loss": {
            "terms": [
                {"criterion": "simcse_unsupervised", "weight": 1.0, "params": {"temperature": 0.05, "pooler": "cls"}},
                {"criterion": "distillcse", "weight": 0.2, "params": {"tau": 0.05, "pooler": "cls"}},
            ]
        },
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {"lr": 1e-4},
        "train": {
            "output_dir": str(run_dir),
            "max_steps": 3,
            "batch_size": 4,
            "micro_batch_size": 2,
            "log_every": 1,
        },
    }

    cfg = recipe_from_dict(recipe_dict)
    report = check_recipe(cfg)
    assert not report.errors, f"Recipe check failed: {report.errors}"

    rt = build_runtime(cfg)
    trainer = Trainer(rt)
    trainer.fit()

    assert rt.state.step == 3
    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    lines = [json.loads(l) for l in metrics_path.read_text().splitlines()]
    last_metric = lines[-1]
    assert any("simcse_unsupervised" in k for k in last_metric)
    assert any("distillcse" in k for k in last_metric)


def test_jasper_2stage_tiny(tiny_bert_setup):
    base_run = Path(tiny_bert_setup["run_dir"]).parent / "jasper_2stage_run"
    stage1_dir = base_run / "stage1"
    stage2_dir = base_run / "stage2"

    # --- Stage 1: Projector alignment (student backbone frozen) ---
    stage1_dict = {
        "models": {
            "student": {
                "path": tiny_bert_setup["student_path"],
                "task": "encoder",
                "dtype": "fp32",
                "trainable": False,
            }
        },
        "data": {
            "train": {"type": "csv", "path": tiny_bert_setup["csv_path"]},
            "transforms": [{"type": "tokenize_text"}],
            "max_length": 32,
        },
        "teacher": {
            "provider": {
                "type": "embedding_store",
                "path": tiny_bert_setup["teacher_emb_path"],
            },
            "input": "same",
        },
        "loss": {
            "terms": [
                {
                    "criterion": "jasper",
                    "weight": 1.0,
                    "params": {
                        "teacher_dim": tiny_bert_setup["teacher_dim"],
                        "pooler": "cls",
                        "w_cos": 1.0,
                        "w_sim": 5.0,
                        "w_tri": 1.0,
                    },
                },
            ]
        },
        "optim": {
            "lr": 1e-4,
            "groups": [{"name": "projector", "match": {"owner": "modules"}, "lr": 1e-4}],
        },
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "train": {
            "output_dir": str(stage1_dir),
            "max_steps": 3,
            "batch_size": 4,
            "micro_batch_size": 2,
            "log_every": 1,
        },
    }

    cfg1 = recipe_from_dict(stage1_dict)
    report1 = check_recipe(cfg1)
    assert not report1.errors, f"Stage 1 recipe check failed: {report1.errors}"

    rt1 = build_runtime(cfg1)
    trainer1 = Trainer(rt1)
    trainer1.fit()

    assert rt1.state.step == 3
    stage1_modules_ckpt = stage1_dir / "checkpoint-final" / "modules.pt"
    assert stage1_modules_ckpt.exists(), "Stage 1 must save modules.pt in checkpoint-final"

    # --- Stage 2: Joint student + projector distillation with MRL ---
    stage2_dict = {
        "models": {
            "student": {
                "path": tiny_bert_setup["student_path"],
                "task": "encoder",
                "dtype": "fp32",
                "trainable": True,
            }
        },
        "data": {
            "train": {"type": "csv", "path": tiny_bert_setup["csv_path"]},
            "transforms": [{"type": "tokenize_text"}],
            "max_length": 32,
        },
        "teacher": {
            "provider": {
                "type": "embedding_store",
                "path": tiny_bert_setup["teacher_emb_path"],
            },
            "input": "same",
        },
        "loss": {
            "terms": [
                {"criterion": "simcse_unsupervised", "weight": 0.7, "params": {"temperature": 0.05, "pooler": "cls"}},
                {
                    "criterion": "jasper",
                    "weight": 0.3,
                    "params": {
                        "teacher_dim": tiny_bert_setup["teacher_dim"],
                        "pooler": "cls",
                        "projector_ckpt": str(stage1_modules_ckpt),
                        "w_cos": 0.5,
                        "w_sim": 2.0,
                        "w_tri": 0.5,
                        "mrl_dims": [16, 8],
                        "w_mrl_sim": 2.0,
                        "w_mrl_tri": 0.5,
                    },
                },
            ]
        },
        "optim": {
            "lr": 2e-5,
            "groups": [
                {"name": "projector", "match": {"owner": "modules"}, "lr": 1e-4},
                {"name": "default", "match": {}},
            ],
        },
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "train": {
            "output_dir": str(stage2_dir),
            "max_steps": 3,
            "batch_size": 4,
            "micro_batch_size": 2,
            "log_every": 1,
        },
    }

    cfg2 = recipe_from_dict(stage2_dict)
    report2 = check_recipe(cfg2)
    assert not report2.errors, f"Stage 2 recipe check failed: {report2.errors}"

    rt2 = build_runtime(cfg2)
    trainer2 = Trainer(rt2)
    trainer2.fit()

    assert rt2.state.step == 3
    metrics_path = stage2_dir / "metrics.jsonl"
    assert metrics_path.exists()
    lines = [json.loads(l) for l in metrics_path.read_text().splitlines()]
    last_metric = lines[-1]
    assert any("simcse_unsupervised" in k for k in last_metric)
    assert any("jasper" in k for k in last_metric)


def test_emo_tiny(tiny_bert_setup):
    run_dir = Path(tiny_bert_setup["run_dir"]).parent / "emo_run"
    recipe_dict = {
        "models": {
            "student": {
                "path": tiny_bert_setup["student_path"],
                "task": "encoder",
                "dtype": "fp32",
                "trainable": True,
            },
            "teacher": {
                "path": tiny_bert_setup["teacher_model_path"],
                "task": "encoder",
                "dtype": "fp32",
                "trainable": False,
            },
        },
        "data": {
            "train": {"type": "csv", "path": tiny_bert_setup["csv_path"]},
            "transforms": [{"type": "tokenize_text", "teacher": True}],
            "max_length": 32,
        },
        "teacher": {
            "provider": "hf_frozen",
            "input": "pretokenized",
        },
        "loss": {
            "terms": [
                {"criterion": "simcse_unsupervised", "weight": 0.5, "params": {"temperature": 0.07, "pooler": "cls"}},
                {
                    "criterion": "emo",
                    "weight": 0.5,
                    "params": {
                        "teacher_dim": tiny_bert_setup["teacher_dim"],
                        "w_att": 0.1,
                        "w_ot": 1.0,
                        "teacher_special": "[CLS]",
                        "student_special": "[CLS]",
                    },
                },
            ]
        },
        "optim": {
            "lr": 1e-4,
            "groups": [
                {"name": "projector", "match": {"owner": "modules"}, "lr": 1e-4},
                {"name": "default", "match": {}},
            ],
        },
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "train": {
            "output_dir": str(run_dir),
            "max_steps": 3,
            "batch_size": 4,
            "micro_batch_size": 2,
            "log_every": 1,
        },
    }

    cfg = recipe_from_dict(recipe_dict)
    report = check_recipe(cfg)
    assert not report.errors, f"Recipe check failed: {report.errors}"

    rt = build_runtime(cfg)
    trainer = Trainer(rt)
    trainer.fit()

    assert rt.state.step == 3
    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    lines = [json.loads(l) for l in metrics_path.read_text().splitlines()]
    last_metric = lines[-1]
    assert any("simcse_unsupervised" in k for k in last_metric)
    assert any("emo" in k for k in last_metric)


def test_mol_tiny(tiny_bert_setup):
    run_dir = Path(tiny_bert_setup["run_dir"]).parent / "mol_run"
    recipe_dict = {
        "models": {
            "student": {
                "path": tiny_bert_setup["student_path"],
                "task": "encoder",
                "dtype": "fp32",
                "trainable": True,
            },
            "teacher": {
                "path": tiny_bert_setup["teacher_model_path"],
                "task": "encoder",
                "dtype": "fp32",
                "trainable": False,
            },
        },
        "data": {
            "train": {"type": "csv", "path": tiny_bert_setup["csv_path"]},
            "transforms": [{"type": "tokenize_text", "teacher": True}],
            "max_length": 32,
        },
        "teacher": {
            "provider": "hf_frozen",
            "input": "pretokenized",
        },
        "loss": {
            "terms": [
                {"criterion": "simcse_unsupervised", "weight": 0.7, "params": {"temperature": 0.07, "pooler": "cls"}},
                {
                    "criterion": "mol",
                    "weight": 0.3,
                    "params": {
                        "teacher_dim": tiny_bert_setup["teacher_dim"],
                        "student_layers": [-2, -1],
                        "teacher_window": 2,
                    },
                },
            ]
        },
        "optim": {
            "lr": 1e-4,
            "groups": [
                {"name": "projector", "match": {"owner": "modules"}, "lr": 1e-4},
                {"name": "default", "match": {}},
            ],
        },
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "train": {
            "output_dir": str(run_dir),
            "max_steps": 3,
            "batch_size": 4,
            "micro_batch_size": 2,
            "log_every": 1,
        },
    }

    cfg = recipe_from_dict(recipe_dict)
    report = check_recipe(cfg)
    assert not report.errors, f"Recipe check failed: {report.errors}"

    rt = build_runtime(cfg)
    trainer = Trainer(rt)
    trainer.fit()

    assert rt.state.step == 3
    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    lines = [json.loads(l) for l in metrics_path.read_text().splitlines()]
    last_metric = lines[-1]
    assert any("simcse_unsupervised" in k for k in last_metric)
    assert any("mol" in k for k in last_metric)


def test_recipe_check():
    repo_root = Path(__file__).resolve().parents[2]
    talas_m_path = repo_root / "recipes/embedding/talas_m.yaml"
    talas_asam_path = repo_root / "recipes/embedding/talas_asam.yaml"
    distillcse_path = repo_root / "recipes/embedding/distillcse.yaml"
    jasper_stage1_path = repo_root / "recipes/embedding/jasper_stage1.yaml"
    jasper_stage2_path = repo_root / "recipes/embedding/jasper_stage2.yaml"
    jasper_path = repo_root / "recipes/embedding/jasper.yaml"
    emo_path = repo_root / "recipes/embedding/emo.yaml"
    mol_path = repo_root / "recipes/embedding/mol.yaml"

    for path in (
        talas_m_path,
        talas_asam_path,
        distillcse_path,
        jasper_stage1_path,
        jasper_stage2_path,
        jasper_path,
        emo_path,
        mol_path,
    ):
        assert path.exists(), f"Recipe file {path} missing"
        cfg = load_recipe(path)
        report = check_recipe(cfg)
        assert not report.errors, f"Validation errors for {path.name}: {report.errors}"
