"""T05: DeepSpeed ZeRO engine (``engine: {type: accelerate, backend: deepspeed}``) vs local.

Each test launches ``torchrun --nproc_per_node 2`` on ``_deepspeed_worker.py``; the local reference
runs in-process on CPU with the same global batch (fp32, no dropout). Needs >= 2 CUDA GPUs.
"""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs >= 2 CUDA GPUs"),
]
deepspeed = pytest.importorskip("deepspeed")

sys.path.insert(0, str(Path(__file__).parent))
import _ds_plugins  # noqa: E402,F401

from lmkd.core.config import recipe_from_dict  # noqa: E402
from lmkd.train.trainer import Trainer  # noqa: E402

WORKER = Path(__file__).with_name("_deepspeed_worker.py")
STEPS = 3


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _torchrun(spec: dict, d: Path) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    (d / "spec.json").write_text(json.dumps(spec))
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(_free_port()),
        str(WORKER),
        str(d / "spec.json"),
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        "OMP_NUM_THREADS": "1",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    rank_lines = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if l.startswith("[rank0]"))
    assert r.returncode == 0, rank_lines[-6000:] or (r.stdout[-3000:] + r.stderr[-3000:])
    return json.loads((d / "result.json").read_text())


@pytest.fixture(scope="module")
def no_dropout_models(tiny_models, tmp_path_factory):
    root = tmp_path_factory.mktemp("nodrop")
    out = {}
    for side in ("student", "teacher"):
        m = AutoModelForCausalLM.from_pretrained(tiny_models[side])
        m.config.update({"resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0})
        m.save_pretrained(root / side)
        AutoTokenizer.from_pretrained(tiny_models[side]).save_pretrained(root / side)
        out[side] = str(root / side)
    return out


def _recipe(base: dict, models: dict, out: Path) -> dict:
    r = copy.deepcopy(base)
    r["models"] = {
        "student": {"path": models["student"], "dtype": "fp32"},
        "teacher": {"path": models["teacher"], "dtype": "fp32"},
    }
    r["teacher"] = {"provider": "hf_frozen", "input": "same"}
    r["loss"] = {
        "terms": [
            {"criterion": "nll"},
            {"criterion": "token_kd", "alias": "kd", "weight": 0.5},
            {"criterion": "_t05_proj", "alias": "proj", "weight": 0.1},
        ]
    }
    r["optim"] = {"lr": 1e-3, "schedulers": [{"type": "linear", "warmup_ratio": 0.5}]}
    r["grad"] = {"clip_norm": 1.0}
    r["train"].update(output_dir=str(out), max_steps=STEPS, batch_size=8, micro_batch_size=2, log_every=1)
    r["callbacks"] = [{"type": "checkpoint", "at_end": False}]
    return r


def _ds_recipe(base, models, out, stage: int, precision: str = "no", ds_extra: dict | None = None) -> dict:
    r = _recipe(base, models, out)
    r["engine"] = {
        "type": "accelerate",
        "backend": "deepspeed",
        "mixed_precision": precision,
        # small buckets: the tiny model must not reserve DS's ~2 GiB default buffers
        "deepspeed_config": {
            "zero_optimization": {
                "stage": stage,
                "reduce_bucket_size": 500_000,
                "allgather_bucket_size": 500_000,
                "stage3_prefetch_bucket_size": 500_000,
            },
            **(ds_extra or {}),
        },
        "batch_size": 8,
        "micro_batch_size": 2,
        "gradient_clipping": 1.0,
    }
    return r


@pytest.fixture(scope="module")
def local_params(base_recipe_module, no_dropout_models, tmp_path_factory):
    out = tmp_path_factory.mktemp("local")
    rt = Trainer.from_config(recipe_from_dict(_recipe(base_recipe_module, no_dropout_models, out))).fit()
    p = {f"student.{n}": t.detach().float().cpu() for n, t in rt.student_model.state_dict().items()}
    p.update({n: t.detach().float().cpu() for n, t in rt.modules.state_dict().items()})
    return p


@pytest.fixture(scope="module")
def base_recipe_module(tiny_models, toy_jsonl, tmp_path_factory):
    return {
        "models": {"student": {"path": tiny_models["student"], "dtype": "fp32"}},
        "data": {
            "train": {"type": "jsonl", "path": toy_jsonl},
            "transforms": [{"type": "tokenize_sft", "use_chat_template": False}],
            "max_length": 64,
        },
        "loss": {"terms": [{"criterion": "nll"}]},
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {"lr": 1e-3},
        "train": {
            "output_dir": str(tmp_path_factory.mktemp("unused")),
            "max_steps": 4,
            "batch_size": 8,
            "micro_batch_size": 2,
            "log_every": 1,
        },
    }


def _run_train(base, models, d: Path, stage: int, **kw) -> tuple[dict, dict]:
    spec = {"mode": "train", "expect_ga": 2, "recipe": _ds_recipe(base, models, d / "run", stage), **kw}
    res = _torchrun(spec, d)
    return res, torch.load(d / "params.pt")


def _assert_matches(local: dict, ds: dict, atol: float = 1e-5) -> None:
    assert set(ds) == set(local), set(ds) ^ set(local)
    for k, v in local.items():
        assert torch.allclose(ds[k], v, atol=atol), (k, (ds[k] - v).abs().max().item())


@pytest.fixture(scope="module")
def zero2_run(base_recipe_module, no_dropout_models, tmp_path_factory):
    return _run_train(base_recipe_module, no_dropout_models, tmp_path_factory.mktemp("z2"), 2)


def test_zero2_matches_local(zero2_run, local_params):
    res, ds = zero2_run
    assert res["world"] == 2 and res["zero_stage"] == 2
    _assert_matches(local_params, ds)


def test_aux_module_grads_synced(zero2_run, local_params, no_dropout_models):
    res, ds = zero2_run
    assert res["proj_same_across_ranks"]
    proj = [k for k in ds if k.startswith("mods.")]
    assert proj and all(torch.allclose(ds[k], local_params[k], atol=1e-5) for k in proj)
    # (tied) lm_head is trained through DeepSpeed: it moved, and exactly like local
    init = AutoModelForCausalLM.from_pretrained(no_dropout_models["student"]).lm_head.weight
    head = ds["student.lm_head.weight"]
    assert (head - init).abs().max() > 1e-4
    assert torch.allclose(head, local_params["student.lm_head.weight"], atol=1e-5)


def test_zero3_save_model(base_recipe_module, no_dropout_models, local_params, tmp_path):
    res, ds = _run_train(base_recipe_module, no_dropout_models, tmp_path, 3, save_model=True)
    assert res["zero_stage"] == 3 and res["proj_same_across_ranks"]
    _assert_matches(local_params, ds)
    saved = AutoModelForCausalLM.from_pretrained(tmp_path / "saved")
    for n, t in saved.state_dict().items():
        assert torch.allclose(t, local_params[f"student.{n}"], atol=1e-5), n
    assert (tmp_path / "saved" / "tokenizer.json").exists()


def _sgd(r: dict) -> dict:
    # SGD without clipping is linear in the gradient: a missing x n_micro (GA) compensation would
    # halve every update here (AdamW + clipping above is nearly scale-invariant and would hide it).
    r["optim"] = {"lr": 0.5, "optimizers": {"default": {"type": "sgd"}}, "schedulers": [{"type": "constant"}]}
    r["grad"] = {"clip_norm": None}
    if r["engine"]["type"] == "accelerate":
        r["engine"]["gradient_clipping"] = None
    return r


def test_zero2_gradient_scale_and_resume(base_recipe_module, no_dropout_models, tmp_path):
    base, models = base_recipe_module, no_dropout_models
    rt = Trainer.from_config(recipe_from_dict(_sgd(_recipe(base, models, tmp_path / "local")))).fit()
    local = {f"student.{n}": t.detach().float().cpu() for n, t in rt.student_model.state_dict().items()}
    local.update({n: t.detach().float().cpu() for n, t in rt.modules.state_dict().items()})

    r = _sgd(_ds_recipe(base, models, tmp_path / "a" / "run", 2))
    r["callbacks"] = [{"type": "checkpoint", "every": 2, "at_end": False, "keep_last": None}]
    _torchrun({"mode": "train", "expect_ga": 2, "recipe": r}, tmp_path / "a")
    _assert_matches(local, torch.load(tmp_path / "a" / "params.pt"))
    ck = tmp_path / "a" / "run" / "checkpoint-2"
    assert (ck / "COMPLETE").exists() and (ck / "optim" / "ds").is_dir()
    assert (ck / "rng" / "rank0.pt").exists() and (ck / "rng" / "rank1.pt").exists()

    r_b = copy.deepcopy(r)
    r_b["train"].update(output_dir=str(tmp_path / "b" / "run"), resume_from=str(ck))
    r_b["callbacks"] = [{"type": "checkpoint", "at_end": False}]
    _torchrun({"mode": "train", "expect_ga": 2, "recipe": r_b}, tmp_path / "b")  # steps 3..3 only
    _assert_matches(local, torch.load(tmp_path / "b" / "params.pt"))


def test_fp16_overflow_skip(base_recipe_module, no_dropout_models, tmp_path, stage=2):
    # loss scale 2^32 overflows fp16 on the first backward: DS skips the step -> no scheduler step
    r = _ds_recipe(
        base_recipe_module,
        no_dropout_models,
        tmp_path / "run",
        stage,
        precision="fp16",
        ds_extra={"fp16": {"initial_scale_power": 32, "loss_scale_window": 1000}},
    )
    res = _torchrun({"mode": "fp16_skip", "recipe": r}, tmp_path)
    assert res["loss_finite"] and res["skipped"]
    assert res["lr_same"] and res["epoch_same"] and res["params_same"]


def test_zero0_matches_local(base_recipe_module, no_dropout_models, local_params, tmp_path):
    # stage 0 = plain DeepSpeed data parallel (no ZeRO optimizer; ds_engine.optimizer = client in fp32)
    res, ds = _run_train(base_recipe_module, no_dropout_models, tmp_path, 0)
    assert res["zero_stage"] == 0 and res["proj_same_across_ranks"]
    _assert_matches(local_params, ds)


def test_zero0_gradient_scale(base_recipe_module, no_dropout_models, tmp_path):
    base, models = base_recipe_module, no_dropout_models
    rt = Trainer.from_config(recipe_from_dict(_sgd(_recipe(base, models, tmp_path / "local")))).fit()
    local = {f"student.{n}": t.detach().float().cpu() for n, t in rt.student_model.state_dict().items()}
    local.update({n: t.detach().float().cpu() for n, t in rt.modules.state_dict().items()})
    r = _sgd(_ds_recipe(base, models, tmp_path / "a" / "run", 0))
    r["callbacks"] = [{"type": "checkpoint", "every": 2, "at_end": False, "keep_last": None}]
    _torchrun({"mode": "train", "expect_ga": 2, "recipe": r}, tmp_path / "a")
    _assert_matches(local, torch.load(tmp_path / "a" / "params.pt"))
    assert (tmp_path / "a" / "run" / "checkpoint-2" / "COMPLETE").exists()
