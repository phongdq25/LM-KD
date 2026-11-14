"""Actual two-CUDA-rank update, accumulation, auxiliary parameters and deterministic resume."""
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import lmkd


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.gpu
@pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="FSDP2 regression requires two CUDA devices")
def test_fsdp2_update_and_resume(tiny_models, toy_jsonl, tmp_path, optimizer):
    worker = Path(__file__).with_name("_fsdp2_worker.py")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
           "--master_addr=127.0.0.1", f"--master_port={_free_port()}",
           str(worker), tiny_models["student"], toy_jsonl, str(tmp_path), optimizer]
    result = subprocess.run(cmd, env={**os.environ, "OMP_NUM_THREADS": "1", "NVIDIA_TF32_OVERRIDE": "0"},
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    assert (tmp_path / "verified").exists()

    rt = lmkd.build({
        "imports": [str(worker)],
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "models": {"student": {"path": tiny_models["student"], "dtype": "fp32"}},
        "data": {"train": {"type": "jsonl", "path": toy_jsonl},
                 "transforms": [{"type": "tokenize_sft", "use_chat_template": False}]},
        "loss": {"terms": [{"criterion": "nll"}, {"criterion": "fsdp_aux_test"}]},
        "optim": {"lr": 0.05 if optimizer == "sgd" else 0.001,
                  "optimizers": {"default": {"type": optimizer}, "aux": {"type": "adamw"}},
                  "groups": [{"name": "projector", "match": {"group": "projector"}, "optimizer": "aux"},
                             {"name": "default", "match": {}}]},
        "train": {"output_dir": str(tmp_path / "local"), "max_steps": 1, "batch_size": 4, "micro_batch_size": 1},
        "grad": {"clip_norm": None if optimizer == "sgd" else 1.0},
    })
    for module in rt.student_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    rt.begin()
    rt.train_step()
    distributed = torch.load(tmp_path / "first_step.pt", weights_only=True)
    for side, module in (("student", rt.student_model), ("aux", rt.modules)):
        for name, value in module.state_dict().items():
            torch.testing.assert_close(distributed[side][name], value, rtol=1e-4, atol=2e-6)
