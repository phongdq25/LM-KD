"""T08: ``iter_student_weights`` under DeepSpeed ZeRO-3 streams the FULL tensors (2 GPUs, torchrun)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs >= 2 CUDA GPUs"),
]
pytest.importorskip("deepspeed")

WORKER = Path(__file__).with_name("_weight_sync_worker.py")


def test_zero3_iter_student_weights_are_full(base_recipe, tiny_models, tmp_path):
    base_recipe["models"]["student"]["dtype"] = "fp32"
    base_recipe["engine"] = {
        "type": "accelerate",
        "backend": "deepspeed",
        "mixed_precision": "no",
        "deepspeed_config": {
            "zero_optimization": {
                "stage": 3,
                "reduce_bucket_size": 500_000,
                "allgather_bucket_size": 500_000,
                "stage3_prefetch_bucket_size": 500_000,
            }
        },
        "batch_size": 8,
        "micro_batch_size": 2,
        "gradient_clipping": 1.0,
    }
    base_recipe["train"].update(batch_size=8, micro_batch_size=2)
    spec = {"recipe": base_recipe, "student": tiny_models["student"]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(port),
        str(WORKER),
        str(tmp_path / "spec.json"),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"), "OMP_NUM_THREADS": "1"}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["ok"] and all(res["ok"].values()), [k for k, v in res["ok"].items() if not v]
