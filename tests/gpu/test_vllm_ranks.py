"""T08: vllm_colocate on every rank of a 2-process run (distributed_executor_backend=external_launcher)."""

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
pytest.importorskip("vllm")

WORKER = Path(__file__).with_name("_vllm_ranks_worker.py")


def test_vllm_colocate_two_ranks(base_recipe, tmp_path):
    free = min(torch.cuda.mem_get_info(i)[0] for i in range(2))
    if free < 0.1 * torch.cuda.mem_get_info(0)[1] + (1 << 30):
        pytest.skip("not enough free GPU memory on both GPUs")
    base_recipe["models"]["student"]["dtype"] = "fp32"
    base_recipe["engine"] = {
        "type": "accelerate",
        "backend": "deepspeed",
        "mixed_precision": "no",
        "deepspeed_config": {"zero_optimization": {"stage": 2}},
        "batch_size": 8,
        "micro_batch_size": 2,
        "gradient_clipping": 1.0,
    }
    base_recipe["train"].update(batch_size=8, micro_batch_size=2)
    base_recipe["rollout"] = {
        "type": "vllm_colocate",
        "gpu_memory_utilization": 0.1,
        "dtype": "float32",
        "enforce_eager": True,
        "max_model_len": 128,
    }
    (tmp_path / "spec.json").write_text(json.dumps({"recipe": base_recipe}))
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
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-4000:]
    for rank in (0, 1):
        res = json.loads((tmp_path / f"result{rank}.json").read_text())
        assert res["diff"] < 1e-2 and res["first"] == 0, (rank, res)
