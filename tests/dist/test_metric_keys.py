"""T05: metric keys are unioned across ranks before the all-reduce (2 gloo processes, CPU only)."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("accelerate")

WORKER = Path(__file__).with_name("_metric_keys_worker.py")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_metric_keys_union(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2",
           "--master_addr", "127.0.0.1", "--master_port", str(_free_port()), str(WORKER), str(tmp_path)]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail("ranks with different metric keys hung in the all-reduce")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert json.loads((tmp_path / "ok").read_text())["world_size"] == 2
    rows = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    s1, s2 = rows
    # missing keys count as 0 on the rank that did not log them; then averaged over 2 ranks
    assert s1["a"] == pytest.approx(2.0) and s1["b"] == pytest.approx(1.0) and s1["c"] == pytest.approx(2.0)
    assert s1["aux/x"] == pytest.approx(3.0)
    assert s2["a"] == pytest.approx(2.5) and "b" not in s2
