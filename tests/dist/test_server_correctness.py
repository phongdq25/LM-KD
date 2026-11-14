"""Native server rollout uses matching collectives on two CPU ranks."""
import os
import socket
import subprocess
import sys
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_server_two_rank_errors_and_sync(tmp_path):
    worker = Path(__file__).with_name("_server_correctness_worker.py")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "AUDIT_SERVER_OUTPUT": str(tmp_path)}
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
                             "--master_addr=127.0.0.1", f"--master_port={_free_port()}",
                             str(worker)], env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert (tmp_path / "rank0.json").read_text() == (tmp_path / "rank1.json").read_text()
