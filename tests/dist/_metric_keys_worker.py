"""torchrun worker for test_metric_keys.py: real ``accelerate`` engine collectives (gloo, CPU) +
ConsoleLogger._reduce when every rank logs a DIFFERENT key set. No training backend is built."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from lmkd.core.state import TrainState
from lmkd.train.callbacks import ConsoleLogger
from lmkd.train.engine import AccelerateEngine


def main(out_dir: str) -> None:
    eng = AccelerateEngine(AccelerateEngine.Config(mixed_precision="no"))
    assert eng.world_size == 2 and eng.device.type == "cpu", (eng.world_size, eng.device)
    assert eng.all_gather_object({"r": eng.rank}) == [{"r": 0}, {"r": 1}]
    x = torch.tensor([1.0 + eng.rank])
    y = eng.all_reduce_sum(x)
    assert y.item() == 3.0 and x.item() == 1.0 + eng.rank      # input is not modified

    rt = SimpleNamespace(engine=eng, state=TrainState(total_steps=2),
                         config=SimpleNamespace(train=SimpleNamespace(output_dir=out_dir, log_every=1)))
    logger = ConsoleLogger(ConsoleLogger.Config())
    logger.on_train_begin(rt)
    # step 1: different keys per rank (+ an aux metric logged on rank 1 only)
    metrics = {"a": 1.0, "b": 2.0} if eng.rank == 0 else {"a": 3.0, "c": 4.0}
    if eng.rank == 1:
        rt.state.log("aux/x", 6.0)
    rt.state.step = 1
    logger.on_step_end(rt, SimpleNamespace(metrics=metrics))
    # step 2: rank 1 logs nothing at all
    rt.state.step, rt.state.done = 2, True
    logger.on_step_end(rt, SimpleNamespace(metrics={"a": 5.0} if eng.rank == 0 else {}))
    eng.barrier()
    if eng.rank == 0:
        (Path(out_dir) / "ok").write_text(json.dumps({"world_size": eng.world_size}))


if __name__ == "__main__":
    main(sys.argv[1])
