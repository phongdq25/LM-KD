from types import SimpleNamespace

import torch

from lmkd.train.callbacks import ConsoleLogger


def test_console_logger_reduce_loss_vs_metrics():
    class FakeEngine:
        world_size = 2
        device = torch.device("cpu")

        def all_gather_object(self, obj):
            return [["loss/total", "loss/nll/nll", "metric/acc"], ["loss/total", "metric/acc"]]

        def all_reduce_sum(self, tensor):
            # Simulate sum across 2 ranks where each rank has equal value
            return tensor * 2.0

    eng = FakeEngine()
    rt = SimpleNamespace(engine=eng)
    logger = ConsoleLogger(ConsoleLogger.Config())
    avg = {"loss/total": 1.5, "loss/nll/nll": 1.0, "metric/acc": 0.8}
    res = logger._reduce(rt, avg)

    # Loss keys are already normalized by global Z; all_reduce_sum gives the true total loss
    assert res["loss/total"] == 3.0
    assert res["loss/nll/nll"] == 2.0
    # Non-loss metrics are averaged across ranks: (0.8 * 2.0) / 2 = 0.8
    assert res["metric/acc"] == 0.8
