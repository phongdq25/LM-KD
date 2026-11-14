"""Unit and reference tests for GradMag and the multi-objective step."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from tests.parity._ref.gradmag_ref import approximate_gradmag_weights

from lmkd.losses.aggregator import GradMag, Norms, WeightedOutput, WeightedSum
from lmkd.losses.base import LossOutput, LossTerm
from lmkd.train.multitask import gradmag_weights, task_grads
from lmkd.train.step_strategies import MultiObjectiveGradStep, StandardStep


@pytest.mark.parametrize("mode", ["approx_gradmag", "approx_gradmag_preserve_mag"])
def test_weights_match_ref(mode):
    torch.manual_seed(14)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    x = torch.randn(5, 3)
    y = model(x)
    losses = [y[:, 0].square().sum(), (y[:, 1] - 2).square().sum()]
    params = list(model[-1].parameters())
    got_w, got_n = gradmag_weights(task_grads(losses, params), mode)
    ref_w, ref_n = approximate_gradmag_weights(losses, params, mode)
    torch.testing.assert_close(got_w, ref_w, atol=1e-6, rtol=0)
    torch.testing.assert_close(got_n, ref_n, atol=1e-6, rtol=0)


def test_requires_step():
    agg = GradMag(GradMag.Config())
    with pytest.raises(ValueError, match="multi_objective_grad"):
        agg.setup(SimpleNamespace(config=SimpleNamespace(step="standard")))
    outs = [WeightedOutput("a", 1.0, LossOutput({"x": LossTerm(torch.tensor(2.0), "mean")}))]
    with pytest.raises(RuntimeError, match="multi_objective_grad"):
        agg.combine(outs, Norms(1, 1, 1))
    with torch.no_grad():
        total, mets = agg.combine(outs, Norms(1, 1, 1))
        torch.testing.assert_close(total, torch.tensor(2.0))


def test_task_grads_unused_param_is_zero():
    p = torch.nn.Parameter(torch.tensor(2.0))
    q = torch.nn.Parameter(torch.tensor(3.0))
    grads = task_grads([p.square()], [p, q])
    torch.testing.assert_close(grads[0][0], torch.tensor(4.0))
    torch.testing.assert_close(grads[0][1], torch.tensor(0.0))


def test_gradmag_combines_after_normalization():
    agg = GradMag(GradMag.Config())
    outs = [
        WeightedOutput("a", 1.0, LossOutput({"x": LossTerm(torch.tensor(8.0), "token_sum")})),
        WeightedOutput("b", 2.0, LossOutput({"y": LossTerm(torch.tensor(6.0), "seq_sum")})),
    ]
    agg.set_weights(torch.tensor([0.25, 0.75]), torch.tensor([2.0, 4.0]))
    total, mets = agg.combine(outs, Norms(4, 3, 1))
    torch.testing.assert_close(total, torch.tensor(3.5))
    assert "gradmag/weight_a" in mets and "gradmag/norm_b" in mets
    with pytest.raises(RuntimeError, match="multi_objective_grad"):
        agg.combine(outs, Norms(4, 3, 1))


def test_step_grad_equals_weighted_sum():
    param = torch.nn.Parameter(torch.tensor(2.0))

    class Engine:
        world_size = 1

        def no_sync(self, module, enabled):
            return torch.no_grad() if False else torch.enable_grad()

        def autocast(self):
            return torch.enable_grad()

        def backward(self, loss):
            loss.backward()

    class Algorithm:
        def loss(self, micro, rt):
            return [
                WeightedOutput("a", 1, LossOutput({"x": LossTerm(param.square(), "mean")})),
                WeightedOutput("b", 1, LossOutput({"x": LossTerm((param - 4).square(), "mean")})),
            ]

    rt = SimpleNamespace(
        engine=Engine(),
        student_module=None,
        algorithm=Algorithm(),
        aggregator=GradMag(GradMag.Config()),
        state=SimpleNamespace(micro_step=0),
    )
    step = MultiObjectiveGradStep(MultiObjectiveGradStep.Config())
    step.rt, step.params = rt, [param]
    step.accumulate([None], Norms(1, 1, 1))
    x = torch.tensor(2.0, requires_grad=True)
    losses = [x.square(), (x - 4).square()]
    weights, _ = approximate_gradmag_weights(losses, [x], "approx_gradmag_preserve_mag")
    expected = torch.autograd.grad((weights.detach() * torch.stack(losses)).sum(), x)[0]
    torch.testing.assert_close(param.grad, expected, atol=1e-6, rtol=0)


def test_sum_combine_equals_standard():
    param = torch.nn.Parameter(torch.tensor(2.0))

    class Algorithm:
        def loss(self, micro, rt):
            return [
                WeightedOutput("a", 1, LossOutput({"x": LossTerm(param.square(), "mean")})),
                WeightedOutput("b", 1, LossOutput({"x": LossTerm((param - 4).square(), "mean")})),
            ]

    engine = SimpleNamespace(
        world_size=1,
        no_sync=lambda *a, **kw: nullcontext(),
        autocast=lambda: nullcontext(),
        backward=lambda loss: loss.backward(),
    )
    rt = SimpleNamespace(
        engine=engine,
        student_module=None,
        algorithm=Algorithm(),
        aggregator=WeightedSum(WeightedSum.Config()),
        state=SimpleNamespace(micro_step=0),
    )
    standard = StandardStep(StandardStep.Config())
    standard.rt = rt
    standard.accumulate([None], Norms(1, 1, 1))
    expected = param.grad.clone()
    param.grad = None
    multi = MultiObjectiveGradStep(MultiObjectiveGradStep.Config(combine="sum"))
    multi.rt = rt
    multi.accumulate([None], Norms(1, 1, 1))
    torch.testing.assert_close(param.grad, expected, atol=1e-6, rtol=0)


def test_multiobjective_and_pcgrad_normalize_metric_across_micros():
    """Verify MultiObjectiveGradStep (both gradmag and pcgrad) divides metric/* by len(micros)."""
    param = torch.nn.Parameter(torch.tensor(2.0))

    class Algorithm:
        def loss(self, micro, rt):
            return [
                WeightedOutput(
                    "a",
                    1.0,
                    LossOutput({"loss": LossTerm(param.square(), "mean")}, metrics={"aux": torch.tensor(10.0)}),
                ),
                WeightedOutput("b", 1.0, LossOutput({"loss": LossTerm((param - 4).square(), "mean")})),
            ]

    engine = SimpleNamespace(
        world_size=1,
        no_sync=lambda *a, **kw: nullcontext(),
        autocast=lambda: nullcontext(),
        backward=lambda loss: loss.backward(),
    )

    # 1. GradMag combine
    agg = GradMag(GradMag.Config())
    rt_gm = SimpleNamespace(
        engine=engine, student_module=None, algorithm=Algorithm(), aggregator=agg, state=SimpleNamespace(micro_step=0)
    )
    gm_step = MultiObjectiveGradStep(MultiObjectiveGradStep.Config(combine="gradmag"))
    gm_step.rt = rt_gm
    gm_step.params = [param]
    metrics_gm = gm_step.accumulate([None, None], Norms(1, 1, 2))
    # metric/a/aux was 10.0 per micro-batch; with 2 micro-batches it should be averaged to 10.0, not 20.0
    assert "metric/a/aux" in metrics_gm
    torch.testing.assert_close(metrics_gm["metric/a/aux"], torch.tensor(10.0))

    # 2. PCGrad combine
    param.grad = None
    rt_pc = SimpleNamespace(
        engine=engine,
        student_module=None,
        algorithm=Algorithm(),
        aggregator=WeightedSum(WeightedSum.Config()),
        state=SimpleNamespace(micro_step=0),
    )
    pc_step = MultiObjectiveGradStep(MultiObjectiveGradStep.Config(combine="pcgrad"))
    pc_step.rt = rt_pc
    pc_step.params = [param]
    metrics_pc = pc_step.accumulate([None, None], Norms(1, 1, 2))
    assert "metric/a/aux" in metrics_pc
    torch.testing.assert_close(metrics_pc["metric/a/aux"], torch.tensor(10.0))
