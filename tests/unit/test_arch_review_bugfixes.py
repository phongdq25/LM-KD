"""Regression tests covering the 10 bugs identified in the LM-KD architectural review."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import lmkd
import lmkd.losses.sft as sft_mod
from baselines.same_tokenizer.fdd.loss import FDDCriterion
from lmkd import Inputs, TokenLoss
from lmkd.cli import main
from lmkd.core.batch import Batch
from lmkd.core.plugin_index import _registry_vars, scan_source
from lmkd.core.registry import CRITERIONS
from lmkd.core.state import TrainState
from lmkd.data.producers import AdaptiveSGOProducer
from lmkd.losses.base import LossContext
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import ForwardRunner, SideOutput, SideRequest
from lmkd.models.logits_view import LogitsView
from lmkd.models.poolers import LastTokenPooler
from lmkd.reproduce import _environment
from lmkd.teachers.signals import TeacherOutput
from lmkd.train.builder import setup_components
from lmkd.train.engine import Engine, LocalEngine, MasterWeightOptimizer
from lmkd.train.step_strategies import count_supervised


def _make_side(h: torch.Tensor, head: torch.nn.Linear, hidden_states: tuple[torch.Tensor, ...] | None = None):
    out = SideOutput(torch.nn.Identity(), HFAutoAdapter(), None, None)
    out._out = SimpleNamespace(last_hidden_state=h, hidden_states=hidden_states or (h,))
    out.logits = LogitsView(h, head)
    return out


def test_bug1_side_output_target_hidden_and_loss_decorator_alignment():
    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    head = torch.nn.Linear(2, 5, bias=False)
    side = _make_side(h, head)

    shifted = side.target_hidden(-1)
    assert torch.equal(shifted[:, 0], torch.zeros(1, 2))
    assert torch.equal(shifted[:, 1:], h[:, :-1])

    captured: dict[str, torch.Tensor] = {}

    @lmkd.loss("bug1_aligned_loss", chunked=False)
    def aligned_loss(s: Inputs) -> TokenLoss:
        captured["last_hidden"] = s.last_hidden.detach().clone()
        return TokenLoss((s.last_hidden.square().mean(-1)) + s.logits.square().mean(-1))

    labels = torch.tensor([[-100, -100, 2, 3]])
    batch = Batch({"student.input_ids": labels.clamp_min(0), "student.labels": labels})
    ctx = LossContext(batch, side, [side], None, None, TrainState(), None)
    aligned_loss.criterion_class()(ctx)

    torch.testing.assert_close(captured["last_hidden"], shifted)


def test_bug1_fdd_criterion_target_aligned_hidden():
    torch.manual_seed(0)
    B, T, D, V = 1, 5, 4, 8
    prompt_len = 2
    s_head = torch.nn.Linear(D, V, bias=False)
    t_head = torch.nn.Linear(D, V, bias=False).requires_grad_(False)
    s_h0 = torch.randn(B, T, D, requires_grad=True)
    s_h1 = torch.randn(B, T, D, requires_grad=True)
    t_h0 = torch.randn(B, T, D)
    t_h1 = torch.randn(B, T, D)

    s_side = _make_side(s_h1, s_head, hidden_states=(s_h0, s_h0, s_h1))
    t_side = _make_side(t_h1, t_head, hidden_states=(t_h0, t_h0, t_h1))
    s_side.model = SimpleNamespace(lm_head=s_head)
    t_side.model = SimpleNamespace(lm_head=t_head)

    labels = torch.tensor([[-100, -100, 1, 2, 3]])
    batch = Batch({"student.input_ids": labels.clamp_min(0), "student.labels": labels})
    ctx = LossContext(batch, s_side, [s_side], TeacherOutput(side=t_side, logits=t_side.logits), None, TrainState(), None)

    crit = FDDCriterion(FDDCriterion.Config(student_layers=[-2, -1], teacher_layers=[-2, -1]))
    loss = crit.forward(ctx).terms["fdd"].value
    loss.backward()

    # Position (prompt_len - 1) = 1 predicts first response token at index 2 -> must receive gradient.
    assert s_h1.grad[:, prompt_len - 1].abs().sum() > 0
    # Final position (T - 1) predicts token T (beyond sequence) -> must receive zero gradient.
    assert s_h1.grad[:, -1].abs().sum() == 0


def test_bug2_setup_components_order_adaptive_sgo():
    order: list[str] = []

    class FakeAlgorithm:
        def __init__(self):
            self.ready = False

        def setup(self, rt):
            order.append("algorithm")
            self.ready = True

        def loss(self, micro, rt):
            assert self.ready, "algorithm.setup must run before producer.setup triggers eval"
            return {"loss": torch.tensor(0.25)}

    class FakeStepStrategy:
        def setup(self, rt):
            order.append("step_strategy")

    class FakeEvaluator:
        alias = "dev_loss"
        name_prefix = "dev_loss"

        def setup(self, rt):
            order.append("evaluator")

        def evaluate(self, rt):
            rt.algorithm.loss(None, rt)
            return {"dev_loss/loss": 0.42}

    producer = AdaptiveSGOProducer(AdaptiveSGOProducer.Config(eval_step_zero=True))
    rt = SimpleNamespace(
        config=SimpleNamespace(train=SimpleNamespace(seed=7, batch_size=2, micro_batch_size=1)),
        engine=SimpleNamespace(world_size=1, rank=0, device=torch.device("cpu")),
        train_dataset=[{"student.input_ids": [1, 2]}, {"student.input_ids": [1, 2]}],
        collator=lambda x: x,
        data_ctx=None,
        teacher=None,
        teacher_input=None,
        algorithm=FakeAlgorithm(),
        step_strategy=FakeStepStrategy(),
        rollout=None,
        producer=producer,
        grad_hooks=[],
        grad_transforms=[],
        callbacks=SimpleNamespace(cbs=[]),
        evaluators=[FakeEvaluator()],
    )
    setup_components(rt)
    assert order.index("algorithm") < order.index("evaluator")
    assert producer.prev == pytest.approx(0.42)


class _DummyScaler:
    def __init__(self, scale: float = 16.0):
        self.scale = scale
        self.unscaled_opts: list[object] = []

    def get_scale(self) -> float:
        return self.scale

    def unscale_(self, opt) -> None:
        self.unscaled_opts.append(opt)
        for g in opt.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.grad.div_(self.scale)

    def step(self, opt) -> None:
        if opt not in self.unscaled_opts:
            self.unscale_(opt)
        opt.step()

    def update(self) -> None:
        self.unscaled_opts.clear()


def test_bug3_master_weight_optimizer_unscale_clip_step_precision():
    # 1. Unscale + step preserves full FP32 master gradient precision (no bfloat16 truncation).
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    raw_sgd = torch.optim.SGD([p], lr=1.0)
    engine = LocalEngine(LocalEngine.Config(device="cpu", mixed_precision="no", master_weights=True))
    engine.scaler = _DummyScaler(scale=16.0)
    _, _, opts = engine.prepare(torch.nn.Identity(), torch.nn.Identity(), {"main": raw_sgd})
    opt = opts["main"]
    assert isinstance(opt, MasterWeightOptimizer)

    # Choose a gradient value whose exact value (1.0 / 16.0 + 1e-4 / 16.0) would lose precision if cast to bf16.
    p.grad = torch.full((4,), 16.0 + 2**-6, dtype=torch.float32).to(torch.bfloat16)
    expected_fp32_grad = p.grad.detach().float() / 16.0

    engine.unscale(opts)
    master_p = opt.param_map[p]
    torch.testing.assert_close(master_p.grad, expected_fp32_grad, atol=0.0, rtol=0.0)

    engine.step_optimizers(opts)
    torch.testing.assert_close(master_p.data, 1.0 - expected_fp32_grad, atol=1e-6, rtol=0.0)

    # 2. If a GradTransform mutates p.grad after unscale(), step() picks up the mutated gradient.
    p2 = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    raw_sgd2 = torch.optim.SGD([p2], lr=1.0)
    engine2 = LocalEngine(LocalEngine.Config(device="cpu", mixed_precision="no", master_weights=True))
    engine2.scaler = _DummyScaler(scale=4.0)
    _, _, opts2 = engine2.prepare(torch.nn.Identity(), torch.nn.Identity(), {"main": raw_sgd2})
    p2.grad = torch.full((4,), 8.0, dtype=torch.bfloat16)
    engine2.unscale(opts2)
    p2.grad.fill_(0.5)  # external GradTransform override
    engine2.step_optimizers(opts2)
    torch.testing.assert_close(opts2["main"].param_map[p2].data, torch.full((4,), 0.5))

    # 3. Calling step_optimizers() without prior unscale() still syncs p.grad to master before scaler.step().
    p3 = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    raw_sgd3 = torch.optim.SGD([p3], lr=1.0)
    engine3 = LocalEngine(LocalEngine.Config(device="cpu", mixed_precision="no", master_weights=True))
    engine3.scaler = _DummyScaler(scale=4.0)
    _, _, opts3 = engine3.prepare(torch.nn.Identity(), torch.nn.Identity(), {"main": raw_sgd3})
    p3.grad = torch.full((4,), 8.0, dtype=torch.bfloat16)
    engine3.step_optimizers(opts3)
    torch.testing.assert_close(opts3["main"].param_map[p3].data, torch.full((4,), -1.0))

    # 4. Engine.clip_grad_norm clips master_p in FP32 and syncs clipped grad back to p.grad.
    p4 = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    raw_sgd4 = torch.optim.SGD([p4], lr=1.0)
    engine4 = Engine()
    mwo4 = MasterWeightOptimizer(raw_sgd4)
    engine4._master_param_map.update(mwo4.param_map)
    engine4._master_optimizers.append(mwo4)
    p4.grad = torch.full((4,), 4.0, dtype=torch.bfloat16)  # total L2 norm = 8.0
    mwo4.sync_grads_to_master()
    norm = engine4.clip_grad_norm([p4], max_norm=2.0)
    assert float(norm) == pytest.approx(8.0)
    torch.testing.assert_close(mwo4.param_map[p4].grad, torch.full((4,), 1.0))
    torch.testing.assert_close(p4.grad.float(), torch.full((4,), 1.0))


def test_bug4_forward_runner_pooled_uses_hooks_not_all_hidden(monkeypatch):
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=32, n_embd=16, n_layer=2, n_head=4, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    )
    model.eval()
    adapter = HFAutoAdapter()
    orig = adapter.forward_hidden

    def checked(*args, **kwargs):
        assert kwargs["output_hidden_states"] is False
        return orig(*args, **kwargs)

    monkeypatch.setattr(adapter, "forward_hidden", checked)
    inputs = {"input_ids": torch.randint(0, 32, (2, 5)), "attention_mask": torch.ones(2, 5, dtype=torch.long)}
    out = ForwardRunner.run(model, adapter, inputs, SideRequest(pooled=((-1, "mean"), (1, "mean"))), grad=False)[0]
    assert out.pooled(1, "mean").shape == (2, 16)
    assert out.pooled(-1, "mean").shape == (2, 16)


def test_bug5_last_token_pooler_non_contiguous_and_zero_mask():
    pooler = LastTokenPooler()
    h = torch.arange(20, dtype=torch.float32).view(2, 5, 2)
    mask = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 0, 0]])
    out = pooler(h, mask)
    torch.testing.assert_close(out[0], h[0, 3])
    torch.testing.assert_close(out[1], h[1, 0])


def test_bug6_loss_decorator_teacher_chunk_guard_and_shape_fallback():
    torch.manual_seed(3)
    s_head = torch.nn.Linear(4, 9, bias=False)
    t_head = torch.nn.Linear(4, 11, bias=False)  # mismatched vocab size
    s_h = torch.randn(2, 5, 4, requires_grad=True)
    t_h = torch.randn(2, 6, 4)  # mismatched sequence length
    s_side = _make_side(s_h, s_head)
    t_side = _make_side(t_h, t_head)

    batch = Batch(
        {
            "student.input_ids": torch.randint(0, 9, (2, 5)),
            "student.labels": torch.tensor([[-100, 1, 2, 3, 4], [-100, 2, 3, -100, -100]]),
            "teacher.labels": torch.tensor([[-100, 1, 2, 3, 4, 5], [-100, 2, 3, 4, -100, -100]]),
        }
    )
    ctx = LossContext(batch, s_side, [s_side], TeacherOutput(side=t_side, logits=t_side.logits), None, TrainState(), None)

    @lmkd.loss("bug6_t_last_hidden", teacher=True, chunk_size=2)
    def uses_teacher_last_hidden(s: Inputs, t: Inputs) -> TokenLoss:
        return TokenLoss((s.last_hidden - t.last_hidden[:, : s.last_hidden.shape[1]]).square().mean(-1))

    res1 = uses_teacher_last_hidden.criterion_class()(ctx)
    assert torch.isfinite(res1.terms["loss"].value)

    @lmkd.loss("bug6_shape_auto_fallback", teacher=True, chunk_size=2)
    def shape_mismatch_auto(s: Inputs, t: Inputs) -> TokenLoss:
        return TokenLoss(s.logits.square().mean(-1) + t.logits.mean().detach() * 0.0)

    res2 = shape_mismatch_auto.criterion_class()(ctx)
    assert torch.isfinite(res2.terms["loss"].value)

    @lmkd.loss("bug6_shape_strict_chunk", teacher=True, chunked=True, chunk_size=2)
    def shape_mismatch_strict(s: Inputs, t: Inputs) -> TokenLoss:
        return TokenLoss(s.logits.square().mean(-1))

    with pytest.raises(ValueError, match="aligned sequence shapes"):
        shape_mismatch_strict.criterion_class()(ctx)


def test_bug7_init_baseline_no_double_import_and_transform_requires(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "examples" / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "examples" / "data" / "train_sample.jsonl").write_text('{"prompt": "hi", "response": "ok"}\n')
    assert main(["init", "gamma_pkg_kd"]) == 0

    sys.path.insert(0, str(tmp_path))
    try:
        importlib.invalidate_caches()
        pkg = importlib.import_module("dev.gamma_pkg_kd")
        assert CRITERIONS.get("gamma_pkg_kd") is pkg.GammaPkgKdLoss
        assert pkg.GammaPkgKdTransform.requires == ("student.input_ids",)
    finally:
        sys.path.pop(0)
        for k in [m for m in list(sys.modules) if m == "dev" or m.startswith("dev.gamma_pkg_kd")]:
            sys.modules.pop(k, None)


def test_bug8_plugin_index_scan_source_decorators_and_kwargs():
    src = """
@lmkd.loss("lit_loss")
def fn1(s): ...

@loss(name="kw_loss")
def fn2(s): ...

@CRITERIONS.register(name="kw_crit")
class CritA: ...

@CRITERIONS.register()
class DefaultCrit: ...

@CRITERIONS.register
class BareCrit: ...
"""
    found = set(scan_source(src, _registry_vars()))
    assert ("criterions", "lit_loss") in found
    assert ("criterions", "kw_loss") in found
    assert ("criterions", "kw_crit") in found
    assert ("criterions", "defaultcrit") in found
    assert ("criterions", "barecrit") in found


def test_bug9_and_bug10_count_supervised_sft_all_reproduce_hashes():
    batch = Batch(
        {
            "student.labels": torch.tensor([[-100, 1, 2], [-100, 3, 4], [-100, 5, 6]]),
            "seq.loss_mask": torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        }
    )
    z_tok, z_seq = count_supervised(batch)
    assert z_tok == 4.0
    assert z_seq == 2.0

    assert set(sft_mod.__all__) == {"NLL", "WeightedNLL"}

    env = _environment("fdd", [])
    assert any("src/baselines/" in path for path in env["source_sha256"])
