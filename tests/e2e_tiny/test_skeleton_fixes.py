"""Regression tests for known architectural edge cases and fixes."""
from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import torch

from lmkd.core.batch import Batch
from lmkd.core.config import recipe_from_dict
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CALLBACKS, CRITERIONS, ENGINES, GRAD_HOOKS, STAGES, TRANSFORMS
from lmkd.data.transforms.base import Transform
from lmkd.losses.base import Criterion, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.modules import LinearProjector
from lmkd.pipeline.stage import Stage
from lmkd.train.builder import build_runtime
from lmkd.train.callbacks import Callback
from lmkd.train.engine import LocalEngine
from lmkd.train.grad import TensorGradHook
from lmkd.train.trainer import Trainer


# ----------------------------------------------------------------------------- test plugins
@ENGINES.register("_test_clip_engine")
class _ClipEngine(LocalEngine):
    handles_clipping = True


@CALLBACKS.register("_test_logs_custom")
class _LogsCustom(Callback):
    class Config(BaseConfig):
        pass

    def on_step_begin(self, rt):
        rt.state.log("custom/x", 3.0)


@GRAD_HOOKS.register("_test_zero_grad")
class _ZeroGrad(TensorGradHook):
    class Config(TensorGradHook.Config):
        pass

    def __call__(self, grad, ctx):
        ctx.state.log("hook/grad_norm", grad.norm())
        return torch.zeros_like(grad)


_SEEN_RT = {}


@TRANSFORMS.register("_test_setup_probe")
class _SetupProbe(Transform):
    class Config(BaseConfig):
        pass

    def setup(self, rt):
        _SEEN_RT["rt"] = rt

    def __call__(self, sample, ctx):
        return sample


@CRITERIONS.register("_test_proj_only")
class _ProjOnly(Criterion):
    class Config(BaseConfig):
        pass

    def features(self):
        return FeatureRequest(student=SideRequest(hidden=(-1,)))

    def build_modules(self, info, modules):
        modules.add(f"{self.alias}.p", LinearProjector(LinearProjector.Config(in_dim=info.student_hidden, out_dim=4)))

    def forward(self, ctx):
        h = ctx.modules.get(f"{self.alias}.p")(ctx.student.hidden(-1))
        m = ctx.batch["student.attention_mask"].unsqueeze(-1).float()
        return LossOutput({"sq": LossTerm(((h - 1.0) ** 2 * m).sum(), "token_sum")})


_RUNS = {"n": 0}


@STAGES.register("_test_samples")
class _SamplesStage(Stage):
    class Config(BaseConfig):
        n: int = 16

    def run(self, ctx):
        _RUNS["n"] += 1
        ctx.output.path.mkdir(parents=True, exist_ok=True)
        rows = [{"meta.id": str(i), "prompt": f"Q: w{i % 5} =", "response": f"A: w{i % 7}"} for i in range(self.cfg.n)]
        (ctx.output.path / "rows.json").write_text(json.dumps(rows))
        return {"n": len(rows)}

    def load_samples(self, art):
        from lmkd.data.transforms.tokenize import TokenizeSFT

        rows = json.loads((art.path / "rows.json").read_text())
        tok = TokenizeSFT(TokenizeSFT.Config(use_chat_template=False))
        return [tok(r, self._ctx) for r in rows]

    def setup(self, rt):
        self._ctx = rt.data_ctx


@STAGES.register("_test_attach")
class _AttachStage(Stage):
    class Config(BaseConfig):
        pass

    def run(self, ctx):
        ctx.output.path.mkdir(parents=True, exist_ok=True)
        return {"inputs": sorted(ctx.inputs)}

    def attach(self, art, rt):
        rt.extra["attached"] = art.key
        rt.extra["attach_inputs"] = art.manifest()["inputs"]


def _metrics(rt):
    return [json.loads(l) for l in (Path(rt.config.train.output_dir) / "metrics.jsonl").read_text().splitlines()]


def _kd_recipe(base_recipe, tiny_models):
    base_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    base_recipe["loss"] = {"terms": [{"criterion": "nll"},
                                     {"criterion": "token_kd", "alias": "kd", "weight": 0.0}]}
    return base_recipe


# ----------------------------------------------------------------------------- K1
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_k1_student_forward_goes_through_ddp_wrapper(base_recipe):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    rt = build_runtime(recipe_from_dict(base_recipe))
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=1)
    try:
        calls = {"n": 0}
        ddp = DDP(rt.student_module)
        ddp.register_forward_pre_hook(lambda m, a, k: calls.__setitem__("n", calls["n"] + 1), with_kwargs=True)
        rt.student_module = ddp
        out = rt.step_strategy.run(rt.producer.next(rt.state), rt)
        assert calls["n"] == 2                     # 8 rows / micro 4 -> two forwards through DDP.forward
        assert torch.isfinite(out.metrics["loss/total"])
    finally:
        dist.destroy_process_group()


# ----------------------------------------------------------------------------- K2
class _SkippingScaler:
    def __init__(self):
        self.s = 1024.0

    def scale(self, loss):
        return loss

    def unscale_(self, opt):
        pass

    def step(self, opt):
        pass                                       # overflow -> skipped

    def update(self):
        self.s /= 2

    def get_scale(self):
        return self.s


def test_k2_scheduler_not_stepped_when_scaler_skips(base_recipe):
    base_recipe["optim"]["schedulers"] = [{"type": "linear", "warmup_ratio": 0.5}]
    rt = build_runtime(recipe_from_dict(base_recipe))
    rt.engine.scaler = _SkippingScaler()
    w0 = next(rt.student_model.parameters()).detach().clone()
    lr0 = rt.schedulers[0].get_last_lr()
    out = rt.step_strategy.run(rt.producer.next(rt.state), rt)
    assert out.skipped
    assert rt.schedulers[0].get_last_lr() == lr0
    assert torch.equal(next(rt.student_model.parameters()).detach(), w0)


# ----------------------------------------------------------------------------- K3
def test_k3_producer_deterministic_resume_and_epoch(base_recipe):
    rt_a = build_runtime(recipe_from_dict(base_recipe))
    seq_a = [rt_a.producer.next(rt_a.state)["student.input_ids"] for _ in range(6)]   # 32 rows / 8 = 4 per epoch
    assert rt_a.state.epoch == pytest.approx(1.5)

    rt_b = build_runtime(recipe_from_dict(base_recipe))
    rt_b.producer.next(rt_b.state)
    rt_b.producer.next(rt_b.state)
    state = rt_b.producer.state_dict()
    rt_c = build_runtime(recipe_from_dict(base_recipe))
    rt_c.producer.load_state_dict(state)
    seq_c = [rt_c.producer.next(rt_c.state)["student.input_ids"] for _ in range(4)]
    for a, c in zip(seq_a[2:], seq_c):
        assert torch.equal(a, c)


# ----------------------------------------------------------------------------- K4 + K9
def test_k4_k9_state_log_and_grad_hook(base_recipe):
    base_recipe["callbacks"] = ["_test_logs_custom"]
    base_recipe["grad"] = {"clip_norm": None, "hooks": [{"type": "_test_zero_grad", "target": "student.hidden[1]"}]}
    base_recipe["train"]["max_steps"] = 2
    base_recipe["train"]["log_every"] = 1
    rt = build_runtime(recipe_from_dict(base_recipe))
    rt.step_strategy.optimizer_step = lambda: True            # keep grads to inspect them
    out = rt.step_strategy.run(rt.producer.next(rt.state), rt)
    # hidden(1) = output of block 1 = transformer.h[0]; zeroing its grad cuts h[0], h[1] still learns
    block0 = rt.student_model.transformer.h[0].mlp.c_fc.weight.grad
    block1 = rt.student_model.transformer.h[1].mlp.c_fc.weight.grad
    assert block0 is None or block0.abs().max() == 0
    assert block1 is not None and block1.abs().max() > 0
    assert "hook/grad_norm" in rt.state.extra["metrics"]

    rt2 = Trainer.from_config(recipe_from_dict(base_recipe)).fit()
    m = _metrics(rt2)
    assert m[-1]["custom/x"] == 3.0 and "hook/grad_norm" in m[-1]


# ----------------------------------------------------------------------------- K5
def test_k5_no_clipnorm_when_engine_clips(base_recipe):
    base_recipe["engine"] = {"type": "_test_clip_engine", "device": "cpu", "mixed_precision": "no"}
    rt = build_runtime(recipe_from_dict(base_recipe))
    assert not any(type(t).__name__ == "ClipNorm" for t in rt.grad_transforms)


# ----------------------------------------------------------------------------- K6
def test_k6_request_follows_current_weights(base_recipe, tiny_models):
    rt = build_runtime(recipe_from_dict(_kd_recipe(base_recipe, tiny_models)))
    assert rt.algorithm.request.teacher is None
    rt.loss_weights["kd"] = 0.5
    assert rt.algorithm.request.teacher is not None
    out = rt.step_strategy.run(rt.producer.next(rt.state), rt)
    assert "loss/kd/fkl" in out.metrics


# ----------------------------------------------------------------------------- K7
def test_k7_rollout_samples_collate(base_recipe):
    from lmkd.rollout.base import Rollout, rollouts_to_samples

    rt = build_runtime(recipe_from_dict(base_recipe))
    rs = [Rollout([3, 4, 5], [6, 1], [-0.5, -0.1]), Rollout([3, 4], [7, 8, 1], [-1.0, -2.0, -0.2])]
    b = rt.collator(rollouts_to_samples(rs))
    assert b["rollout.rollout_logp"].shape == b["student.input_ids"].shape
    assert b["rollout.rollout_logp"][0, 3].item() == -0.5 and b["student.labels"][0, 3].item() == 6


# ----------------------------------------------------------------------------- K8
def test_k8_response_shift_aligns_teacher_to_student(base_recipe, tiny_models):
    from lmkd.data.labels import make_labels
    from lmkd.teachers.signals import TeacherSignalSpec

    base_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    base_recipe["teacher"] = {"provider": {"type": "hf_frozen", "position_map": "response_shift"}, "input": "same"}
    rt = build_runtime(recipe_from_dict(base_recipe))
    resp = [10, 11, 12, 1]
    s_ids, t_ids = [3, 4] + resp, [5, 6, 7, 8, 9] + resp                  # privileged (longer) teacher prompt
    sample = lambda ids, plen, side: {f"{side}.input_ids": ids, f"{side}.attention_mask": [1] * len(ids),
                                      f"{side}.labels": make_labels(ids, plen).tolist(), f"{side}.prompt_len": plen}
    b = rt.collator([{**sample(s_ids, 2, "student"), **sample(t_ids, 5, "teacher")}])
    out = rt.teacher.signals(b, TeacherSignalSpec(kind="sampled_logp"), None, None)
    with torch.no_grad():
        logits = rt.teacher_model(input_ids=b["teacher.input_ids"]).logits.float()
    ref = torch.log_softmax(logits[0, 4:8], -1).gather(-1, torch.tensor(resp)[:, None]).squeeze(-1)
    assert torch.allclose(out.token_logp[0, 2:6], ref, atol=1e-5)          # student positions 2..5


# ----------------------------------------------------------------------------- K10 + K11
def test_k10_k11_pipeline_named_inputs_attach_and_cache(base_recipe):
    base_recipe["pipeline"] = [{"stage": "_test_samples", "name": "data"},
                               {"stage": "_test_attach", "inputs": {"data": "data"}}]
    base_recipe["data"]["transforms"] = [{"type": "_test_setup_probe"}]
    _RUNS["n"] = 0
    rt = build_runtime(recipe_from_dict(base_recipe))
    assert len(rt.train_dataset) == 16 and rt.extra["attach_inputs"] == ["data"]
    assert _SEEN_RT["rt"] is rt                                           # K11 transform.setup(rt)
    rt2 = build_runtime(recipe_from_dict(base_recipe))                    # cached: run() not called again
    assert _RUNS["n"] == 1 and rt2.extra["attached"] == rt.extra["attached"]


def test_k10_pipeline_without_sample_stage_falls_back(base_recipe):
    base_recipe["pipeline"] = [{"stage": "_test_attach"}]                 # produces no samples
    rt = build_runtime(recipe_from_dict(base_recipe))
    assert len(rt.train_dataset) == 32                                    # fell back to data.train (toy_jsonl)
    assert rt.extra["attach_inputs"] == []


def test_k10_check_rejects_unknown_input(base_recipe):
    from lmkd.core.validate import check_recipe

    base_recipe["pipeline"] = [{"stage": "_test_attach", "inputs": {"data": "nope"}}]
    rep = check_recipe(recipe_from_dict(base_recipe))
    assert not rep.ok and "not an earlier stage name" in rep.render()


# ----------------------------------------------------------------------------- K12
def test_k12_frozen_student_trains_aux_modules_only(base_recipe):
    base_recipe["models"]["student"]["trainable"] = False
    base_recipe["loss"] = {"terms": [{"criterion": "_test_proj_only", "alias": "po"}]}
    base_recipe["train"]["max_steps"] = 3
    rt = build_runtime(recipe_from_dict(base_recipe))
    w0 = {n: p.detach().clone() for n, p in rt.student_model.named_parameters()}
    m0 = rt.modules.get("po.p").proj.weight.detach().clone()
    Trainer(rt).fit()
    assert all(torch.equal(p, w0[n]) for n, p in rt.student_model.named_parameters())
    assert not torch.equal(rt.modules.get("po.p").proj.weight, m0)


# ----------------------------------------------------------------------------- K13
def test_k13_split_refuses_to_cut_groups():
    b = Batch({"student.input_ids": torch.zeros(6, 3, dtype=torch.long),
               "group.index": torch.tensor([0, 1, 2, 0, 1, 2])})
    assert len(b.split(2)) == 2
    with pytest.raises(ValueError, match="cuts a group"):
        b.split(3)


# ----------------------------------------------------------------------------- K14
def test_k14_csv_and_tokenize_text(tmp_path, tiny_models):
    from transformers import AutoTokenizer

    from lmkd.data.context import DataContext
    from lmkd.data.sources import CsvSource
    from lmkd.data.transforms.tokenize import TokenizeSFT, TokenizeText

    p = tmp_path / "t.csv"
    p.write_text(",text,source\n0,w1 w2 w3,a\n1,w4,b\n")
    rows = list(CsvSource(CsvSource.Config(path=str(p))).records())
    assert rows[0]["text"] == "w1 w2 w3" and rows[1]["meta.id"] == "1"
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    ctx = DataContext(student_tokenizer=tok, max_length=2)
    s = TokenizeText(TokenizeText.Config())(rows[0], ctx)
    assert len(s["student.input_ids"]) == 2 and "student.labels" not in s
    cfg = TokenizeSFT.Config(use_chat_template=False, prompt_add_special_tokens=False)
    out = TokenizeSFT(cfg)({"prompt": "Q:", "response": "A:"}, DataContext(student_tokenizer=tok))
    assert out["student.prompt_len"] == 1
