"""Regression unit tests for bugs and optimizations from the 2x A100 full-flow audit."""

from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from baselines.preference import ADPA, advantage_expectation_loss
from baselines.same_tokenizer import skewed_forward_kl, skewed_reverse_kl
from lmkd.core.batch import Batch
from lmkd.core.config import recipe_from_dict
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.core.state import TrainState
from lmkd.core.validate import check_recipe
from lmkd.data.templates import build_prompt
from lmkd.data.vocab_map import VocabMap
from lmkd.eval.evaluators import DevLoss
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.losses.cross_tokenizer._common import DetachedLinear
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.pipeline.artifacts import ArtifactStore
from lmkd.pipeline.stages.preference import pad_mcq_template
from lmkd.reproduce import resolve_jobs
from lmkd.rollout.base import Rollout
from lmkd.train.builder import build_runtime
from lmkd.train.engine import AccelerateEngine, FSDP2Engine, LocalEngine


@CRITERIONS.register("_test_req_adv_ids_missing", force=True)
class _MissingAdvIdsCriterion(Criterion):
    needs_teacher: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ("cache.adv_ids",)

    class Config(BaseConfig):
        pass

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx: LossContext) -> LossOutput:
        raise AssertionError("Criterion with missing required cache keys must be skipped by DevLoss")


def test_dev_loss_bf16_autocast_augment_and_empty_shard_collectives(base_recipe, toy_jsonl, tmp_path, monkeypatch):
    rt = build_runtime(recipe_from_dict(base_recipe))

    # Spy on autocast + algorithm.augment + fp32 module under bf16 input
    fp32_proj = nn.Linear(4, 4)
    augment_seen: dict[str, bool] = {"called": False, "autocast": False}
    orig_augment = rt.algorithm.augment

    def _wrapped_augment(batch, runtime):
        augment_seen["called"] = True
        augment_seen["autocast"] = torch.is_autocast_enabled("cpu")
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        _ = fp32_proj(x)
        return orig_augment(batch, runtime)

    monkeypatch.setattr(rt.engine, "autocast", lambda: torch.amp.autocast("cpu", dtype=torch.bfloat16))
    monkeypatch.setattr(rt.algorithm, "augment", _wrapped_augment)

    ev = DevLoss(DevLoss.Config(data={"type": "jsonl", "path": toy_jsonl}, max_samples=4, batch_size=2))
    res = ev.evaluate(rt)
    assert augment_seen["called"] and augment_seen["autocast"]
    assert math.isfinite(res["dev_loss/loss"])

    # Empty shard on rank=1 with world_size=2 must still participate in collectives
    one_row = tmp_path / "one.jsonl"
    one_row.write_text(json.dumps({"id": "0", "prompt": "Q: w0 + w1 =", "response": "A: w1"}) + "\n")

    monkeypatch.setattr(rt.engine, "world_size", 2, raising=False)
    monkeypatch.setattr(rt.engine, "rank", 1, raising=False)
    monkeypatch.setattr(rt.engine, "is_partitioned", False, raising=False)

    gather_calls: list[object] = []
    reduce_calls: list[torch.Tensor] = []

    def _mock_all_gather(obj):
        gather_calls.append(obj)
        return [["nll"], obj]

    def _mock_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
        reduce_calls.append(tensor.clone())
        # Rank 0 contributed loss_num=6.0, z_tok=2.0, nll_num=6.0
        return tensor + torch.tensor([6.0, 2.0, 6.0], dtype=tensor.dtype, device=tensor.device)

    monkeypatch.setattr(rt.engine, "all_gather_object", _mock_all_gather)
    monkeypatch.setattr(rt.engine, "all_reduce_sum", _mock_all_reduce_sum)

    ev_empty = DevLoss(DevLoss.Config(data={"type": "jsonl", "path": str(one_row)}, max_samples=4, batch_size=2))
    ev_empty.setup(rt)
    assert len(ev_empty.loader) == 0
    empty_res = ev_empty.evaluate(rt)
    assert len(gather_calls) == 1
    assert len(reduce_calls) == 1 and reduce_calls[0].tolist() == [0.0, 0.0, 0.0]
    assert empty_res["dev_loss/loss"] == pytest.approx(3.0)
    assert empty_res["dev_loss/nll"] == pytest.approx(3.0)


def test_dev_loss_infers_sgl_spectral_transforms(base_recipe, tmp_path):
    cot_path = tmp_path / "cot.jsonl"
    rows = [
        {"id": "0", "problem": "Q: w0 + w1 =", "response": "w2 w3", "answer": "w3"},
        {"id": "1", "problem": "Q: w2 + w3 =", "response": "w4 w5", "answer": "w5"},
    ]
    cot_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    cfg_dict = copy.deepcopy(base_recipe)
    cfg_dict["data"]["transforms"] = []
    cfg_dict["train"]["batch_size"] = 2
    cfg_dict["train"]["micro_batch_size"] = 2
    cfg_dict["pipeline"] = [
        {
            "stage": "sgl_spectral",
            "source": {"type": "jsonl", "path": str(cot_path)},
            "energy_cutoff": 0.95,
            "top_p": 0.95,
            "dataset": True,
        }
    ]
    rt = build_runtime(recipe_from_dict(cfg_dict))
    ev = DevLoss(DevLoss.Config(data={"type": "jsonl", "path": str(cot_path)}, max_samples=2, batch_size=2))
    ev.setup(rt)
    assert len(ev.dataset) == 2
    assert "student.input_ids" in ev.dataset[0]
    res = ev.evaluate(rt)
    assert math.isfinite(res["dev_loss/loss"])


@pytest.mark.parametrize(
    "terms",
    [
        [{"criterion": "_test_req_adv_ids_missing", "alias": "req_crit"}],
        [{"criterion": "_test_req_adv_ids_missing", "alias": "req_crit"}, {"criterion": "nll", "alias": "nll"}],
    ],
)
def test_dev_loss_skips_missing_requires_criteria(base_recipe, toy_jsonl, terms):
    cfg_dict = copy.deepcopy(base_recipe)
    cfg_dict["loss"]["terms"] = terms
    rt = build_runtime(recipe_from_dict(cfg_dict))
    before_weights = dict(rt.loss_weights)

    ev = DevLoss(DevLoss.Config(data={"type": "jsonl", "path": toy_jsonl}, max_samples=4, batch_size=2))
    res = ev.evaluate(rt)
    assert math.isfinite(res["dev_loss/loss"])
    assert rt.loss_weights == before_weights


def test_detached_linear_and_logits_view_sharding_guard():
    torch.manual_seed(0)
    head = nn.Linear(8, 16)
    dlin = DetachedLinear(head)
    h1 = torch.randn(2, 4, 8, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)

    view = LogitsView(hidden=h1, lm_head=dlin)
    assert view._efficient_ok() is True

    out = dlin(h1)
    ref = F.linear(h2, head.weight.detach(), head.bias.detach())
    torch.testing.assert_close(out, ref)

    (out * out).sum().backward()
    (ref * ref).sum().backward()
    torch.testing.assert_close(h1.grad, h2.grad)
    assert head.weight.grad is None
    assert head.bias.grad is None

    head_zero3 = nn.Linear(8, 16)
    head_zero3.weight.ds_id = 0  # type: ignore[attr-defined]
    assert LogitsView(hidden=h1, lm_head=DetachedLinear(head_zero3))._efficient_ok() is False

    head_fsdp = nn.Linear(8, 16)
    head_fsdp._lmkd_fsdp_managed = True  # type: ignore[attr-defined]
    assert LogitsView(hidden=h1, lm_head=DetachedLinear(head_fsdp))._efficient_ok() is False


def test_vocab_map_save_atomic_and_artifact_finalize_ignores_tmp(tiny_models, tmp_path):
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    vm = VocabMap.build(tok, tok)
    vm_path = tmp_path / "vm_dir" / "vocab_map.pt"
    vm.save(vm_path)
    assert vm_path.exists()
    assert not [p for p in vm_path.parent.iterdir() if p.name.endswith(".tmp")]
    loaded = VocabMap.load(vm_path)
    assert torch.equal(loaded.tea2stu, vm.tea2stu)
    assert torch.equal(loaded.overlap_s, vm.overlap_s)

    store = ArtifactStore(tmp_path / "artifacts")
    art = store.artifact("node", "stage", {"k": 1})
    art.path.mkdir(parents=True, exist_ok=True)
    (art.path / "data.txt").write_text("payload")
    (art.path / "manifest.json.99999.tmp").write_text("partial")
    store.finalize(art, {"config": {}, "inputs": {}})
    files = art.manifest()["files"]
    assert "data.txt" in files
    assert "manifest.json.99999.tmp" not in files


def test_prompt_and_pad_mcq_without_chat_template(tiny_models):
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    assert getattr(tok, "chat_template", None) is None

    prompt = build_prompt(tok, "Compute 2+2", chat_template=True)
    assert "Compute 2+2" in prompt

    tmpl = pad_mcq_template(tok)
    assert "### Query" in tmpl.template and "$instruction" in tmpl.template


def test_resolve_jobs_vpd_propagates_student_to_sample_candidates(tmp_path):
    jobs = resolve_jobs("vpd", tmp_path / "vpd", student="/custom/student", teacher="/custom/teacher")
    sample_stages = [
        st
        for st in jobs[0].pipeline
        if (st.get("stage") if isinstance(st, dict) else getattr(st, "stage", None)) == "sample_candidates"
    ]
    assert len(sample_stages) == 1
    st0 = sample_stages[0]
    model = st0["model"] if isinstance(st0, dict) else st0.model
    assert model == "/custom/student"


def test_check_recipe_rejects_undefined_group_optimizer(base_recipe):
    cfg_dict = copy.deepcopy(base_recipe)
    cfg_dict["optim"]["groups"] = [{"name": "default", "match": {}, "optimizer": "undefined_opt"}]
    rep = check_recipe(recipe_from_dict(cfg_dict))
    assert not rep.ok
    assert any("undefined_opt" in err for err in rep.errors)


def test_same_tokenizer_exports_skewed_kl():
    assert callable(skewed_forward_kl)
    assert callable(skewed_reverse_kl)
    s_logits = torch.randn(2, 5, 11)
    t_logits = torch.randn(2, 5, 11)
    assert torch.isfinite(skewed_forward_kl(s_logits, t_logits, alpha=0.1)).all()
    assert torch.isfinite(skewed_reverse_kl(s_logits, t_logits, alpha=0.1)).all()


def test_adpa_chunked_matches_advantage_expectation_loss():
    torch.manual_seed(7)
    B, T, H, V, K = 4, 6, 8, 12, 3
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    adv_ids = torch.randint(0, V, (B, T, K))
    adv_val = torch.randn(B, T, K)

    head_chunk = nn.Linear(H, V)
    head_ref = copy.deepcopy(head_chunk)
    h_chunk = torch.randn(B, T, H, requires_grad=True)
    h_ref = h_chunk.detach().clone().requires_grad_(True)

    batch = Batch(
        {
            "student.labels": labels,
            "group.size": torch.tensor([2, 2, 2, 2]),
            "group.valid": torch.tensor([True, True, True, True]),
            "cache.adv_ids": adv_ids,
            "cache.adv_val": adv_val,
        }
    )
    side = SimpleNamespace(logits=LogitsView(hidden=h_chunk, lm_head=head_chunk))
    ctx = LossContext(
        batch=batch,
        student=side,  # type: ignore[arg-type]
        student_views=[side],  # type: ignore[list-item]
        teacher=None,
        modules=ModuleRegistry(),
        state=SimpleNamespace(step=0),
        rt=SimpleNamespace(student_model=None),
    )
    adpa = ADPA(ADPA.Config(chunk_size=2))
    out = adpa.forward(ctx)
    loss_chunk = out.terms["adpa"].value
    loss_chunk.backward()

    probs_rej = F.pad(torch.softmax(head_ref(h_ref[1::2, :-1]), dim=-1), (0, 0, 1, 0))
    loss_ref = advantage_expectation_loss(probs_rej, labels[1::2], adv_ids[1::2], adv_val[1::2])
    loss_ref.backward()

    torch.testing.assert_close(loss_chunk, loss_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(h_chunk.grad, h_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(head_chunk.weight.grad, head_ref.weight.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(head_chunk.bias.grad, head_ref.bias.grad, atol=1e-5, rtol=1e-5)


def test_distillm2_producer_strips_rejected_rollout_logp(base_recipe, tiny_models, monkeypatch):
    cfg_dict = copy.deepcopy(base_recipe)
    cfg_dict["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    cfg_dict["teacher"] = {"provider": "hf_frozen", "input": "same"}
    cfg_dict["rollout"] = {"type": "hf"}
    cfg_dict["producer"] = {"type": "distillm2", "sampling": {"max_new_tokens": 4, "return_logprobs": True}}
    cfg_dict["loss"]["terms"] = [
        {
            "criterion": "distillm2",
            "alias": "distillm2",
            "params": {
                "beta": 1.0,
                "base_alpha_1": 0.1,
                "base_alpha_2": 0.1,
                "update_alpha": False,
                "gradual_beta": False,
                "average_log_prob": True,
            },
        }
    ]
    cfg_dict["data"]["max_length"] = 20
    cfg_dict["train"]["batch_size"] = 4
    cfg_dict["train"]["micro_batch_size"] = 2

    rt = build_runtime(recipe_from_dict(cfg_dict))

    def _mock_generate(prompts, spec, **kw):
        return [
            [Rollout(prompt_ids=list(p), response_ids=[5, 6, 1], response_logp=[-0.5, -0.4, -0.3])] for p in prompts
        ]

    monkeypatch.setattr(rt.rollout, "generate", _mock_generate)
    batch = rt.producer.next(TrainState())
    assert "student.input_ids" in batch
    assert "rollout.rollout_logp" not in batch
    assert (batch["student.prompt_len"] <= batch["student.input_ids"].shape[1]).all()


@pytest.mark.parametrize("raw_val", [False, None, "none", "fp32", "no"])
def test_engine_configs_normalize_mixed_precision(raw_val):
    assert LocalEngine.Config(mixed_precision=raw_val).mixed_precision == "no"
    assert AccelerateEngine.Config(mixed_precision=raw_val).mixed_precision == "no"
    assert FSDP2Engine.Config(mixed_precision=raw_val).mixed_precision == "no"
