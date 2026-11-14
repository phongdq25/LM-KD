"""T08: rollout helpers + HF rollout engine (CPU)."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.conftest import _make_tokenizer

from lmkd.core.config import recipe_from_dict
from lmkd.models.logits_view import LogitsView
from lmkd.rollout.base import Rollout, SamplingSpec, eos_ids, prompts_from_batch, rollouts_to_samples, sampling_from_cfg
from lmkd.rollout.vllm_colocate import VLLMColocate
from lmkd.train.builder import build_runtime


def _rt(base_recipe, **rollout):
    base_recipe["rollout"] = {"type": "hf", **rollout}
    return build_runtime(recipe_from_dict(base_recipe))


def _sample(ids, plen):
    return {
        "student.input_ids": ids,
        "student.attention_mask": [1] * len(ids),
        "student.labels": [-100] * plen + ids[plen:],
        "student.prompt_len": plen,
    }


def test_prompts_from_batch(base_recipe):
    rt = build_runtime(recipe_from_dict(base_recipe))
    b = rt.collator([_sample([5, 6, 7, 8, 9], 3), _sample([4, 1, 8], 2)])
    assert prompts_from_batch(b) == [[5, 6, 7], [4, 1]]  # eos(1) inside the prompt is kept, no pad
    b["student.attention_mask"][1, 0] = 0
    with pytest.raises(ValueError, match="left-padded"):
        prompts_from_batch(b)


def test_sampling_from_cfg():
    assert sampling_from_cfg(None) == SamplingSpec()
    assert sampling_from_cfg({"n": 3}).n == 3
    with pytest.raises(ValueError, match="valid keys"):
        sampling_from_cfg({"bogus": 1})


def test_eos_ids():
    class T:
        eos_token_id = 1

    assert eos_ids(T(), [1, 5, 5]) == [1, 5]
    T.eos_token_id = None
    assert eos_ids(T(), [7]) == [7]


def test_hf_generate_shapes(base_recipe):
    rt = _rt(base_recipe, batch_size=2)
    rt.student_model.train()
    prompts = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    outs = rt.rollout.generate(prompts, SamplingSpec(n=2, max_new_tokens=5, temperature=1.0, seed=0))
    assert len(outs) == 3
    for p, rs in zip(prompts, outs):
        assert len(rs) == 2
        for r in rs:
            assert r.prompt_ids == p and 1 <= len(r.response_ids) <= 5
            assert r.finish_reason in ("stop", "length")
            assert len(r.response_logp) == len(r.response_ids)
    assert rt.student_model.training  # train mode restored


def test_hf_eos_truncation_pad_eq_eos(base_recipe, tiny_models):
    src = Path(tiny_models["student"])
    dst = src.parent / "student_pad_eos"
    shutil.copytree(src, dst, dirs_exist_ok=True)
    _make_tokenizer(dst, eos_is_pad=True)
    base_recipe["models"]["student"]["path"] = str(dst)
    rt = _rt(base_recipe)
    assert rt.student_tokenizer.pad_token_id == rt.student_tokenizer.eos_token_id == 1
    prompts = [[5, 1, 7], [8, 9]]  # eos in the middle of a prompt
    outs = rt.rollout.generate(prompts, SamplingSpec(n=8, max_new_tokens=12, temperature=1.5, seed=1))
    assert outs[0][0].prompt_ids == [5, 1, 7]
    seen_stop = False
    for rs in outs:
        for r in rs:
            if r.finish_reason == "stop":
                seen_stop = True
                assert r.response_ids[-1] == 1 and 1 not in r.response_ids[:-1]
            else:
                assert len(r.response_ids) == 12
    assert seen_stop
    # labels stay target-aligned and keep every response token incl. the final eos (pad == eos)
    rollouts = [r for rs in outs for r in rs]
    b = rt.collator(rollouts_to_samples(rollouts))
    for i, r in enumerate(rollouts):
        plen, n = len(r.prompt_ids), len(r.response_ids)
        assert b["student.prompt_len"][i].item() == plen
        assert b["student.labels"][i, :plen].eq(-100).all()
        assert b["student.labels"][i, plen : plen + n].tolist() == r.response_ids
        assert b["student.labels"][i, plen + n :].eq(-100).all()  # padding (== eos) is never supervised
        assert b["student.attention_mask"][i].sum().item() == plen + n


def test_hf_response_logp_matches_forward(base_recipe):
    rt = _rt(base_recipe)
    outs = rt.rollout.generate(
        [[5, 6, 7], [8, 9]], SamplingSpec(n=2, max_new_tokens=6, temperature=1.0, top_p=1.0, seed=3)
    )
    rollouts = [r for rs in outs for r in rs]
    b = rt.collator(rollouts_to_samples(rollouts))
    rt.student_model.eval()
    with torch.no_grad():
        h = rt.student_model(
            input_ids=b["student.input_ids"], attention_mask=b["student.attention_mask"], output_hidden_states=True
        ).hidden_states[-1]
    view = LogitsView(h, rt.student_adapter.lm_head(rt.student_model))
    lp = view.target_logp(b["student.labels"])
    m = b["student.labels"] != -100
    assert m.any()
    assert (lp - b["rollout.rollout_logp"]).abs()[m].max() < 1e-4


def test_rollouts_to_samples_labels():
    r = Rollout([5, 6, 7], [8, 1], [-0.5, -0.1])
    s = rollouts_to_samples([r], copy_meta=[{"meta.solution": "x", "other": 1, "meta.problem": "p"}])[0]
    assert s["student.labels"] == [-100, -100, -100, 8, 1] and s["student.prompt_len"] == 3
    assert s["meta.solution"] == "x" and s["meta.problem"] == "p" and "other" not in s
    assert s["rollout.rollout_logp"] == [0.0, 0.0, 0.0, -0.5, -0.1]
    with pytest.raises(ValueError):
        rollouts_to_samples([r], copy_meta=[])


def test_hf_noops_and_registry(base_recipe):
    rt = _rt(base_recipe)
    rt.rollout.update_weights(iter([]))
    rt.rollout.sleep()
    rt.rollout.wake_up()
    rt.rollout.reset_prefix_cache()


def test_hf_top_k_disabled_not_model_default(base_recipe):
    """top_k<=0 must pass top_k=0 (off); None would fall back to generation_config (20/50 for real models)."""
    rt = _rt(base_recipe)
    seen = {}
    real = rt.student_model.generate

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    rt.student_model.generate = spy
    rt.rollout.generate([[5, 6]], SamplingSpec(max_new_tokens=2, top_k=-1))
    assert seen["top_k"] == 0
    rt.rollout.generate([[5, 6]], SamplingSpec(max_new_tokens=2, top_k=7))
    assert seen["top_k"] == 7


def test_hf_seed_does_not_reset_global_rng(base_recipe):
    rt = _rt(base_recipe)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    rt.rollout.generate([[5, 6, 7]], SamplingSpec(n=2, max_new_tokens=4, seed=9))
    assert torch.equal(torch.rand(3), expected)  # generation seeded itself without touching the trainer RNG
    a = rt.rollout.generate([[5, 6, 7]], SamplingSpec(n=2, max_new_tokens=4, seed=9))
    b = rt.rollout.generate([[5, 6, 7]], SamplingSpec(n=2, max_new_tokens=4, seed=9))
    assert [r.response_ids for r in a[0]] == [r.response_ids for r in b[0]]


def test_vllm_colocate_config_guards():
    rt = SimpleNamespace(
        config=SimpleNamespace(
            models=SimpleNamespace(student=SimpleNamespace(lora={"r": 4}, path="x", tokenizer=None))
        ),
        student_tokenizer=None,
        engine=SimpleNamespace(world_size=1, rank=0),
    )
    with pytest.raises(ValueError, match="sleep_level=2"):
        VLLMColocate(VLLMColocate.Config(sleep_level=2, lora_mode="hot_load")).setup(rt)
    with pytest.raises(ValueError, match="lora_mode"):
        VLLMColocate(VLLMColocate.Config(lora_mode="bogus")).setup(rt)
