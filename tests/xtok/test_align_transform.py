from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from lmkd.core.config import recipe_from_dict
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ALIGNERS, TRANSFORMS
from lmkd.core.validate import check_recipe
from lmkd.data.aligners.base import Aligner
from lmkd.data.collator import Collator
from lmkd.data.transforms.base import run_transforms
from lmkd.pipeline.runner import build_runtime_for_pipeline, run_pipeline


def _collator(ctx):
    return Collator(
        {"student": ctx.student_tokenizer.pad_token_id, "teacher": ctx.teacher_tokenizer.pad_token_id},
        pad_to_multiple_of=None,
    )


def test_align_in_memory_and_collate(xtok_records, xtok_ctx):
    tfs = [
        TRANSFORMS.build(
            {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"}
        ),
        TRANSFORMS.build({"type": "token_offsets", "teacher": True}),
        TRANSFORMS.build({"type": "align", "aligner": "exact_token"}),
    ]
    samples = run_transforms(xtok_records[:6], tfs, xtok_ctx)
    batch = _collator(xtok_ctx)(samples)
    t_s = batch["student.input_ids"].shape[1]
    t_t = batch["teacher.input_ids"].shape[1]
    assert batch["align.eta_t_for_s"].shape == (6, t_s)
    assert batch["align.eta_s_for_t"].shape == (6, t_t)
    assert batch["student.offsets"].shape == (6, t_s, 2) and batch["teacher.offsets"].shape == (6, t_t, 2)
    for b, s in enumerate(samples):
        n = len(s["student.input_ids"])
        assert (batch["align.eta_t_for_s"][b, n:] == -1).all()  # pad value of the FieldSpec
        assert (batch["student.offsets"][b, n:] == -1).all()


def test_token_offsets_drop_and_raise(xtok_records, xtok_ctx):
    base = TRANSFORMS.build({"type": "tokenize_sft", "use_chat_template": False})
    s = base(dict(xtok_records[0]), xtok_ctx)
    s["student.input_ids"] = [int(i) % 5 + 3 for i in s["student.input_ids"]]  # break the round trip
    before = xtok_ctx.extra.get("offsets_dropped", 0)
    assert TRANSFORMS.build({"type": "token_offsets"})(dict(s), xtok_ctx) is None
    assert xtok_ctx.extra["offsets_dropped"] == before + 1
    with pytest.raises(ValueError, match="ids_mismatch"):
        TRANSFORMS.build({"type": "token_offsets", "on_mismatch": "raise"})(dict(s), xtok_ctx)


def test_tokenize_sft_teacher_prompt_text(xtok_records, xtok_ctx):
    rec = dict(xtok_records[1])
    same = TRANSFORMS.build(
        {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"}
    )(dict(rec), xtok_ctx)
    own = TRANSFORMS.build({"type": "tokenize_sft", "use_chat_template": False, "teacher": True})(dict(rec), xtok_ctx)
    assert same["meta.teacher_prompt_text"] == same["meta.prompt_text"] == own["meta.teacher_prompt_text"]
    assert same["teacher.input_ids"] == own["teacher.input_ids"]  # no chat template here -> identical
    with pytest.raises(ValueError):
        TRANSFORMS.build({"type": "tokenize_sft", "teacher_prompt_text": "other"})


def _recipe(base_recipe, xtok_models, transforms):
    base_recipe["models"]["student"] = {"path": xtok_models["student"], "dtype": "fp32"}
    base_recipe["models"]["teacher"] = {"path": xtok_models["teacher"], "dtype": "fp32"}
    base_recipe["data"]["transforms"] = transforms
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "pretokenized"}
    base_recipe["loss"]["terms"] = [{"criterion": "nll"}]
    return base_recipe


def test_check_requires_offsets(base_recipe, xtok_models):
    tok = {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"}
    bad = recipe_from_dict(_recipe(base_recipe, xtok_models, [tok, {"type": "align", "aligner": "char_span"}]))
    rep = check_recipe(bad)
    assert any("student.offsets" in e for e in rep.errors), rep.errors
    good = recipe_from_dict(
        _recipe(
            base_recipe,
            xtok_models,
            [
                tok,
                {"type": "load_vocab_map"},
                {"type": "token_offsets", "teacher": True},
                {"type": "align", "aligner": "char_span"},
                {"type": "align", "aligner": "exact_token"},
            ],
        )
    )
    assert check_recipe(good).ok


def test_xtok_base_recipe_checks():
    from pathlib import Path

    from lmkd.core.config import load_recipe

    path = Path(__file__).resolve().parents[2] / "recipes/cross_tokenizer/_base_xtok.yaml"
    cfg = load_recipe(str(path))
    names = [t["type"] if isinstance(t, dict) else t for t in cfg.data.transforms]
    assert names == ["tokenize_sft", "load_vocab_map", "token_offsets", "align"]


# --------------------------------------------------------------------------- pipeline
def _pipeline_cfg(base_recipe, xtok_models, toy_jsonl, tmp_path, workers=0):
    base_recipe = _recipe(base_recipe, xtok_models, [])
    base_recipe["train"]["output_dir"] = str(tmp_path / "run")
    base_recipe["pipeline"] = [
        {
            "stage": "prepare_dataset",
            "name": "data",
            "source": {"type": "jsonl", "path": toy_jsonl},
            "transforms": [
                {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"},
                {"type": "token_offsets", "teacher": True},
            ],
        },
        {"stage": "vocab_map", "name": "vm"},
        {
            "stage": "align_precompute",
            "name": "aligned",
            "inputs": {"data": "data"},
            "num_workers": workers,
            "aligner": {"type": "exact_token"},
        },
    ]
    return recipe_from_dict(base_recipe)


def test_align_precompute_cached(base_recipe, xtok_models, toy_jsonl, tmp_path, monkeypatch):
    from lmkd.pipeline.stages.xtok import AlignPrecompute, VocabMapStage

    calls = {"align": 0, "vm": 0}
    for cls, key in ((AlignPrecompute, "align"), (VocabMapStage, "vm")):
        orig = cls.run

        def counted(self, ctx, _orig=orig, _key=key):
            calls[_key] += 1
            return _orig(self, ctx)

        monkeypatch.setattr(cls, "run", counted)
    cfg = _pipeline_cfg(base_recipe, xtok_models, toy_jsonl, tmp_path)
    rt = build_runtime_for_pipeline(cfg)
    samples = run_pipeline(rt)
    assert calls == {"align": 1, "vm": 1}
    assert len(samples) == 32 and "align.eta_t_for_s" in samples[0]
    assert rt.data_ctx.vocab_map is not None
    s = samples[0]
    assert len(s["align.eta_t_for_s"]) == len(s["student.input_ids"])

    rt2 = build_runtime_for_pipeline(cfg)
    assert rt2.data_ctx.vocab_map is None
    samples2 = run_pipeline(rt2)
    assert calls == {"align": 1, "vm": 1}  # everything cached ...
    assert rt2.data_ctx.vocab_map is not None  # ... but attach() still loaded the map
    assert [x["align.eta_t_for_s"] for x in samples2] == [x["align.eta_t_for_s"] for x in samples]


def test_align_precompute_workers_equal(base_recipe, xtok_models, toy_jsonl, tmp_path):
    cfg0 = _pipeline_cfg(base_recipe, xtok_models, toy_jsonl, tmp_path / "a", workers=0)
    out0 = run_pipeline(build_runtime_for_pipeline(cfg0))
    cfg2 = _pipeline_cfg(base_recipe, xtok_models, toy_jsonl, tmp_path / "b", workers=2)
    out2 = run_pipeline(build_runtime_for_pipeline(cfg2))
    assert [s["align.eta_t_for_s"] for s in out0] == [s["align.eta_t_for_s"] for s in out2]


# --------------------------------------------------------------------------- on_fail
@ALIGNERS.register("_test_failing_aligner", force=True)
class _Failing(Aligner):
    """Raises ValueError for odd ids (test only)."""

    requires = ()
    provides = ("align.eta_t_for_s",)

    class Config(BaseConfig):
        pass

    def __call__(self, sample, ctx):
        if sample["meta.id"] == "r1":
            raise ValueError("boom")
        return {"align.eta_t_for_s": [0] * len(sample["student.input_ids"])}

    def empty(self, sample):
        return {"align.eta_t_for_s": [-1] * len(sample["student.input_ids"])}


def test_on_fail_drop_vs_empty(xtok_records, xtok_ctx):
    tok = TRANSFORMS.build({"type": "tokenize_sft", "use_chat_template": False})
    recs = [dict(r, **{"meta.id": r["id"]}) for r in xtok_records[:3]]
    base = run_transforms(recs, [tok], xtok_ctx)
    drop = run_transforms(base, [TRANSFORMS.build({"type": "align", "aligner": "_test_failing_aligner"})], xtok_ctx)
    assert len(drop) == 2
    keep = run_transforms(
        base, [TRANSFORMS.build({"type": "align", "aligner": "_test_failing_aligner", "on_fail": "empty"})], xtok_ctx
    )
    assert len(keep) == 3 and set(keep[1]["align.eta_t_for_s"]) == {-1}


# --------------------------------------------------------------------------- speed
def test_offline_alignment_speed_1000_samples(xtok_samples, xtok_ctx):
    pytest.importorskip("editdistance")
    from lmkd.data.vocab_map import VocabMap

    xtok_ctx.vocab_map = VocabMap.build(xtok_ctx.student_tokenizer, xtok_ctx.teacher_tokenizer)
    try:
        samples = [xtok_samples[i % len(xtok_samples)] for i in range(1000)]
        aligners = [
            ALIGNERS.build(a)
            for a in (
                "exact_token",
                "char_span",
                "lcs_span",
                {"type": "dtw_edit", "full_sequence": True},
            )
        ]
        t0 = time.time()
        for s in samples:
            for al in aligners:
                al(s, xtok_ctx)
        elapsed = time.time() - t0
    finally:
        xtok_ctx.vocab_map = None
    print(f"\n[T11] 4 aligners x 1000 toy samples: {elapsed:.2f}s")
    assert elapsed < 10


# --------------------------------------------------------------------------- misc
def test_load_vocab_map_transform_setup(xtok_models, xtok_tokenizers, tmp_path):
    stu, tea = xtok_tokenizers
    from lmkd.data.context import DataContext

    rt = SimpleNamespace(
        student_tokenizer=stu, teacher_tokenizer=tea, data_ctx=DataContext(student_tokenizer=stu, teacher_tokenizer=tea)
    )
    cache = tmp_path / "vm.pt"
    tf = TRANSFORMS.build({"type": "load_vocab_map", "cache_path": str(cache), "space_mask_mode": "space"})
    tf.setup(rt)
    assert cache.exists() and rt.data_ctx.vocab_map.space_mask_t is not None
    rt.data_ctx.vocab_map = None
    tf2 = TRANSFORMS.build({"type": "load_vocab_map", "cache_path": str(cache)})
    tf2.setup(rt)  # loaded from the cache file
    assert rt.data_ctx.vocab_map.space_mask_t is not None
    sample = {"a": 1}
    assert tf2(sample, rt.data_ctx) is sample
