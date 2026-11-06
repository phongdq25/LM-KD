"""IMPACT training, shared projection cache, and offline BI calibration."""

from __future__ import annotations

import copy
import json
import math

import torch

from lmkd.core.config import recipe_from_dict
from lmkd.core.registry import LAYER_MAPPERS
from lmkd.pipeline.runner import build_runtime_for_pipeline, run_pipeline
from lmkd.train.builder import build_runtime


def _recipe(base_recipe, xtok_models, tmp_path, *, with_dskd=False):
    recipe = copy.deepcopy(base_recipe)
    recipe["models"]["student"] = {"path": xtok_models["student"], "dtype": "fp32"}
    recipe["models"]["teacher"] = {"path": xtok_models["teacher"], "dtype": "fp32"}
    recipe["teacher"] = {"provider": "hf_frozen", "input": "pretokenized"}
    recipe["data"]["transforms"] = [
        {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"},
        {"type": "load_vocab_map", "cache_path": str(tmp_path / "vm.pt")},
        {"type": "token_offsets", "teacher": True},
        {"type": "align", "aligner": {"type": "boundary_interp"}},
    ]
    terms = [{"criterion": "nll", "weight": 1.0}]
    if with_dskd:
        recipe["data"]["transforms"].append({"type": "align", "aligner": {"type": "exact_token"}})
        terms.append({"criterion": "dskdv2_eta", "alias": "kd", "weight": 0.5})
    terms.append({"criterion": "impact", "weight": 1.0, "params": {"layer_mapper": {"type": "first", "k": 2}}})
    recipe["loss"] = {"terms": terms}
    recipe["train"].update(
        {"output_dir": str(tmp_path / "run"), "batch_size": 2, "micro_batch_size": 2, "max_steps": 20, "log_every": 1}
    )
    recipe["optim"]["lr"] = 5e-3
    return recipe


def test_impact_e2e(base_recipe, xtok_models, tmp_path):
    recipe = _recipe(base_recipe, xtok_models, tmp_path)
    rt = build_runtime(recipe_from_dict(recipe))
    batch = rt.collator([rt.train_dataset[0], rt.train_dataset[1]])
    rt.begin()
    for _ in range(20):
        rt.train_step(batch=batch)
    rt.end()
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    values = [row["loss/impact/impact"] for row in rows]
    assert len(values) == 20 and all(math.isfinite(v) for v in values)
    assert values[-1] < values[0]


def test_shared_cache_with_dskd(base_recipe, xtok_models, tmp_path, monkeypatch):
    from baselines.cross_tokenizer.impact import Impact

    seen: list[tuple[int, int, dict[int, int]]] = []
    original = Impact.forward

    def capture(self, ctx):
        calls: dict[int, int] = {}
        hidden = ctx.teacher.side.hidden

        def counted(layer):
            calls[layer] = calls.get(layer, 0) + 1
            return hidden(layer)

        ctx.teacher.side.hidden = counted
        output = original(self, ctx)
        original(self, ctx)  # a second criterion using the same per-step context reuses projections
        keys = [key for key in ctx.shared if isinstance(key, tuple) and key[0] == "impact_proj"]
        seen.append((len(keys), len(set(keys)), calls))
        return output

    monkeypatch.setattr(Impact, "forward", capture)
    recipe = _recipe(base_recipe, xtok_models, tmp_path, with_dskd=True)
    recipe["train"]["max_steps"] = 1
    rt = build_runtime(recipe_from_dict(recipe))
    rt.begin()
    rt.train_step()
    rt.end()
    assert seen and all(total == unique == 2 and calls == {1: 1, 2: 1} for total, unique, calls in seen)


def test_calibrate_stage_attach(base_recipe, xtok_models, toy_jsonl, tmp_path):
    recipe = _recipe(base_recipe, xtok_models, tmp_path)
    recipe["data"]["train"] = {"type": "jsonl", "path": str(toy_jsonl)}
    recipe["pipeline"] = [{"stage": "impact_calibrate", "samples": 4, "batch_size": 2}]
    rt = build_runtime_for_pipeline(recipe_from_dict(recipe))
    run_pipeline(rt)
    scores = rt.extra["impact_layer_scores"]
    assert isinstance(scores, torch.Tensor) and scores.shape == (3,)
    mapper = LAYER_MAPPERS.build({"type": "topk_bi", "k": 2})
    mapper.prepare(rt)
    assert mapper.state_dict()["frozen"] is not None
    rt2 = build_runtime_for_pipeline(recipe_from_dict(recipe))
    run_pipeline(rt2)
    torch.testing.assert_close(rt2.extra["impact_layer_scores"], scores)
