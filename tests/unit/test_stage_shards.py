"""Every T19 stage writes only its shard and merges in original id order."""

from __future__ import annotations

import json

from baselines.longCoT.palign import Generate, PAlignSearch, VerifyFilter
from baselines.longCoT.sgl_spectral import AnswerGain, BuildWeights
from baselines.longCoT.ssft import IGAttribution
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl, write_jsonl
from lmkd.pipeline.stage import StageContext


def _art(tmp_path, name):
    return Artifact(name, name, tmp_path / name)


def _compare(stage, inputs, tmp_path, filename):
    whole = _art(tmp_path, "whole")
    stage.run(StageContext(rt=None, inputs=inputs, output=whole))
    shards = []
    for index in range(2):
        art = _art(tmp_path, f"shard{index}")
        info = stage.run(StageContext(rt=None, inputs=inputs, output=art, shard=(index, 2)))
        (art.path / "manifest.json").write_text(json.dumps({"info": info}))
        shards.append(art.path)
    merged = _art(tmp_path, "merged")
    stage.merge(StageContext(rt=None, inputs=inputs, output=merged), shards)
    assert read_jsonl(merged.path / filename) == read_jsonl(whole.path / filename)


def test_answer_gain_shard_merge(tmp_path, tiny_models) -> None:
    data = _art(tmp_path, "data")
    rows = [
        {
            "meta.id": str(i),
            "student.input_ids": [3, 4, 5, 6, 7],
            "seq.response_span": [2, 5],
            "seq.steps": [[2, 3], [3, 5]],
            "meta.response_text": "w3 \\boxed{w4}",
        }
        for i in range(3)
    ]
    write_jsonl(data.path / "samples.jsonl", rows)
    stage = AnswerGain(AnswerGain.Config(model=tiny_models["student"], engine="hf", max_model_len=128))
    _compare(stage, {"data": data}, tmp_path, "signals.jsonl")


def test_ig_shard_merge(tmp_path, tiny_models) -> None:
    data = _art(tmp_path, "data")
    rows = [
        {"meta.id": str(i), "meta.question": "w1", "meta.answer": "w4", "meta.segments": ["w2 ", "w3"]}
        for i in range(3)
    ]
    write_jsonl(data.path / "samples.jsonl", rows)
    stage = IGAttribution(
        IGAttribution.Config(
            model=tiny_models["student"], dtype="fp32", steps=2, batch_size=2, gradient_checkpointing=False
        )
    )
    _compare(stage, {"data": data}, tmp_path, "signals.jsonl")


def test_palign_shard_merge(tmp_path, tiny_models, monkeypatch) -> None:
    data = _art(tmp_path, "data")
    write_jsonl(
        data.path / "records.jsonl",
        [
            {"meta.id": str(i), "question": "Q", "answer": "w4", "deepseek_thinking_trajectory": "w1. w2. w3."}
            for i in range(3)
        ],
    )

    def judge(self, prompts, model, tokenizer):
        return ["[ENOUGH]" if p.count("w") >= 2 else "[NOT_ENOUGH]" for p in prompts]

    monkeypatch.setattr(PAlignSearch, "_judge", judge)
    stage = PAlignSearch(PAlignSearch.Config(model=tiny_models["student"], engine="hf"))
    _compare(stage, {"data": data}, tmp_path, "records.jsonl")


def test_generate_shard_merge(tmp_path, tiny_models) -> None:
    data = _art(tmp_path, "data")
    write_jsonl(data.path / "records.jsonl", [{"meta.id": str(i), "prompt": "w1 w2"} for i in range(3)])
    stage = Generate(
        Generate.Config(
            model=tiny_models["student"],
            engine="hf",
            chat_template=False,
            sampling={"temperature": 0.0, "max_new_tokens": 2},
        )
    )
    _compare(stage, {"prev": data}, tmp_path, "records.jsonl")


def test_verify_shard_merge(tmp_path) -> None:
    data = _art(tmp_path, "data")
    write_jsonl(
        data.path / "records.jsonl",
        [
            {"meta.id": str(i), "question": "Q", "prefix": "P.", "answer": "yes", "output": "yes" if i != 1 else "no"}
            for i in range(3)
        ],
    )
    stage = VerifyFilter(VerifyFilter.Config(grader="exact_match"))
    _compare(stage, {"prev": data}, tmp_path, "records.jsonl")


def test_build_weights_shard_merge(tmp_path) -> None:
    data, signals = _art(tmp_path, "data"), _art(tmp_path, "signals")
    write_jsonl(
        data.path / "samples.jsonl",
        [
            {
                "meta.id": str(i),
                "student.input_ids": [1, 2, 3],
                "student.labels": [-100, 2, 3],
                "seq.response_span": [1, 3],
                "seq.steps": [[1, 2], [2, 3]],
            }
            for i in range(3)
        ],
    )
    write_jsonl(
        signals.path / "signals.jsonl",
        [{"meta.id": str(i), "step_strength": [0.8, 0.2], "step_entropy": [1.0, 2.0]} for i in range(3)],
    )
    stage = BuildWeights(
        BuildWeights.Config(
            chain=[{"type": "select_energy_top_p", "p": 0.7}, {"type": "tail_boost", "boost": 2.0}]
        )
    )
    _compare(stage, {"data": data, "signals": signals}, tmp_path, "samples.jsonl")
