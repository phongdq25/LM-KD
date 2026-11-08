"""HF generation, filtering, and artifact-record source tests."""

from __future__ import annotations

import pytest

from baselines.longCoT.common import engine_name as _engine_name
from baselines.longCoT.palign import Generate, VerifyFilter
from lmkd.data.sources import ArtifactRecords
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl, write_jsonl
from lmkd.pipeline.stage import StageContext
from lmkd.pipeline.stages.prepare import PrepareDataset


def test_generate_hf_resumes(tmp_path, tiny_models) -> None:
    data = Artifact("data", "text", tmp_path / "data")
    out = Artifact("gen", "text", tmp_path / "gen")
    write_jsonl(data.path / "records.jsonl", [{"meta.id": "a", "prompt": "w1 w2"}])
    stage = Generate(
        Generate.Config(
            model=tiny_models["student"],
            engine="hf",
            chat_template=False,
            sampling={"temperature": 0.0, "max_new_tokens": 2},
        )
    )
    ctx = StageContext(rt=None, inputs={"prev": data}, output=out)
    stage.run(ctx)
    first = read_jsonl(out.path / "records.jsonl")
    assert len(first) == 1 and "output" in first[0]
    stage.run(ctx)
    assert read_jsonl(out.path / "records.jsonl") == first


def test_unknown_generate_engine_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown generation engine"):
        _engine_name("unknown")


def test_generate_multiple_candidates_have_unique_ids(tmp_path, tiny_models) -> None:
    data = Artifact("data", "text", tmp_path / "data")
    out = Artifact("gen", "text", tmp_path / "gen")
    write_jsonl(data.path / "records.jsonl", [{"meta.id": "a", "prompt": "w1 w2"}])
    stage = Generate(
        Generate.Config(
            model=tiny_models["student"],
            engine="hf",
            chat_template=False,
            n=2,
            sampling={"temperature": 0.8, "max_new_tokens": 2, "seed": 0},
        )
    )
    stage.run(StageContext(rt=None, inputs={"prev": data}, output=out))
    rows = read_jsonl(out.path / "records.jsonl")
    assert [row["meta.id"] for row in rows] == ["a#0", "a#1"]
    assert all(row["meta.source_id"] == "a" for row in rows)


def test_verify_filter_and_artifact_source(tmp_path) -> None:
    data = Artifact("data", "text", tmp_path / "data")
    out = Artifact("verify", "text", tmp_path / "verify")
    write_jsonl(
        data.path / "records.jsonl",
        [
            {"meta.id": "a", "question": "Q", "prefix": "First.", "answer": "yes", "output": "yes"},
            {"meta.id": "b", "question": "Q", "prefix": "First.", "answer": "yes", "output": "no"},
            {"meta.id": "c", "question": "Q", "prefix": "First.", "answer": "", "output": "yes"},
        ],
    )
    stage = VerifyFilter(VerifyFilter.Config(grader="exact_match"))
    info = stage.run(StageContext(rt=None, inputs={"prev": data}, output=out))
    assert info == {"kept": 1, "dropped": {"no_gold": 1, "wrong": 1}}
    source = ArtifactRecords(ArtifactRecords.Config(node="verify"))
    source.bind(out)
    rows = list(source.records())
    assert rows[0]["output"] == "<Begin_of_Prefix>First.<End_of_Prefix>\nyes"
    prepared = Artifact("prepared", "dataset", tmp_path / "prepared")
    stage2 = PrepareDataset(PrepareDataset.Config(source={"type": "artifact_records", "node": "verify"}))
    info2 = stage2.run(StageContext(rt=type("RT", (), {"data_ctx": None})(), inputs={"prev": out}, output=prepared))
    assert info2["n"] == 1
    assert read_jsonl(prepared.path / "samples.jsonl")[0]["output"] == rows[0]["output"]
