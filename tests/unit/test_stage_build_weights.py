"""Signal joining and transform-chain tests for build_weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from baselines.longCoT.sgl_spectral import BuildWeights
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl, write_jsonl
from lmkd.pipeline.stage import StageContext


def _art(tmp_path, name: str) -> Artifact:
    path = tmp_path / name
    path.mkdir()
    return Artifact(name, name, path)


def test_join_and_chain(tmp_path) -> None:
    data, sig, out = (_art(tmp_path, name) for name in ("data", "signals", "output"))
    samples = [
        {
            "meta.id": str(i),
            "student.input_ids": [1, 2, 3, 4],
            "student.labels": [-100, 2, 3, 4],
            "seq.response_span": [1, 4],
            "seq.steps": [[1, 2], [2, 3], [3, 4]],
        }
        for i in range(3)
    ]
    write_jsonl(data.path / "samples.jsonl", samples)
    write_jsonl(
        sig.path / "signals.jsonl",
        [{"meta.id": str(i), "step_strength": [0.8, 0.1, 0.1], "step_entropy": [1.0, 2.0, 3.0]} for i in range(3)],
    )
    chain = [{"type": "select_energy_top_p", "p": 0.7}, {"type": "tail_boost", "boost": 2.0}]
    stage = BuildWeights(BuildWeights.Config(chain=chain))
    ctx = StageContext(rt=SimpleNamespace(data_ctx=None), inputs={"data": data, "signals": sig}, output=out)
    info = stage.run(ctx)
    rows = read_jsonl(out.path / "samples.jsonl")
    assert info["n"] == 3
    assert info["selection_stats"]["step_retention"] == pytest.approx(1 / 3)
    assert info["selection_stats"]["token_retention"] == pytest.approx(1 / 3)
    assert all(row["seq.loss_mask"] == [0, 1, 0, 0] for row in rows)
    assert all(len(row["seq.token_weights"]) == 4 for row in rows)

    write_jsonl(
        sig.path / "signals.jsonl",
        [
            {"meta.id": "0", "step_strength": [1, 0, 0], "step_entropy": [1, 2, 3]},
            {"meta.id": "1", "step_strength": [1, 0, 0], "step_entropy": [1, 2, 3]},
        ],
    )
    with pytest.raises(ValueError, match="2"):
        stage.run(ctx)
    dropping = BuildWeights(BuildWeights.Config(chain=chain, missing="drop"))
    assert dropping.run(ctx)["n"] == 2


def test_skip_transform_still_runs_independent_steps(tmp_path) -> None:
    data, sig, out = (_art(tmp_path, name) for name in ("data", "signals", "output"))
    write_jsonl(
        data.path / "samples.jsonl",
        [
            {"meta.id": sid, "student.input_ids": [1, 2, 3], "student.labels": [-100, 2, 3], "seq.loss_mask": [0, 1, 1]}
            for sid in ("a", "b")
        ],
    )
    write_jsonl(sig.path / "signals.jsonl", [{"meta.id": "a", "step_strength": [1.0]}])
    stage = BuildWeights(
        BuildWeights.Config(chain=[{"type": "tail_boost", "tail_tokens": 1}], missing="skip_transform")
    )
    stage.run(StageContext(rt=SimpleNamespace(data_ctx=None), inputs={"data": data, "signals": sig}, output=out))
    rows = read_jsonl(out.path / "samples.jsonl")
    assert len(rows) == 2
    assert all("seq.token_weights" in row for row in rows)


def test_invalid_chain_rejected_when_stage_config_is_built() -> None:
    with pytest.raises(ValueError, match="boost_factor"):
        BuildWeights.Config(chain=[{"type": "tail_boost", "boost_factor": 3.0}])
