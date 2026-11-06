"""SSFT Integrated Gradients shape, normalization, and compact output."""

from __future__ import annotations

import pytest

from baselines.longCoT.ssft import IGAttribution
from baselines.longCoT.ssft.pipeline import _ig_example, _integrated_gradients
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl, write_jsonl
from lmkd.pipeline.stage import StageContext


def test_ig_shapes_and_norm(tmp_path, tiny_models) -> None:
    data = Artifact("data", "data", tmp_path / "data")
    out = Artifact("ig", "ig", tmp_path / "ig")
    sample = {"meta.id": "a", "meta.question": "w1 w2", "meta.answer": "w5", "meta.segments": ["w3 ", "w4"]}
    write_jsonl(data.path / "samples.jsonl", [sample])
    stage = IGAttribution(
        IGAttribution.Config(
            model=tiny_models["student"], dtype="fp32", gradient_checkpointing=False, steps=4, batch_size=2
        )
    )
    stage.run(StageContext(rt=None, inputs={"data": data}, output=out))
    compact = read_jsonl(out.path / "signals.jsonl")[0]["segment_ig"]
    assert len(compact) == 2
    model, tok, adapter = stage._prepare_model()
    ids, spans, answer = _ig_example(
        tok, "w1 w2", "w5", sample["meta.segments"], stage.cfg.user_template, stage.cfg.answer_template
    )
    attribution = _integrated_gradients(model, adapter, ids, answer, tok.pad_token_id, 4, 2)
    assert float(attribution.square().sum()) == pytest.approx(1.0, abs=1e-4)
    assert [row[0] for row in compact] == [b - a for a, b in spans]
