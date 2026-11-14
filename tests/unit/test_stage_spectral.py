"""Spectral signal stage alignment, resume, and shard tests."""

from __future__ import annotations

import pytest
import torch

from baselines.longCoT.sgl_spectral import SpectralCapture
from baselines.longCoT.sgl_spectral.spectral import analytic_gradients_and_entropies, effective_rank
from lmkd.core.config import ModelSpec
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.loading import load_model
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl, write_jsonl
from lmkd.pipeline.stage import StageContext


def _data(tmp_path, count=4):
    folder = tmp_path / "data"
    folder.mkdir()
    rows = [
        {
            "meta.id": str(i),
            "student.input_ids": [3, 4, 5, 6, 7, 8],
            "seq.response_span": [2, 6],
            "seq.steps": [[2, 4], [4, 6]],
        }
        for i in range(count)
    ]
    write_jsonl(folder / "samples.jsonl", rows)
    return Artifact("data", "data", folder), rows


def test_capture_resume(tmp_path, tiny_models, monkeypatch) -> None:
    data, _ = _data(tmp_path)
    output = Artifact("capture", "capture", tmp_path / "capture")
    stage = SpectralCapture(SpectralCapture.Config(model=tiny_models["student"], dtype="fp32", device="cpu"))
    calls = 0
    original = HFAutoAdapter.forward_hidden

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HFAutoAdapter, "forward_hidden", counted)
    ctx = StageContext(rt=None, inputs={"data": data}, output=output)
    assert stage.run(ctx)["computed"] == 4
    assert calls == 4
    rows = read_jsonl(output.path / "signals.jsonl")
    write_jsonl(output.path / "signals.jsonl", rows[:2])
    assert stage.run(ctx)["computed"] == 2
    assert calls == 6


def test_target_alignment(tmp_path, tiny_models) -> None:
    data, samples = _data(tmp_path, count=1)
    output = Artifact("capture", "capture", tmp_path / "capture")
    stage = SpectralCapture(SpectralCapture.Config(model=tiny_models["student"], dtype="fp32", device="cpu"))
    stage.run(StageContext(rt=None, inputs={"data": data}, output=output))
    actual = read_jsonl(output.path / "signals.jsonl")[0]
    model = load_model(ModelSpec(path=tiny_models["student"], dtype="fp32"), trainable=False)
    adapter = HFAutoAdapter(HFAutoAdapter.Config())
    ids = torch.tensor(samples[0]["student.input_ids"])[None]
    with torch.no_grad():
        hidden = adapter.forward_hidden(model, ids, None).last_hidden_state[0]
        weight = adapter.lm_head(model).weight.float()
        grad, _ = analytic_gradients_and_entropies(hidden[1:5], ids[0, 2:6], weight)
        u, s, _ = torch.linalg.svd(grad)
    k = effective_rank(s)
    leverage = u[:, :k].square().sum(-1)
    expected = [float(leverage[:2].mean()), float(leverage[2:].mean())]
    assert actual["step_strength"] == pytest.approx(expected, abs=1e-5)


def test_shard_merge_matches_whole(tmp_path, tiny_models) -> None:
    data, rows = _data(tmp_path)
    stage = SpectralCapture(SpectralCapture.Config(model=tiny_models["student"], dtype="fp32", device="cpu"))
    whole = Artifact("whole", "capture", tmp_path / "whole")
    stage.run(StageContext(rt=None, inputs={"data": data}, output=whole))
    dirs = [tmp_path / "shard0", tmp_path / "shard1"]
    for index, folder in enumerate(dirs):
        stage.run(
            StageContext(rt=None, inputs={"data": data}, output=Artifact("shard", "capture", folder), shard=(index, 2))
        )
    merged = Artifact("merged", "capture", tmp_path / "merged")
    stage.merge(StageContext(rt=None, inputs={"data": data}, output=merged), dirs)
    assert [r["meta.id"] for r in read_jsonl(merged.path / "signals.jsonl")] == [r["meta.id"] for r in rows]
    assert read_jsonl(merged.path / "signals.jsonl") == read_jsonl(whole.path / "signals.jsonl")
