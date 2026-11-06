"""Answer-gain HF scoring and probe-length behavior."""

from __future__ import annotations

from baselines.longCoT.sgl_spectral import AnswerGain
from lmkd.core.config import ModelSpec
from lmkd.models.loading import load_tokenizer
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl, write_jsonl
from lmkd.pipeline.stage import StageContext


def test_gain_hf_engine(tmp_path, tiny_models) -> None:
    tok = load_tokenizer(ModelSpec(path=tiny_models["student"]))
    ids = tok("w1 w2 w3 w4 w5", add_special_tokens=False)["input_ids"]
    row = {
        "meta.id": "a",
        "student.input_ids": ids,
        "seq.response_span": [2, 5],
        "seq.steps": [[2, 3], [3, 5]],
        "meta.response_text": "reason \\boxed{w5}",
    }
    data = Artifact("data", "data", tmp_path / "data")
    output = Artifact("gain", "gain", tmp_path / "gain")
    write_jsonl(data.path / "samples.jsonl", [row])
    stage = AnswerGain(AnswerGain.Config(model=tiny_models["student"], engine="hf", max_model_len=128))
    stage.run(StageContext(rt=None, inputs={"data": data}, output=output))
    gains = read_jsonl(output.path / "signals.jsonl")[0]["step_gain"]
    assert len(gains) == 2
    assert all(isinstance(g, float) for g in gains)
    probe_n = len(tok(stage.cfg.probe, add_special_tokens=False)["input_ids"])
    answer_n = len(tok("w5}", add_special_tokens=False)["input_ids"])
    stage.cfg.max_model_len = 3 + probe_n + answer_n + 1
    output2 = Artifact("gain2", "gain", tmp_path / "gain2")
    stage.run(StageContext(rt=None, inputs={"data": data}, output=output2))
    assert read_jsonl(output2.path / "signals.jsonl")[0]["step_gain"][-1] == 0.0
