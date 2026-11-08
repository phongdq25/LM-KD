"""vLLM prompt-logprob answer gain on the local tiny model."""

from __future__ import annotations

import pytest
import torch

from baselines.longCoT.sgl_spectral import AnswerGain
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl, write_jsonl
from lmkd.pipeline.stage import StageContext


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_answer_gain_vllm_matches_hf(tmp_path, tiny_models) -> None:
    data = Artifact("data", "dataset", tmp_path / "data")
    write_jsonl(
        data.path / "samples.jsonl",
        [
            {
                "meta.id": "a",
                "student.input_ids": [3, 4, 5, 6, 7],
                "seq.response_span": [2, 5],
                "seq.steps": [[2, 3], [3, 5]],
                "meta.response_text": "w3 \\boxed{w4}",
            }
        ],
    )
    hf = AnswerGain(AnswerGain.Config(model=tiny_models["student"], engine="hf", max_model_len=128))
    vllm = AnswerGain(
        AnswerGain.Config(model=tiny_models["student"], engine="vllm", max_model_len=128, gpu_memory_utilization=0.1)
    )
    hf_out = Artifact("hf", "signals", tmp_path / "hf")
    vllm_out = Artifact("vllm", "signals", tmp_path / "vllm")
    hf.run(StageContext(rt=None, inputs={"data": data}, output=hf_out))
    vllm.run(StageContext(rt=None, inputs={"data": data}, output=vllm_out))
    expected = read_jsonl(hf_out.path / "signals.jsonl")[0]["step_gain"]
    actual = read_jsonl(vllm_out.path / "signals.jsonl")[0]["step_gain"]
    assert actual == pytest.approx(expected, abs=1e-3)
