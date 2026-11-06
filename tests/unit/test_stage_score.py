"""Offline preference score stage and PAD prompt parity."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2ForSequenceClassification

from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl, write_jsonl
from lmkd.pipeline.stage import StageContext
from lmkd.pipeline.stages.preference import Score, pad_mcq_template, pairwise_scores, score_record

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Awesome-Human-AI-Alignment-main" / "src"))
from human_alignment.integrations.distillation_data import _pad_mcq_template  # noqa: E402


class FixedModel:
    """Constant next-token logits for order-invariant A/B scoring."""

    def __init__(self, vocab):
        self.vocab = vocab

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab))


def test_pairwise_order_balanced(tiny_models):
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    tok.add_tokens(["A", "B"])
    model = FixedModel(len(tok))
    a = pairwise_scores(model, tok, "Q: w1", ["w2", "w3"], batch_size=2)
    b = pairwise_scores(model, tok, "Q: w1", ["w3", "w2"], batch_size=2)
    assert a == pytest.approx(b[::-1], abs=1e-6)


def test_pad_mcq_template_verbatim(tiny_models):
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    tok.chat_template = "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n{% endfor %}assistant:"
    ours = pad_mcq_template(tok)
    original = _pad_mcq_template(tok)
    params = {
        "identifier_list_str": "(A, or B)",
        "instruction": "Q?",
        "candidate_response_list_str": 'A) """\nX\n"""\nB) """\nY\n"""',
    }
    assert ours.safe_substitute(**params) == original.safe_substitute(**params)


def test_score_stage_sequence_and_resume(tiny_models, tmp_path):
    incoming = tmp_path / "in"
    incoming.mkdir()
    write_jsonl(incoming / "records.jsonl", [{"meta.id": "p1", "prompt": "Q: w1", "responses": ["A: w2", "A: w3"]}])
    outgoing = tmp_path / "out"
    stage = Score(Score.Config(mode="sequence_logprob", model=tiny_models["teacher"], batch_size=2))
    ctx = StageContext(
        rt=SimpleNamespace(),
        inputs={"data": Artifact("raw", "k", incoming, "in")},
        output=Artifact("score", "k", outgoing, "out"),
    )
    first = stage.run(ctx)
    second = stage.run(ctx)
    scored = read_jsonl(outgoing / "records.jsonl")
    assert first["computed"] == 1 and second["computed"] == 0
    assert len(scored[0]["teacher_scores"]) == 2
    assert all(torch.isfinite(torch.tensor(scored[0]["teacher_scores"])))


def test_pad_mcq_scores_and_avglogp(tiny_models):
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    tok.add_tokens(["A", "B"])
    tok.chat_template = "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n{% endfor %}assistant:"
    result = score_record(
        {"prompt": "Q?", "responses": [" w1", " w2"]}, FixedModel(len(tok)), tok, "pad_mcq", 2, "teacher_scores"
    )
    assert len(result["scores_mcq"]) == len(result["scores_avglogp"]) == 2
    assert torch.isfinite(torch.tensor(result["scores_mcq"])).all()


def test_score_stage_reward_model(tiny_models, tmp_path):
    model_dir = tmp_path / "reward"
    model_dir.mkdir()
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    tok.save_pretrained(model_dir)
    cfg = GPT2Config(
        vocab_size=len(tok), n_positions=64, n_embd=16, n_layer=1, n_head=2, num_labels=1, pad_token_id=tok.pad_token_id
    )
    GPT2ForSequenceClassification(cfg).save_pretrained(model_dir)
    incoming = tmp_path / "raw"
    incoming.mkdir()
    write_jsonl(incoming / "records.jsonl", [{"meta.id": "x", "prompt": "Q: w1", "responses": [" A: w2", " A: w3"]}])
    stage = Score(Score.Config(mode="reward_model", model=str(model_dir), batch_size=2))
    outgoing = tmp_path / "scored"
    ctx = StageContext(
        rt=SimpleNamespace(), inputs={"data": Artifact("raw", "k", incoming)}, output=Artifact("score", "k", outgoing)
    )
    stage.run(ctx)
    assert torch.isfinite(torch.tensor(read_jsonl(outgoing / "records.jsonl")[0]["rewards"])).all()
