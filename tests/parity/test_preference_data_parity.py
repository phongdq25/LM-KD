"""Parity with Awesome-Human-AI-Alignment preference data functions."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from lmkd.data.context import DataContext
from lmkd.data.transforms.listwise import TokenizeListwise
from lmkd.losses.functional.sequence import sequence_logps
from lmkd.models.logits_view import LogitsView
from lmkd.pipeline.stages.preference import compress_target_logits, sparse_advantages

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Awesome-Human-AI-Alignment-main" / "src"))
from human_alignment.config import PreferenceDistillationConfig  # noqa: E402
from human_alignment.integrations.preference_distillation import _tokenize_response  # noqa: E402
from human_alignment.mechanisms.training.distillation_losses import (  # noqa: E402
    causal_token_log_probs,
    compress_probabilities,
)
from human_alignment.supervision._preference_math import compressed_log_ratio_advantages  # noqa: E402


@pytest.mark.parity
def test_tokenize_parity(tiny_models):
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    prompt, response = "w0 w1 w2 w3", " w4 w5"
    cfg = TokenizeListwise.Config(max_length=6, max_prompt_length=3, format_plain_prompts=False)
    ours = TokenizeListwise(cfg)(
        {"prompt": prompt, "responses": [response, " w6"]},
        DataContext(tok, max_length=6),
    )["group.members"][0]
    original = _tokenize_response(
        tok,
        prompt,
        response,
        PreferenceDistillationConfig(
            max_length=6,
            max_prompt_length=3,
        ),
    )
    assert ours["student.input_ids"] == original["input_ids"]
    assert ours["student.labels"][1:] == original["labels"][1:]


@pytest.mark.parity
@pytest.mark.parametrize("average", [False, True])
def test_sequence_logps_parity(average):
    torch.manual_seed(3)
    logits = torch.randn(3, 7, 11)
    labels = torch.randint(0, 11, (3, 7))
    labels[:, :2] = -100
    labels[1, 5:] = -100
    ours, token, mask = sequence_logps(LogitsView(logits=logits), labels, average=average)
    ref, ref_token, ref_mask = causal_token_log_probs(logits, labels, average=average)
    torch.testing.assert_close(ours, ref, atol=1e-6, rtol=0)
    torch.testing.assert_close(token[:, 1:], ref_token, atol=1e-6, rtol=0)
    assert torch.equal(mask[:, 1:], ref_mask)


@pytest.mark.parity
def test_compress_parity():
    torch.manual_seed(4)
    logits = torch.randn(2, 5, 13)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 2, 3, -100, 4]])
    ids, values, tail = compress_target_logits(LogitsView(logits=logits), labels, 4)
    for row in range(2):
        valid = labels[row, 1:].ne(-100)
        ref = compress_probabilities(logits[row, :-1][valid], top_k=4)
        for target, item in zip(labels[row].ne(-100).nonzero().flatten().tolist(), ref):
            assert ids[row, target].tolist() == item["indices"]
            torch.testing.assert_close(values[row, target], torch.tensor(item["values"]), atol=1e-6, rtol=0)
            assert abs(tail[row, target].item() - item["remaining_probs_sum"]) <= 1e-6


@pytest.mark.parity
def test_adv_sparse_parity():
    torch.manual_seed(5)
    teacher = torch.randn(1, 4, 13)
    reference = torch.randn(1, 4, 13)
    labels = torch.tensor([[-100, 1, 2, 3]])
    t = compress_target_logits(LogitsView(logits=teacher), labels, 3)
    r = compress_target_logits(LogitsView(logits=reference), labels, 3)
    ids, values, _ = sparse_advantages(t, r, labels, max_k=6)
    ref_t = compress_probabilities(teacher[0, :-1], top_k=3)
    ref_r = compress_probabilities(reference[0, :-1], top_k=3)
    ref = compressed_log_ratio_advantages(ref_t, ref_r)
    for target, item in enumerate(ref, start=1):
        assert ids[0, target, : len(item["indices"])].tolist() == item["indices"]
        torch.testing.assert_close(
            values[0, target, : len(item["values"])], torch.tensor(item["values"]), atol=1e-6, rtol=0
        )
