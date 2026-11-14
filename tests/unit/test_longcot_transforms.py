from __future__ import annotations

import math

import torch
from transformers import AutoTokenizer

from baselines.longCoT import RenderCoT, Segment, TokenizeCoT
from baselines.longCoT.sgl_spectral import Provenance, SelectEnergyTopP, StepTransitions, TailBoost
from baselines.longCoT.ssft import SelectCumulativeStrength
from lmkd.core.config import recipe_from_dict
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TRANSFORMS
from lmkd.core.validate import check_recipe
from lmkd.data.collator import Collator
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import CorpusTransform, run_transforms


def _make_dummy_ctx(tiny_models, max_length=128):
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    return DataContext(student_tokenizer=tok, max_length=max_length)


def test_tokenize_cot_drop_long(tiny_models):
    ctx = _make_dummy_ctx(tiny_models, max_length=10)
    transform = TokenizeCoT(TokenizeCoT.Config())
    sample = {
        "meta.prompt_text": "Please answer this math problem in detail.",
        "meta.response_text": "Here is a very long response that will certainly exceed ten tokens when tokenized.",
    }
    res = transform(sample, ctx)
    assert res is None


def test_energy_select_masks_stop_token(tiny_models):
    ctx = _make_dummy_ctx(tiny_models, max_length=128)
    # Prompt: 2 tokens (0, 1)
    # Response: 6 tokens (2, 3), (4, 5), (6, 7)
    # Stop: 2 tokens (8, 9)
    sample = {
        "student.input_ids": list(range(10)),
        "seq.response_span": [2, 8],
        "seq.steps": [(2, 4), (4, 6), (6, 8)],
        "signals.step_strength": [0.1, 0.8, 0.1],
    }
    transform = SelectEnergyTopP(SelectEnergyTopP.Config(p=0.5))
    out = transform(sample, ctx)

    assert out["meta.selected_steps"] == [1]
    mask = out["seq.loss_mask"]
    assert len(mask) == 10
    # Prompt: 0, 1 -> 0
    assert mask[0] == 0 and mask[1] == 0
    # Step 0: 2, 3 -> 0
    assert mask[2] == 0 and mask[3] == 0
    # Step 1 (selected): 4, 5 -> 1
    assert mask[4] == 1 and mask[5] == 1
    # Step 2: 6, 7 -> 0
    assert mask[6] == 0 and mask[7] == 0
    # Stop tokens: 8, 9 -> 1
    assert mask[8] == 1 and mask[9] == 1


def test_energy_zero_total_keeps_all(tiny_models):
    ctx = _make_dummy_ctx(tiny_models, max_length=128)
    sample = {
        "student.input_ids": list(range(8)),
        "seq.response_span": [2, 6],
        "seq.steps": [(2, 4), (4, 6)],
        "signals.step_strength": [0.0, 0.0],
    }
    transform = SelectEnergyTopP(SelectEnergyTopP.Config(p=0.95))
    out = transform(sample, ctx)
    assert out["meta.selected_steps"] == [0, 1]


def test_ssft_select_examples(tiny_models):
    ctx = _make_dummy_ctx(tiny_models, max_length=128)
    compact = [[4, 0.1, 0.1], [9, 0.9, 0.1], [1, 0.0, 0.0]]
    sample = {
        "student.input_ids": list(range(20)),
        "seq.steps": [(0, 4), (4, 13), (13, 14)],
        "signals.segment_ig": compact,
    }
    transform = SelectCumulativeStrength(
        SelectCumulativeStrength.Config(cumulative_ratio=0.7, coherence_max=0.8, always_keep=[0, -2, -1])
    )
    out = transform(sample, ctx)
    assert out is not None
    assert 0 in out["meta.selected_steps"]
    assert 1 in out["meta.selected_steps"]
    assert 2 in out["meta.selected_steps"]


def test_tail_boost_mass_preserved(tiny_models):
    ctx = _make_dummy_ctx(tiny_models, max_length=128)
    loss_mask = [0, 0, 1, 1, 1, 1, 1, 1]
    loss_weights = [0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    initial_mass = sum(loss_weights)

    sample = {
        "seq.loss_mask": loss_mask,
        "seq.token_weights": loss_weights,
        "student.labels": [-100, -100, 1, 2, 3, 4, 5, 6],
    }
    transform = TailBoost(TailBoost.Config(tail_tokens=2, boost=3.0))
    out = transform(sample, ctx)
    new_weights = out["seq.token_weights"]
    # Total mass of supervised tokens must be preserved
    assert math.isclose(sum(new_weights), initial_mass, rel_tol=1e-7)
    # Last 2 tokens were boosted before normalization
    assert new_weights[-1] > new_weights[2]
    assert math.isclose(new_weights[-1], new_weights[-2], rel_tol=1e-7)


def test_provenance_junction(tiny_models):
    ctx = _make_dummy_ctx(tiny_models, max_length=128)
    # Response has "<End_of_Prefix>"
    # Let response be "Prefix text <End_of_Prefix> Continuation text"
    # starts: [0, 7, 12, 28, 41]
    sample = {
        "student.input_ids": list(range(10)),
        "student.labels": [-100, -100] + list(range(8)),
        "seq.response_span": [2, 10],
        "meta.response_text": "Prefix text <End_of_Prefix> Continuation text",
        "meta.token_starts": [0, 7, 12, 28, 41, 50, 60, 70],
        "seq.loss_mask": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    # Marker starts at 12, len=15, boundary_char = 27
    # Starts < 27: 0, 7, 12 -> 3 tokens prefix
    transform = Provenance(Provenance.Config(junction_tokens=2, marker="<End_of_Prefix>"))
    out = transform(sample, ctx)
    source = out["seq.token_source"]
    assert source[0] == 0 and source[1] == 0  # prompt
    # 3 prefix tokens -> 1
    assert source[2:5] == [1, 1, 1]
    # 2 junction tokens -> 3
    assert source[5:7] == [3, 3]
    # remainder -> 2
    assert source[7:10] == [2, 2, 2]
    assert out["meta.prefix_boundary"] == 2 + 3


def test_transitions_merge_short(tiny_models):
    ctx = _make_dummy_ctx(tiny_models, max_length=128)
    response = "<think>\nShort\n\nLonger paragraph with enough tokens to form a proper step\n\nAnother very long paragraph with many tokens\n</think>\nFinal solution"
    # Token starts every 4 chars
    starts = list(range(0, len(response), 4))
    sample = {
        "student.input_ids": list(range(len(starts) + 2)),
        "seq.response_span": [2, len(starts) + 2],
        "meta.response_text": response,
        "meta.token_starts": starts,
    }
    transform = StepTransitions(StepTransitions.Config(min_step_tokens=4))
    out = transform(sample, ctx)
    assert "seq.step_id" in out
    assert "seq.step_end" in out
    assert "seq.pair_src" in out


def test_corpus_transform_fit_called(tiny_models):
    ctx = _make_dummy_ctx(tiny_models, max_length=128)

    seen = {"count": 0}

    @TRANSFORMS.register("_test_dummy_corpus_transform", force=True)
    class _DummyCorpus(CorpusTransform):
        class Config(BaseConfig):
            pass

        def fit(self, samples, ctx):
            seen["count"] = len(samples)

        def __call__(self, sample, ctx):
            sample["fitted"] = seen["count"]
            return sample

    samples = [{"id": 1}, {"id": 2}, {"id": 3}]
    t = _DummyCorpus(_DummyCorpus.Config())
    res = run_transforms(samples, [t], ctx)
    assert seen["count"] == 3
    assert all(r["fitted"] == 3 for r in res)


def test_render_tokenize_segment_collate(tiny_models):
    ctx = _make_dummy_ctx(tiny_models, max_length=128)
    sample = {
        "problem": "What is 2+2?",
        "response": "<think>\n2+2=4.\n</think>\n\\boxed{4}",
    }
    t_render = RenderCoT(RenderCoT.Config(chat_template=False))
    t_tok = TokenizeCoT(TokenizeCoT.Config(verify=False))
    t_seg = Segment(Segment.Config())

    s1 = t_seg(t_tok(t_render(dict(sample), ctx), ctx), ctx)
    sample2 = {
        "problem": "Solve x=5.",
        "response": "<think>\nx is 5.\n</think>\n\\boxed{5}",
    }
    s2 = t_seg(t_tok(t_render(dict(sample2), ctx), ctx), ctx)

    collator = Collator({"student": ctx.student_tokenizer.pad_token_id})
    batch = collator([s1, s2])
    assert "seq.steps" in batch
    assert isinstance(batch["seq.steps"], list)
    assert all(isinstance(t, torch.Tensor) for t in batch["seq.steps"])
    assert batch["seq.response_span"].shape == (2, 2)


def test_check_recipe_requires_missing_signal(base_recipe):
    base_recipe["data"]["transforms"] = [
        {"type": "render_cot"},
        {"type": "tokenize_cot"},
        {"type": "segment"},
        {"type": "select_energy_top_p"},
    ]
    cfg = recipe_from_dict(base_recipe)
    rep = check_recipe(cfg)
    assert not rep.ok
    assert any("signals.step_strength" in err for err in rep.errors)
