"""Unit tests for Multi-granularity Text Alignment (MTA) loss and aligner."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from baselines.mta import (
    MTA,
    MtaSpan,
    aggregate_spans_for_model,
    compute_relational_span_loss,
    compute_token_cosine_loss,
    compute_token_weights,
    extract_spans_and_words,
)
from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext, ModelPairInfo
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


def test_baselines_mta_exports():
    """Verify that baselines.mta exports the canonical classes."""
    assert MTA is not None
    assert MtaSpan is not None



def test_compute_token_weights():
    torch.manual_seed(42)
    B, L, D = 2, 5, 8
    h = torch.randn(B, L, D)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    weights = compute_token_weights(h, mask)

    assert weights.shape == (B, L)
    assert not weights.requires_grad
    # Masked positions should have 0 weight
    assert weights[0, 4] == 0.0
    assert weights[1, 3] == 0.0
    assert weights[1, 4] == 0.0
    # Valid positions should have strictly positive weights
    assert (weights[0, :4] > 0).all()
    assert (weights[1, :3] > 0).all()


def test_aggregate_spans():
    torch.manual_seed(42)
    B, L, D = 2, 6, 4
    h = torch.ones(B, L, D)
    weights = torch.ones(B, L) / L
    mask = torch.ones(B, L)
    # Offsets: each token covers 5 chars: [0, 5), [5, 10), [10, 15), ...
    offsets = torch.tensor([[[i * 5, (i + 1) * 5] for i in range(L)] for _ in range(B)])

    # Spans: item 0 has 2 spans, item 1 has 1 span
    spans = [
        [(0, 10), (10, 25)],  # covers tokens [0, 1] and [2, 3, 4]
        [(5, 15)],  # covers tokens [1, 2]
    ]

    s_repr, w_sum, valid_mask = aggregate_spans_for_model(h, weights, mask, offsets, spans)
    assert s_repr is not None and w_sum is not None and valid_mask is not None
    assert s_repr.shape == (2, 2, D)
    assert valid_mask[0, 0] and valid_mask[0, 1]
    assert valid_mask[1, 0] and not valid_mask[1, 1]
    # Represented vectors should be close to 1.0 since h is all ones
    torch.testing.assert_close(s_repr[valid_mask], torch.ones_like(s_repr[valid_mask]))


def test_compute_relational_span_loss_zero():
    torch.manual_seed(42)
    B, Max_Spans, D = 2, 3, 8
    repr_vec = torch.randn(B, Max_Spans, D)
    valid_mask = torch.ones(B, Max_Spans, dtype=torch.bool)
    w_sum = torch.ones(B, Max_Spans)

    loss = compute_relational_span_loss(repr_vec, repr_vec, valid_mask, w_sum)
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_compute_relational_span_loss_gradient():
    torch.manual_seed(42)
    B, Max_Spans, D_s, D_t = 2, 4, 6, 8
    s_repr = torch.randn(B, Max_Spans, D_s, requires_grad=True)
    t_repr = torch.randn(B, Max_Spans, D_t)
    valid_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    w_sum = torch.rand(B, Max_Spans) + 0.1
    proj = nn.Linear(D_s, D_t)

    loss = compute_relational_span_loss(s_repr, t_repr, valid_mask, w_sum, projector=proj)
    assert loss.item() > 0.0
    loss.backward()
    assert s_repr.grad is not None
    assert s_repr.grad.abs().sum() > 0
    assert proj.weight.grad is not None


def test_compute_token_cosine_loss():
    torch.manual_seed(42)
    B, L, D = 2, 4, 6
    s_h = torch.randn(B, L, D, requires_grad=True)
    t_h = torch.randn(B, L, D)
    weights = torch.ones(B, L)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    loss = compute_token_cosine_loss(s_h, t_h, None, weights, mask)
    assert loss.item() >= 0.0
    loss.backward()
    assert s_h.grad is not None


def test_extract_spans_heuristic():
    texts = ["Antigravity is a powerful coding assistant.", "Short text."]
    spans, words = extract_spans_and_words(texts, nlp=None)
    assert len(spans) == 2 and len(words) == 2
    assert len(words[0]) > 0
    assert len(spans[0]) > 0


def test_mta_criterion_lifecycle():
    torch.manual_seed(42)
    B, L, D_s, D_t = 2, 6, 4, 8
    s_h = torch.randn(B, L, D_s, requires_grad=True)
    t_h = torch.randn(B, L, D_t)
    mask = torch.ones(B, L, dtype=torch.long)
    offsets = torch.tensor([[[i * 5, (i + 1) * 5] for i in range(L)] for _ in range(B)], dtype=torch.long)

    cfg = MTA.Config(
        student_layers=[-1],
        teacher_layers=[-1],
        split_layer_mapping=[0, 0, 1],
        token_loss_weight=0.1,
    )
    crit = MTA(cfg)
    modules = ModuleRegistry()
    crit.build_modules(
        ModelPairInfo(
            student_hidden=D_s,
            student_layers=2,
            student_vocab=100,
            teacher_hidden=D_t,
            teacher_layers=3,
            teacher_vocab=100,
        ),
        modules,
    )

    assert "mta__proj_0" in modules

    spans = [[[0, 10], [10, 25]], [[5, 15]]]
    words = [[[0, 5], [5, 10], [10, 15]], [[5, 10]]]
    batch = Batch(
        {
            "student.input_ids": torch.ones(B, L, dtype=torch.long),
            "student.attention_mask": mask,
            "student.offsets": offsets,
            "align.mta_spans": spans,
            "align.mta_words": words,
        }
    )

    side_s = SimpleNamespace(hidden=lambda l: s_h, num_layers=2)
    side_t = SimpleNamespace(hidden=lambda l: t_h, num_layers=3)
    teacher_out = TeacherOutput(side=side_t)
    ctx = LossContext(batch, side_s, [], teacher_out, modules, None, None)

    out = crit(ctx)
    assert "mta" in out.terms
    loss = out.terms["mta"].value
    assert loss.item() > 0.0

    loss.backward()
    assert s_h.grad is not None
    assert s_h.grad.abs().sum() > 0


def test_mta_span_aligner():
    aligner = MtaSpan()
    sample = {
        "student.input_ids": [1, 2, 3],
        "student.prompt_len": 1,
        "teacher.input_ids": [1, 2, 3],
        "teacher.prompt_len": 1,
        "meta.response_text": "Machine learning knowledge distillation",
    }
    res = aligner(sample, None)
    assert "align.mta_spans" in res
    assert "align.mta_words" in res
    assert len(res["align.mta_words"]) > 0
