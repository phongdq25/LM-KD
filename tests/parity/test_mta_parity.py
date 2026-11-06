"""Numerical parity tests for MTA between LM-KD modular implementation and reference baseline."""

import pytest
import torch
import torch.nn as nn
from tests.parity._ref.mta_ref import (
    ref_aggregate_spans_for_model,
    ref_compute_hidden_span_loss,
    ref_compute_token_weights,
)

from baselines.mta import (
    aggregate_spans_for_model,
    compute_relational_span_loss,
    compute_token_weights,
)


@pytest.mark.parity
def test_token_weights_parity():
    torch.manual_seed(101)
    B, L, D = 3, 12, 16
    h = torch.randn(B, L, D)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    our_w = compute_token_weights(h, mask)
    ref_w = ref_compute_token_weights(h, mask)

    torch.testing.assert_close(our_w, ref_w, atol=1e-6, rtol=1e-5)


@pytest.mark.parity
def test_aggregate_spans_parity():
    torch.manual_seed(102)
    B, L, D = 2, 10, 8
    h = torch.randn(B, L, D)
    mask = torch.ones(B, L)
    weights = torch.softmax(torch.randn(B, L), dim=-1)
    offsets = torch.tensor([[[i * 4, (i + 1) * 4] for i in range(L)] for _ in range(B)])

    spans = [
        [(0, 8), (8, 20), (20, 40)],
        [(4, 16), (20, 32)],
    ]

    our_repr, our_w_sum, our_valid = aggregate_spans_for_model(h, weights, mask, offsets, spans)
    ref_repr, ref_w_sum, ref_valid = ref_aggregate_spans_for_model(h, weights, mask, offsets, spans)

    assert our_repr is not None and ref_repr is not None
    assert our_w_sum is not None and ref_w_sum is not None
    assert our_valid is not None and ref_valid is not None

    torch.testing.assert_close(our_repr, ref_repr, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(our_w_sum, ref_w_sum, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(our_valid, ref_valid)


@pytest.mark.parity
def test_relational_span_loss_parity():
    torch.manual_seed(103)
    B, Max_Spans, D_s, D_t = 3, 5, 8, 12
    s_repr = torch.randn(B, Max_Spans, D_s)
    t_repr = torch.randn(B, Max_Spans, D_t)
    valid_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, False, False, False],
        ]
    )
    w_sum = torch.rand(B, Max_Spans) + 0.05
    proj = nn.Linear(D_s, D_t)

    our_loss = compute_relational_span_loss(s_repr, t_repr, valid_mask, w_sum, projector=proj)
    ref_loss = ref_compute_hidden_span_loss(proj, s_repr, t_repr, valid_mask, w_sum)

    torch.testing.assert_close(our_loss, ref_loss, atol=1e-6, rtol=1e-5)
