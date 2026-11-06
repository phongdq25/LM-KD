"""Multi-granularity Text Alignment (MTA) distillation baseline.

Provides:
- ``MTA``: Criterion for multi-granularity alignment across student and teacher hidden states.
- ``MtaSpan``: Aligner for linguistic phrase and word spans extraction.
- Functional primitives for token weights, span representations, relational span loss, and token cosine loss.
"""

from __future__ import annotations

from baselines.mta.aligner import MtaSpan
from baselines.mta.functional import (
    aggregate_spans_for_model,
    compute_relational_span_loss,
    compute_token_cosine_loss,
    compute_token_weights,
    extract_spans_and_words,
    filter_overlapping_spans,
)
from baselines.mta.loss import MTA

__all__ = [
    "MTA",
    "MtaSpan",
    "compute_token_weights",
    "aggregate_spans_for_model",
    "compute_relational_span_loss",
    "compute_token_cosine_loss",
    "filter_overlapping_spans",
    "extract_spans_and_words",
]

