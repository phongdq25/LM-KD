"""E-SCOT baseline: Entropic Soft Context-aware Optimal Transport for cross-tokenizer distillation."""

from __future__ import annotations

from baselines.cross_tokenizer.escot.aligner import CharSpan, _span_align
from baselines.cross_tokenizer.escot.loss import (
    ESCOST,
    ESCOT,
    Escost,
    Escot,
    compute_renyi_entropies,
    escot_weights,
)

__all__ = [
    "ESCOT",
    "Escot",
    "ESCOST",
    "Escost",
    "compute_renyi_entropies",
    "escot_weights",
    "CharSpan",
    "_span_align",
]

