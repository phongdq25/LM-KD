"""SRA baseline: Span Representation Alignment for cross-tokenizer distillation."""

from __future__ import annotations

from baselines.cross_tokenizer.sra.aligner import (
    LcsSpan,
    lcs_segments,
    longest_common_subsequence,
    pad_rows,
)
from baselines.cross_tokenizer.sra.loss import SRA

__all__ = [
    "SRA",
    "LcsSpan",
    "longest_common_subsequence",
    "lcs_segments",
    "pad_rows",
]

