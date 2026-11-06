"""Same-tokenizer baselines package for LM-KD."""

from __future__ import annotations

from baselines.same_tokenizer import fdd
from baselines.same_tokenizer.fdd import FDD, FDDCriterion, compute_fdd_loss, soft_label_distill_loss

__all__ = [
    "fdd",
    "FDD",
    "FDDCriterion",
    "compute_fdd_loss",
    "soft_label_distill_loss",
]

