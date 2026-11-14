"""Same-tokenizer baselines package for LM-KD."""

from __future__ import annotations

from baselines.same_tokenizer import distillm2, fdd
from baselines.same_tokenizer.distillm2 import (
    DistiLLM2,
    DistiLLM2Criterion,
    DistiLLM2Producer,
    compute_dynamic_alpha,
    skewed_forward_kl,
    skewed_reverse_kl,
)
from baselines.same_tokenizer.fdd import FDD, FDDCriterion, compute_fdd_loss, soft_label_distill_loss

__all__ = [
    "fdd",
    "FDD",
    "FDDCriterion",
    "compute_fdd_loss",
    "soft_label_distill_loss",
    "distillm2",
    "DistiLLM2",
    "DistiLLM2Criterion",
    "DistiLLM2Producer",
    "compute_dynamic_alpha",
    "skewed_forward_kl",
    "skewed_reverse_kl",
]
