"""DistiLLM-2 baseline for same-tokenizer LLM distillation.

Paper: Ko et al. "DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs", ICML 2025 (Oral).
"""

from __future__ import annotations

from baselines.same_tokenizer.distillm2.loss import (
    DistiLLM2,
    DistiLLM2Criterion,
    compute_dynamic_alpha,
)
from baselines.same_tokenizer.distillm2.producer import DistiLLM2EpochProducer, DistiLLM2Producer
from lmkd.losses.functional import skewed_forward_kl, skewed_reverse_kl

__all__ = [
    "DistiLLM2",
    "DistiLLM2Criterion",
    "DistiLLM2Producer",
    "DistiLLM2EpochProducer",
    "compute_dynamic_alpha",
    "skewed_forward_kl",
    "skewed_reverse_kl",
]
