"""DWA-KD: Dynamic Window Alignment Knowledge Distillation baseline."""

from __future__ import annotations

from baselines.cross_tokenizer.dwa.loss import DWAKD
from baselines.cross_tokenizer.dwa.soft_dtw import (
    SoftDTW,
    compute_dwa_band_penalty,
    pairwise_cosine_distance,
    soft_dtw_divergence,
)

__all__ = [
    "DWAKD",
    "SoftDTW",
    "compute_dwa_band_penalty",
    "pairwise_cosine_distance",
    "soft_dtw_divergence",
]

