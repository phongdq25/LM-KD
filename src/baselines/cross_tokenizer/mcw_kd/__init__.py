"""MCW-KD baseline: Multi-Cost Wasserstein Distillation for cross-tokenizer distillation."""

from __future__ import annotations

from baselines.cross_tokenizer.mcw_kd.aligner import DtwEdit, dtw, dtw_alignment
from baselines.cross_tokenizer.mcw_kd.etp import (
    ETP,
    DualPotentialNetwork,
    MultiCostSemiDualOT,
)
from baselines.cross_tokenizer.mcw_kd.loss import (
    MCWHidden,
    McwHidden,
    MCWLogits,
    McwLogits,
    OTHidden,
    OTLogits,
    context_windows,
)

__all__ = [
    "OTLogits",
    "OTHidden",
    "MCWLogits",
    "MCWHidden",
    "McwLogits",
    "McwHidden",
    "ETP",
    "DualPotentialNetwork",
    "MultiCostSemiDualOT",
    "context_windows",
    "DtwEdit",
    "dtw",
    "dtw_alignment",
]
