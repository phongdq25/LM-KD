"""LUFFY baseline alias in self_distill namespace."""

from __future__ import annotations

from baselines.luffy import (
    LUFFY,
    GRPOSplit,
    LUFFYGroupProducer,
    LUFFYLoss,
    LUFFYPrefixWeight,
    compute_grpo_outcome_advantage_split,
    compute_sft_pure_loss,
    compute_token_on_off_policy_loss,
    luffy_loss,
)

__all__ = [
    "LUFFY",
    "LUFFYLoss",
    "GRPOSplit",
    "LUFFYPrefixWeight",
    "LUFFYGroupProducer",
    "compute_token_on_off_policy_loss",
    "compute_grpo_outcome_advantage_split",
    "compute_sft_pure_loss",
    "luffy_loss",
]

