"""LUFFY: Learning to Reason Under Off-Policy Guidance baseline for LM-KD.

Method overview:
- Generates on-policy rollouts and integrates external off-policy reasoning traces into GRPO.
- Employs Policy Shaping via Regularized Importance Sampling (e.g. p / (p + 0.1)) to emphasize
  low-probability yet critical reasoning steps without destabilizing gradient variance.
- Evaluates group-relative advantages with on-policy baseline estimation (`grpo_split`)
  and prefix reward reweighting (`luffy_prefix_weight`).

References:
- Paper: LUFFY: Learning to Reason Under Off-Policy Guidance (NeurIPS 2025)
  https://arxiv.org/abs/2504.14945
- Source: LUFFY/luffy/verl/verl/mix_src/
"""

from __future__ import annotations

from baselines.luffy.advantage import (
    GRPOSplit,
    LUFFYPrefixWeight,
    compute_grpo_outcome_advantage_split,
)
from baselines.luffy.loss import (
    LUFFYLoss,
    compute_sft_pure_loss,
    compute_token_on_off_policy_loss,
    luffy_loss,
)
from baselines.luffy.producer import (
    LUFFYGroupProducer,
)

# Canonical method class alias
LUFFY = LUFFYLoss

__all__ = [
    # Canonical Method Class
    "LUFFY",
    "LUFFYLoss",
    # Advantage & Transforms
    "GRPOSplit",
    "LUFFYPrefixWeight",
    "compute_grpo_outcome_advantage_split",
    # Loss & Functional
    "compute_token_on_off_policy_loss",
    "compute_sft_pure_loss",
    "luffy_loss",
    # Producer
    "LUFFYGroupProducer",
]

