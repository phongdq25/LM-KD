"""SDPO: Self-Distillation with Preference Optimization.

Method overview:
- Reconstructs prompts using peer rollouts (successful solutions or verifier feedback).
- Evaluates self-distillation loss over the student's top-k support with a residual tail bucket.
- Applies importance sampling correction against rollout probabilities.

References:
"""

from __future__ import annotations

from baselines.self_distill.sdpo.config import SDPOConfig
from baselines.self_distill.sdpo.input_builder import SDPOReprompt
from baselines.self_distill.sdpo.loss import (
    SDPODistill,
    SDPOLoss,
    TopKKD,
    add_tail,
    compute_self_distillation_loss,
    sdpo_per_token,
)

# Canonical method class alias
SDPO = SDPOLoss

__all__ = [
    "SDPO",
    "SDPOLoss",
    "SDPODistill",
    "TopKKD",
    "SDPOReprompt",
    "SDPOConfig",
    "sdpo_per_token",
    "compute_self_distillation_loss",
    "add_tail",
]
