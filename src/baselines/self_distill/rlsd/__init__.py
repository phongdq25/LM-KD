"""RLSD: Reinforcement Learning Self-Distillation.

Method overview:
- Generates groups of responses on-policy (rollout groups with task verifiers).
- Self-Teacher Guided Confidence Advantage (STGCA): modulates policy gradient advantages
  using the log-prob delta between the privileged self-teacher and student.
- Scheduled lambda interpolation: warms up and decays the distillation weight.
- Optimizes policy via clipped surrogate objective with global token count normalization.

References:
- Paper: Reinforcement Learning Self-Distillation (2024)
- Source: Self-Distill/RLSD-main (verl/workers/actor/dp_opsd_actor.py & opsd_trainer.py)
"""

from __future__ import annotations

from baselines.self_distill.rlsd.advantage import (
    STGCA,
    LambdaSchedule,
    RLSDReweight,
    effective_lambda,
    rlsd_reweight,
)
from baselines.self_distill.rlsd.loss import (
    PPOClip,
    RLSDLoss,
    grpo_loss,
    ppo_clip_per_token,
)
from baselines.self_distill.rlsd.teacher import RLSDConfig

# Canonical method class alias
RLSD = RLSDLoss

__all__ = [
    "RLSD",
    "RLSDLoss",
    "PPOClip",
    "RLSDReweight",
    "STGCA",
    "LambdaSchedule",
    "RLSDConfig",
    "ppo_clip_per_token",
    "grpo_loss",
    "rlsd_reweight",
    "effective_lambda",
]
