"""On-Policy & Self-Distillation baselines for LM-KD.

Contains canonical method implementations ("phương pháp của từng cái") built upon
the core RL/verl execution infrastructure in `src/lmkd`:
- OPSD: On-Policy Self-Distillation (GJSD divergence, Tinker reverse-KL, ReasonFirst)
- SDPO: Self-Distillation with Preference Optimization (Student top-k support, tail bucket, SDPOReprompt, IS clipping)
- RLSD: Reinforcement Learning Self-Distillation (STGCA advantage modulation, scheduled lambda, privileged reference teacher)
- LUFFY: Learning to Reason Under Off-Policy Guidance (Policy Shaping via Regularized IS, GRPOSplit, Prefix Weighting)
"""

from __future__ import annotations

from baselines.self_distill import (
    common,
    luffy,
    opsd,
    rlsd,
    sdpo,
)
from baselines.self_distill.common import (
    add_tail,
    effective_lambda,
    masked_mean,
    renorm_topk,
)
from baselines.self_distill.luffy import (
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
from baselines.self_distill.opsd import (
    OPSD,
    OPSDConfig,
    ReasonFirst,
    TinkerRKL,
    generalized_jsd,
    opsd_tinker_loss,
)
from baselines.self_distill.rlsd import (
    RLSD,
    STGCA,
    LambdaSchedule,
    PPOClip,
    RLSDConfig,
    RLSDLoss,
    RLSDReweight,
    grpo_loss,
    ppo_clip_per_token,
    rlsd_reweight,
)
from baselines.self_distill.sdpo import (
    SDPO,
    SDPOConfig,
    SDPODistill,
    SDPOLoss,
    SDPOReprompt,
    TopKKD,
    compute_self_distillation_loss,
    sdpo_per_token,
)

__all__ = [
    # Submodules
    "common",
    "luffy",
    "opsd",
    "sdpo",
    "rlsd",
    # Canonical Method Classes
    "LUFFY",
    "OPSD",
    "SDPO",
    "RLSD",
    # Specific Criteria
    "LUFFYLoss",
    "TinkerRKL",
    "SDPOLoss",
    "SDPODistill",
    "TopKKD",
    "RLSDLoss",
    "PPOClip",
    # Input Builders & Advantage Transforms & Producers
    "GRPOSplit",
    "LUFFYPrefixWeight",
    "LUFFYGroupProducer",
    "ReasonFirst",
    "SDPOReprompt",
    "RLSDReweight",
    "STGCA",
    "LambdaSchedule",
    # Config Dataclasses
    "OPSDConfig",
    "SDPOConfig",
    "RLSDConfig",
    # Functional Primitives
    "compute_token_on_off_policy_loss",
    "compute_grpo_outcome_advantage_split",
    "compute_sft_pure_loss",
    "luffy_loss",
    "generalized_jsd",
    "opsd_tinker_loss",
    "sdpo_per_token",
    "compute_self_distillation_loss",
    "add_tail",
    "renorm_topk",
    "ppo_clip_per_token",
    "grpo_loss",
    "rlsd_reweight",
    "masked_mean",
    "effective_lambda",
]
