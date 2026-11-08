"""Functional distillation loss primitives (Tier 1: Pure PyTorch, zero framework dependencies).

Operates on raw torch.Tensors and can be used directly in any custom loop, HF Trainer, or TRL Trainer.
"""

from __future__ import annotations

from lmkd.losses.functional.contrastive import info_nce_loss
from lmkd.losses.functional.divergences import (
    TokenFn,
    amid,
    csd,
    forward_kl,
    generalized_jsd,
    get_divergence,
    js_distance,
    reverse_kl,
    skewed_forward_kl,
    skewed_reverse_kl,
    symmetric_kl,
    tv_distance,
)
from lmkd.losses.functional.rl import (
    grpo_loss,
    masked_mean,
    ppo_clip_per_token,
)
from lmkd.losses.functional.sequence import (
    group_view,
    masked_seq_mean,
    pairwise,
    sequence_logps,
)
from lmkd.rl.advantages import grpo_advantage

__all__ = [
    # contrastive
    "info_nce_loss",
    # divergences
    "TokenFn",
    "amid",
    "forward_kl",
    "reverse_kl",
    "symmetric_kl",
    "js_distance",
    "tv_distance",
    "skewed_forward_kl",
    "skewed_reverse_kl",
    "csd",
    "generalized_jsd",
    "get_divergence",
    # sequence utilities
    "sequence_logps",
    "group_view",
    "pairwise",
    "masked_seq_mean",
    # RL
    "masked_mean",
    "ppo_clip_per_token",
    "grpo_loss",
    "grpo_advantage",
]
