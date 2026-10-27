"""Functional distillation loss primitives (Tier 1: Pure PyTorch, zero framework dependencies).

Operates on raw torch.Tensors and can be used directly in any custom loop, HF Trainer, or TRL Trainer.
"""
from __future__ import annotations

from lmkd.losses.functional.divergences import (
    TokenFn,
    csd_surrogate,
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
from lmkd.losses.functional.preference import (
    advantage_expectation_loss,
    compressed_projection_kl,
    ctpd_loss,
    dp3o_loss,
    dpo_loss,
    pbsd_loss,
    plackett_luce_log_probability,
    ppd_loss,
    sigmoid_dpo_loss,
    tvkd_loss,
    tvkd_scores,
    vpd_loss,
)
from lmkd.losses.functional.rl import (
    add_tail,
    grpo_loss,
    masked_mean,
    ppo_clip_per_token,
    rlsd_reweight,
    sdpo_per_token,
)
from lmkd.losses.functional.sequence import (
    group_view,
    masked_seq_mean,
    pairwise,
    sequence_logps,
)
from lmkd.rl.advantages import grpo_advantage

__all__ = [
    # divergences
    "TokenFn",
    "forward_kl",
    "reverse_kl",
    "symmetric_kl",
    "js_distance",
    "tv_distance",
    "skewed_forward_kl",
    "skewed_reverse_kl",
    "csd_surrogate",
    "generalized_jsd",
    "get_divergence",
    # sequence utilities
    "sequence_logps",
    "group_view",
    "pairwise",
    "masked_seq_mean",
    # preference & ranking
    "plackett_luce_log_probability",
    "vpd_loss",
    "ppd_loss",
    "compressed_projection_kl",
    "tvkd_loss",
    "tvkd_scores",
    "advantage_expectation_loss",
    "ctpd_loss",
    "sigmoid_dpo_loss",
    "dpo_loss",
    "dp3o_loss",
    "pbsd_loss",
    # RL
    "masked_mean",
    "ppo_clip_per_token",
    "grpo_loss",
    "grpo_advantage",
    "rlsd_reweight",
    "add_tail",
    "sdpo_per_token",
]

