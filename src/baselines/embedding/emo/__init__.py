"""EMO baseline for embedding knowledge distillation."""

from __future__ import annotations

from baselines.embedding.emo.loss import (
    EMO,
    EMOKD,
    CKALoss,
    align_tokens,
    build_reciprocal_mapping_from_token_lists,
    compute_att_loss_2,
    compute_ot_loss,
    compute_token_importance,
    pairwise_attention_distance,
    project_importance,
    sinkhorn,
)

__all__ = [
    "CKALoss",
    "EMO",
    "EMOKD",
    "align_tokens",
    "build_reciprocal_mapping_from_token_lists",
    "compute_att_loss_2",
    "compute_ot_loss",
    "compute_token_importance",
    "pairwise_attention_distance",
    "project_importance",
    "sinkhorn",
]
