"""Embedding distillation baselines."""

from __future__ import annotations

from baselines.embedding import distillcse, emo, jasper, mol, talas
from baselines.embedding.distillcse import (
    DistillCSE,
    DistillCSEKD,
    distillcse_kd_loss,
)
from baselines.embedding.emo import (
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
from baselines.embedding.jasper import (
    Jasper,
    cosine_embedding_loss,
    get_score_diff,
    pair_inbatch_similarity_loss,
    pair_inbatch_triplet_loss,
)
from baselines.embedding.mol import (
    LinearCKA,
    MoL,
    MoLKD,
    align_teacher_to_student,
    mol_kd_loss,
)
from baselines.embedding.talas import LASD

__all__ = [
    "distillcse",
    "emo",
    "jasper",
    "mol",
    "talas",
    "CKALoss",
    "DistillCSE",
    "DistillCSEKD",
    "EMO",
    "EMOKD",
    "Jasper",
    "LASD",
    "LinearCKA",
    "MoL",
    "MoLKD",
    "align_teacher_to_student",
    "align_tokens",
    "build_reciprocal_mapping_from_token_lists",
    "compute_att_loss_2",
    "compute_ot_loss",
    "compute_token_importance",
    "cosine_embedding_loss",
    "distillcse_kd_loss",
    "get_score_diff",
    "mol_kd_loss",
    "pair_inbatch_similarity_loss",
    "pair_inbatch_triplet_loss",
    "pairwise_attention_distance",
    "project_importance",
    "sinkhorn",
]
