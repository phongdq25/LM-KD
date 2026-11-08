"""Dual-Space Knowledge Distillation (DSKD) baselines: DSKD, DSKD-CMA, DSKDv2, and DSKDv2-ETA."""

from __future__ import annotations

from baselines.dskd.common import (
    _compute_s2t_hidden,
    _DualSpaceBase,
    _get_embeddings,
    _get_heads,
    _init_pinv_and_buffers,
    _load_projector_checkpoint,
    _view_argmax,
)
from baselines.dskd.loss import DSKD, DSKDCMA

__all__ = [
    "DSKD",
    "DSKDCMA",
    "_DualSpaceBase",
    "_init_pinv_and_buffers",
    "_compute_s2t_hidden",
    "_load_projector_checkpoint",
    "_view_argmax",
    "_get_heads",
    "_get_embeddings",
]
