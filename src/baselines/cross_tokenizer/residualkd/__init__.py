"""ResidualKD baseline: residual student loss and projector pretraining (same and cross tokenizer)."""

from __future__ import annotations

from baselines.cross_tokenizer.residualkd.loss import (
    ProjectorAS,
    ProjectorSA,
    ProjectorTA,
    ResidualKD,
    ResidualPretrain,
    positional_labels,
    positional_logits,
    residual_core,
)

__all__ = [
    "ResidualKD",
    "ResidualPretrain",
    "ProjectorTA",
    "ProjectorSA",
    "ProjectorAS",
    "residual_core",
    "positional_labels",
    "positional_logits",
]
