"""SPACE (Self-Play via Noise Contrastive Estimation) baseline."""

from __future__ import annotations

from baselines.self_play.space.loss import SPACE, space_loss

__all__ = ["SPACE", "space_loss"]
