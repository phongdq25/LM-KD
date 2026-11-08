"""RSPO (Regularized Self-Play Preference Optimization) baseline."""

from __future__ import annotations

from baselines.self_play.rspo.loss import RSPO, rspo_loss

__all__ = ["RSPO", "rspo_loss"]
