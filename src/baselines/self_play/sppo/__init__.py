"""SPPO (Self-Play Preference Optimization) baseline."""

from __future__ import annotations

from baselines.self_play.sppo.loss import SPPO, sppo_loss

__all__ = ["SPPO", "sppo_loss"]
