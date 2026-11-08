"""SPIN (Self-Play Fine-Tuning) baseline."""

from __future__ import annotations

from baselines.self_play.spin.loss import SPIN, spin_loss

__all__ = ["SPIN", "spin_loss"]
