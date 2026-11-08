"""T-SPIN (Triplet Self-Play Fine-Tuning) baseline."""

from __future__ import annotations

from baselines.self_play.tspin.loss import TSPIN, tspin_loss

__all__ = ["TSPIN", "tspin_loss"]
