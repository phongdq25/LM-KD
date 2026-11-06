"""ADPA: Advantage Distillation Preference Alignment."""

from __future__ import annotations

from baselines.preference.adpa.loss import ADPA, advantage_expectation_loss

__all__ = ["ADPA", "advantage_expectation_loss"]

