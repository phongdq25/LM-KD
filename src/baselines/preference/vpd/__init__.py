"""VPD (PAD): Variational Preference Distillation."""

from __future__ import annotations

from baselines.preference.vpd.loss import PAD, VPD, vpd_loss

__all__ = ["VPD", "PAD", "vpd_loss"]

