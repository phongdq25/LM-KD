"""TVKD: Total Variation Preference Distillation."""

from __future__ import annotations

from baselines.preference.tvkd.loss import TVKD, tvkd_loss, tvkd_scores

__all__ = ["TVKD", "tvkd_loss", "tvkd_scores"]

