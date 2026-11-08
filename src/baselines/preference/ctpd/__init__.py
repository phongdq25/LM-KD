"""CTPD: Contrastive Teacher Preference Distillation."""

from __future__ import annotations

from baselines.preference.ctpd.loss import CTPD, ctpd_loss

__all__ = ["CTPD", "ctpd_loss"]

