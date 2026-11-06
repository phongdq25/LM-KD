"""Feature Dynamics Distillation (FDD) baseline."""

from __future__ import annotations

from baselines.same_tokenizer.fdd.functional import compute_fdd_loss, soft_label_distill_loss
from baselines.same_tokenizer.fdd.loss import FDD, FDDCriterion

__all__ = ["FDD", "FDDCriterion", "compute_fdd_loss", "soft_label_distill_loss"]

