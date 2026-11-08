"""Preference distillation baselines for LM-KD.

Core supported knowledge distillation methods for preference/ranking:
- VPD (PAD): Variational Preference Distillation
- PPD: Pairwise Preference Distillation (listwise ranking)
- DCKD (CompressedKL, ChosenNLL): Dual-Constraint Knowledge Distillation
- TVKD: Total Variation Preference Distillation
- ADPA: Adaptive Preference Alignment Distillation
- CTPD: Contrastive Teacher Preference Distillation
- DP3O: Direct Preference with Teacher Distillation
"""

from __future__ import annotations

from baselines.preference import (
    adpa,
    common,
    ctpd,
    dckd,
    dp3o,
    pbsd,
    ppd,
    tvkd,
    vpd,
)
from baselines.preference.adpa import ADPA, advantage_expectation_loss
from baselines.preference.common import plackett_luce_log_probability
from baselines.preference.ctpd import CTPD, ctpd_loss
from baselines.preference.dckd import (
    DCKD,
    ChosenNLL,
    CompressedKL,
    _project_support,
    compressed_projection_kl,
)
from baselines.preference.dp3o import DP3O, dp3o_loss
from baselines.preference.pbsd import PBSD, PBSDOnline, pbsd_loss
from baselines.preference.ppd import PPD, ppd_loss
from baselines.preference.tvkd import TVKD, tvkd_loss, tvkd_scores
from baselines.preference.vpd import PAD, VPD, vpd_loss

__all__ = [
    # Submodules
    "adpa",
    "common",
    "ctpd",
    "dckd",
    "dp3o",
    "pbsd",
    "ppd",
    "tvkd",
    "vpd",
    # Criterion classes
    "VPD",
    "PAD",
    "PPD",
    "DCKD",
    "CompressedKL",
    "ChosenNLL",
    "TVKD",
    "ADPA",
    "CTPD",
    "DP3O",
    "PBSD",
    # Producers
    "PBSDOnline",
    # Functional primitives
    "vpd_loss",
    "ppd_loss",
    "compressed_projection_kl",
    "_project_support",
    "tvkd_loss",
    "tvkd_scores",
    "advantage_expectation_loss",
    "ctpd_loss",
    "dp3o_loss",
    "pbsd_loss",
    "plackett_luce_log_probability",
]
