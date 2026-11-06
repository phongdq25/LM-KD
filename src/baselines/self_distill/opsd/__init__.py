"""OPSD: On-Policy Self-Distillation.

Method overview:
- Student generates completions on-policy.
- Self-Teacher provides reference target distribution via Generalized JSD or sampled reverse-KL.
- Optional ReasonFirst analysis context or privileged solution prompts.

References:
- Paper: On-Policy Self-Distillation for Language Models
- Source: Self-Distill/OPSD-main (opsd_trainer.py)
"""

from __future__ import annotations

from baselines.self_distill.opsd.config import OPSDConfig
from baselines.self_distill.opsd.input_builder import ReasonFirst
from baselines.self_distill.opsd.loss import (
    OPSD,
    TinkerRKL,
    generalized_jsd,
    opsd_tinker_loss,
)

__all__ = [
    "OPSD",
    "TinkerRKL",
    "ReasonFirst",
    "OPSDConfig",
    "generalized_jsd",
    "opsd_tinker_loss",
]
