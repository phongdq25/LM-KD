"""SSFT (Segment-Selective SFT) Baseline Package."""

from __future__ import annotations

from baselines.longCoT.ssft.pipeline import (
    ANSWER_TEMPLATE,
    USER_TEMPLATE,
    IGAttribution,
    SSFTStage,
    compact_ig,
)
from baselines.longCoT.ssft.transforms import (
    SelectCumulativeStrength,
    select_segments,
)

__all__ = [
    "SSFTStage",
    "IGAttribution",
    "SelectCumulativeStrength",
    "select_segments",
    "compact_ig",
    "USER_TEMPLATE",
    "ANSWER_TEMPLATE",
]
