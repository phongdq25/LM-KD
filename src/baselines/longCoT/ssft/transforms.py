"""SSFT cumulative attribution mass selection transforms."""

from __future__ import annotations

from typing import Any

import numpy as np

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import Transform


def select_segments(
    compact: list[list[float]] | list[tuple[int, float, float]],
    cumulative_ratio: float = 0.7,
    coherence_max: float = 0.8,
) -> list[int]:
    """Pick important segments from attribution mass."""
    strengths = [sum_abs / (n**0.5) if n else 0.0 for n, sum_abs, _ in compact]
    ranked = sorted(enumerate(strengths), key=lambda item: -item[1])
    order = [index for index, _ in ranked]
    values = np.array([value for _, value in ranked])
    total = values.sum()
    if total <= 0:
        return []
    cumulative = np.cumsum(values / total)
    cut = 0
    for cut in range(len(cumulative)):
        if cumulative[cut] >= cumulative_ratio:
            break
    important = sorted(order[: cut + 1])
    coherence = [abs(sum_signed) / sum_abs if sum_abs > 0 else 1.0 for _, sum_abs, sum_signed in compact]
    return [index for index in important if coherence[index] <= coherence_max]


@TRANSFORMS.register(
    "select_cumulative_strength"
)
class SelectCumulativeStrength(Transform):
    """SSFT cumulative attribution mass step selection.

    """

    requires = ("seq.steps", "signals.segment_ig")
    provides = ("seq.loss_mask", "meta.selected_steps")

    class Config(BaseConfig):
        cumulative_ratio: float = 0.7
        coherence_max: float = 0.8
        always_keep: list[int] = [0, -2, -1]

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        compact = sample.get("signals.segment_ig")
        if compact is None:
            raise KeyError("sample missing 'signals.segment_ig'")

        k_steps = len(sample["seq.steps"])
        if len(compact) != k_steps:
            compact_filtered = [row for row in compact if row[0] > 0]
            if len(compact_filtered) == k_steps:
                compact = compact_filtered
            else:
                raise ValueError(f"record {sample.get('meta.id')}: steps={k_steps}, segment_ig={len(compact)}")
        selected = select_segments(compact, self.cfg.cumulative_ratio, self.cfg.coherence_max)
        always_norm = {k if k >= 0 else k_steps + k for k in self.cfg.always_keep}
        keep = {i for i in set(selected) | always_norm if 0 <= i < k_steps}
        sorted_selected = sorted(keep)

        seq_len = len(sample["student.input_ids"])
        mask = [0] * seq_len
        for idx in sorted_selected:
            start, end = sample["seq.steps"][idx]
            for pos in range(start, min(end, seq_len)):
                mask[pos] = 1

        resp_end = sample.get("seq.response_span", [0, seq_len])[1]
        for pos in range(resp_end, seq_len):
            mask[pos] = 1

        if sum(mask) == 0:
            return None

        sample["seq.loss_mask"] = mask
        sample["meta.selected_steps"] = sorted_selected
        return sample
