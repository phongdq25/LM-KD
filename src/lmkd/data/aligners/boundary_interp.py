"""``boundary_interp`` aligner (IMPACT boundary-preserving increments, vectorised)."""
from __future__ import annotations

import bisect
from typing import Any, Literal

import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ALIGNERS
from lmkd.data.aligners.base import IDS_OFFSETS, Aligner, reg_field

reg_field("align.impact_lo", "token_k", torch.long, 0, "student", "[T_s,2] lower prefix-state index of boundary a_i/b_i")
reg_field("align.impact_hi", "token_k", torch.long, 0, "student", "[T_s,2] upper prefix-state index")
reg_field("align.impact_w", "token_k", torch.float32, 0.0, "student", "[T_s,2] interpolation weight on hi")


def boundary_coeffs(c: int, t_off: list, mode: str = "linear") -> tuple[int, int, float]:
    """``(lo, hi, w)`` with ``state(c) = (1-w) P[lo] + w P[hi]`` for ``P = [0; teacher states]`` (indices 0..n).

    Same case analysis, in the same order, as ``_get_teacher_boundary_state``
    (IMPACT/code/impact_utils.py:665-705) but returns coefficients instead of the interpolated tensor."""
    n = len(t_off)
    if n == 0 or c <= t_off[0][0]:
        return 0, 0, 0.0
    if c >= t_off[-1][1]:
        return n, n, 0.0
    mode = (mode or "linear").lower()
    for j in range(n):
        a_j, b_j = t_off[j]
        if c == a_j:
            return j, j, 0.0
        if c == b_j:
            return j + 1, j + 1, 0.0
        if a_j < c < b_j:
            if mode == "containing":
                return j + 1, j + 1, 0.0
            if mode == "nearest":
                return (j, j, 0.0) if (c - a_j) <= (b_j - c) else (j + 1, j + 1, 0.0)
            span_len = b_j - a_j
            if span_len == 0:
                return j, j, 0.0
            return j, j + 1, (c - a_j) / span_len
    return n, n, 0.0


def _is_monotone(t_off: list) -> bool:
    return all(a <= b for a, b in t_off) and all(
        t_off[k][0] <= t_off[k + 1][0] and t_off[k][1] <= t_off[k + 1][1] for k in range(len(t_off) - 1))


def boundary_table(cs: list[int], t_off: list, mode: str = "linear") -> list[tuple[int, int, float]]:
    """:func:`boundary_coeffs` for many boundaries; O(log n) each when the teacher offsets are monotone.

    With non-decreasing starts and ends the first matching ``j`` of the original linear scan is the first one with
    ``b_j >= c`` (bisect); if its start is after ``c`` nothing matches (falls off the loop -> ``n``). Non-monotone
    offsets use the literal scan."""
    mode = (mode or "linear").lower()
    if not t_off or not _is_monotone(t_off):
        return [boundary_coeffs(c, t_off, mode) for c in cs]
    n = len(t_off)
    ends = [b for _, b in t_off]
    out: list[tuple[int, int, float]] = []
    for c in cs:
        if c <= t_off[0][0]:
            out.append((0, 0, 0.0))
        elif c >= ends[-1]:
            out.append((n, n, 0.0))
        else:
            j = bisect.bisect_left(ends, c)
            out.append(_at(c, j, t_off, mode) if t_off[j][0] <= c else (n, n, 0.0))
    return out


def _at(c: int, j: int, t_off: list, mode: str) -> tuple[int, int, float]:
    """Coefficients of boundary ``c`` known to satisfy ``a_j <= c <= b_j`` (first such ``j``)."""
    a_j, b_j = t_off[j]
    if c == a_j:
        return j, j, 0.0
    if c == b_j:
        return j + 1, j + 1, 0.0
    if mode == "containing":
        return j + 1, j + 1, 0.0
    if mode == "nearest":
        return (j, j, 0.0) if (c - a_j) <= (b_j - c) else (j + 1, j + 1, 0.0)
    return j, j + 1, (c - a_j) / (b_j - a_j)


@ALIGNERS.register("boundary_interp", source="white-box-distillation/cross-tokenizer/IMPACT/code/impact_utils.py:665-745")
class BoundaryInterp(Aligner):
    """For every student token's boundaries ``(a_i, b_i)``, the teacher prefix-state indices and weight that
    interpolate the teacher state at that character (IMPACT ``compute_teacher_increments_batch``).

    ``state(c) = (1-w) P[lo] + w P[hi]``, ``P = cat([0, teacher_states[:t_len]])``. Columns: 0 = left boundary
    ``a_i``, 1 = right boundary ``b_i``, for EVERY student position (prompt included, as the original uses the
    attention mask). Position type: positional. ``meta.impact_mode`` records ``mode`` so the ``impact`` criterion
    can check it matches its own. At train time IMPACT is only ``gather + lerp``.
    """
    requires = IDS_OFFSETS
    provides = ("align.impact_lo", "align.impact_hi", "align.impact_w")

    class Config(BaseConfig):
        mode: Literal["linear", "nearest", "containing"] = "linear"

    def empty(self, sample: dict) -> dict[str, Any]:
        n = len(sample["student.input_ids"])
        return {"align.impact_lo": [[0, 0] for _ in range(n)],
                "align.impact_hi": [[0, 0] for _ in range(n)], "align.impact_w": [[0.0, 0.0] for _ in range(n)], "meta.impact_mode": self.cfg.mode}

    def __call__(self, sample: dict, ctx: Any) -> dict[str, Any]:
        t_len = len(sample["teacher.input_ids"])
        t_off = [tuple(o) for o in sample["teacher.offsets"]][:t_len]
        s_off = sample["student.offsets"]
        n = min(len(sample["student.input_ids"]), len(s_off))
        left = boundary_table([int(s_off[i][0]) for i in range(n)], t_off, self.cfg.mode)
        right = boundary_table([int(s_off[i][1]) for i in range(n)], t_off, self.cfg.mode)
        out = self.empty(sample)
        for i in range(n):
            out["align.impact_lo"][i] = [left[i][0], right[i][0]]
            out["align.impact_hi"][i] = [left[i][1], right[i][1]]
            out["align.impact_w"][i] = [left[i][2], right[i][2]]
        return out
