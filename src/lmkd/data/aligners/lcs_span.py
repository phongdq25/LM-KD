"""``lcs_span`` aligner (SRA span pooling indices from a character-boundary LCS)."""
from __future__ import annotations

from typing import Any

import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ALIGNERS
from lmkd.data.aligners.base import IDS_OFFSETS, Aligner, reg_field, unpack

reg_field("align.sra_s_idx", "ragged", torch.long, -1, None, "[N,K_s] student positional indices per span, pad -1")
reg_field("align.sra_t_idx", "ragged", torch.long, -1, None, "[N,K_t] teacher positional indices per span, pad -1")


def longest_common_subsequence(a, b, s_i=0, s_j=0, n_span=1024):
    """Greedy matching of character-boundary positions between two tokenizations.

    Port of IMPACT/code/sra_span_utils.py:15-75 (``N_SPAN`` is the ``n_span`` argument). Compares end offsets, a token
    with end 0 (BOS / special) resets the word boundary. Returns 1-indexed ``(student_idx+1, teacher_idx+1)`` pairs."""
    m, n = len(a), len(b)
    i, j = s_i, s_j
    result = []

    while i < m and j < n:
        if a[i][1] == 0:
            i += 1
            continue
        if b[j][1] == 0:
            j += 1
            continue

        if a[i][1] == b[j][1]:
            result.append((i + 1, j + 1))
            i += 1
            j += 1
        elif a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

        if i + 1 < m and j + 1 < n and (
            (a[i][1] == 0 and a[i + 1][1] > 0) or (b[j][1] == 0 and b[j + 1][1] > 0)
        ):
            while j < n and b[j][1] > 0:
                j += 1
            while i < m and a[i][1] > 0:
                i += 1
            i += 1
            j += 1
            result.append((i, j))

    size = len(result)
    if size > n_span:
        step = size / n_span
        return [result[int((k + 1) * step) - 1] for k in range(n_span)]

    return result


def lcs_segments(s_off: list, t_off: list, s_start: int, t_start: int, n_span: int = 1024
                 ) -> tuple[list[list[int]], list[list[int]]]:
    """Segments of one sample: lists of student / teacher positional indices (one list per span).

    Port of the per-sample body of ``prepare_pooler_v2`` (sra_span_utils.py:151-193) with clips ``len(offsets)``."""
    marks = [(s_start, t_start)] + longest_common_subsequence(s_off, t_off, s_start, t_start, n_span)
    s_seg: list[list[int]] = []
    t_seg: list[list[int]] = []
    for k in range(1, len(marks)):
        s_lo, s_hi = marks[k - 1][0], min(marks[k][0], len(s_off))
        t_lo, t_hi = marks[k - 1][1], min(marks[k][1], len(t_off))
        if s_hi > s_lo and t_hi > t_lo:
            s_seg.append(list(range(s_lo, s_hi)))
            t_seg.append(list(range(t_lo, t_hi)))
    if not s_seg:
        s_seg.append([max(s_start, 0)])
        t_seg.append([max(t_start, 0)])
    return s_seg, t_seg


def pad_rows(rows: list[list[int]], pad: int = -1) -> list[list[int]]:
    """Right-pad every row to the longest one with ``pad``."""
    width = max(len(r) for r in rows)
    return [r + [pad] * (width - len(r)) for r in rows]


@ALIGNERS.register("lcs_span", source="white-box-distillation/cross-tokenizer/IMPACT/code/sra_span_utils.py:15-200")
class LcsSpan(Aligner):
    """Span segments shared by both tokenizations, delimited by common character boundaries (SRA).

    ``sra_s_idx`` ``[N, K_s]`` / ``sra_t_idx`` ``[N, K_t]``: for each of the N spans the positional indices of its
    student / teacher tokens, padded with -1 inside the sample (padding across samples is the criterion's job,
    ``pad_ragged_2d``). Position type: positional. A token whose offset end is 0 (BOS/special) resets the word
    boundary, as in the original.
    """
    requires = IDS_OFFSETS

    provides = ("align.sra_s_idx", "align.sra_t_idx")
    class Config(BaseConfig):
        n_span_max: int = 1024

    def empty(self, sample: dict) -> dict[str, Any]:
        _, ps, _, pt = unpack(sample)
        return {"align.sra_s_idx": [[max(ps, 0)]], "align.sra_t_idx": [[max(pt, 0)]]}

    def __call__(self, sample: dict, ctx: Any) -> dict[str, Any]:
        _, ps, _, pt = unpack(sample)
        s_seg, t_seg = lcs_segments(sample["student.offsets"], sample["teacher.offsets"], ps, pt,
                                    self.cfg.n_span_max)
        return {"align.sra_s_idx": pad_rows(s_seg), "align.sra_t_idx": pad_rows(t_seg)}
