"""``char_span`` aligner (E-SCOT character-span neighbourhoods)."""
from __future__ import annotations

from typing import Any, Literal

import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ALIGNERS
from lmkd.data.aligners.base import IDS_OFFSETS, Aligner, reg_field, unpack

reg_field("align.span_pairs", "ragged", torch.long, 0, None, "[P,2] (student target t, teacher target t) overlaps")


def _span_align(s_spans, t_spans):
    """Algorithm 1 of paper 15849 - linear two-pointer sweep for character-span overlap.

    Verbatim port of IMPACT/code/criterions/E_SCOT.py:283-302. Entry ``i`` lists the teacher indices whose span
    ``[t_start, t_end)`` intersects student span ``i``."""
    neighborhoods: list[list[int]] = [[] for _ in range(len(s_spans))]
    if not t_spans:
        return neighborhoods
    j0 = 0
    for i, (st_i, ed_i) in enumerate(s_spans):
        # Advance j0 past teacher tokens that fully precede student token i.
        while j0 < len(t_spans) and t_spans[j0][1] <= st_i:
            j0 += 1
        # Collect all teacher tokens whose start is before student i's end.
        k = j0
        while k < len(t_spans) and t_spans[k][0] < ed_i:
            neighborhoods[i].append(k)
            k += 1
    return neighborhoods


@ALIGNERS.register("char_span", source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/E_SCOT.py:176-195,283-302")
class CharSpan(Aligner):
    """Pairs of (student target, teacher target) whose character spans overlap (E-SCOT neighbourhoods).

    ``span_of='context'`` (E-SCOT): the span of target ``t`` is the offset of the CONTEXT token ``t-1`` (the token the
    distribution is conditioned on, ``E_SCOT.py:192``); ``'target'`` uses ``offsets[t]``. Targets are
    ``t in [prompt_len, len)``. Output ``align.span_pairs`` ``[P,2]`` = (t_s, t_t), target positions.
    """
    requires = IDS_OFFSETS

    provides = ("align.span_pairs",)
    class Config(BaseConfig):
        span_of: Literal["context", "target"] = "context"

    def empty(self, sample: dict) -> dict[str, Any]:
        return {"align.span_pairs": []}

    def __call__(self, sample: dict, ctx: Any) -> dict[str, Any]:
        s_ids, ps, t_ids, pt = unpack(sample)
        shift = 1 if self.cfg.span_of == "context" else 0
        s_t = list(range(max(ps, shift), len(s_ids)))
        t_t = list(range(max(pt, shift), len(t_ids)))
        s_off, t_off = sample["student.offsets"], sample["teacher.offsets"]
        s_spans = [tuple(s_off[t - shift]) for t in s_t]
        t_spans = [tuple(t_off[t - shift]) for t in t_t]
        pairs = [[s_t[i], t_t[j]] for i, nb in enumerate(_span_align(s_spans, t_spans)) for j in nb]
        return {"align.span_pairs": pairs}
