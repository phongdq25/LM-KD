"""``parent_groups`` aligner (CTPD: groups of tokens between shared character boundaries)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ALIGNERS
from lmkd.data.aligners.base import IDS_OFFSETS, Aligner, reg_field, unpack
from lmkd.data.aligners.lcs_span import pad_rows

reg_field("align.parent_groups_s", "ragged", torch.long, -1, None, "[G,K_s] student positional indices per parent")
reg_field("align.parent_groups_t", "ragged", torch.long, -1, None, "[G,K_t] teacher positional indices per parent")


def common_parent_groups(student_offsets: Sequence[Sequence[int]], teacher_offsets: Sequence[Sequence[int]]
                         ) -> tuple[list[list[int]], list[list[int]]]:
    """Group two tokenizations into spans delimited by shared character boundaries.

    Port of Awesome-Human-AI-Alignment-main/src/human_alignment/supervision/preference_distillation.py:65
    (``DatasetFormatError`` -> ``ValueError``). Offsets are relative to the start of the response."""
    if not student_offsets or not teacher_offsets:
        raise ValueError("CTPD requires non-empty response tokenizations")
    end = max(int(student_offsets[-1][1]), int(teacher_offsets[-1][1]))
    student_boundaries = {0, end, *(int(offset[1]) for offset in student_offsets)}
    teacher_boundaries = {0, end, *(int(offset[1]) for offset in teacher_offsets)}
    boundaries = sorted(student_boundaries & teacher_boundaries)
    if boundaries[0] != 0 or boundaries[-1] != end:
        raise ValueError("Tokenizers do not cover the same response span")

    def groups(offsets: Sequence[Sequence[int]]) -> list[list[int]]:
        result = []
        for start, stop in zip(boundaries, boundaries[1:]):
            indices = [
                index
                for index, (token_start, token_stop) in enumerate(offsets)
                if int(token_stop) > start and int(token_stop) <= stop
            ]
            if not indices:
                raise ValueError("A shared CTPD span contains no tokens")
            result.append(indices)
        return result

    return groups(student_offsets), groups(teacher_offsets)


@ALIGNERS.register("parent_groups", source="Awesome-Human-AI-Alignment-main/src/human_alignment/supervision/"
                   "preference_distillation.py:65")
class ParentGroups(Aligner):
    """Student / teacher response tokens grouped by shared character boundaries (CTPD ``common_parent_groups``).

    Writes ``parent_groups_s`` ``[G,K_s]`` and ``parent_groups_t`` ``[G,K_t]`` (positional indices into the full
    sequences, pad -1; group g of both sides covers the same characters). Offsets of the response are rebased to its
    start (``meta.prompt_text`` length, else the first response token). Works on ONE text: for chosen/rejected pairs
    T22 calls the aligner once per side. Position type: positional.
    """
    requires = IDS_OFFSETS

    provides = ("align.parent_groups_s", "align.parent_groups_t")
    class Config(BaseConfig):
        pass

    def empty(self, sample: dict) -> dict[str, Any]:
        _, ps, _, pt = unpack(sample)
        return {"align.parent_groups_s": [[ps]], "align.parent_groups_t": [[pt]]}

    @staticmethod
    def _rebased(offsets: list, plen: int, base: int) -> list[list[int]]:
        return [[max(int(s) - base, 0), max(int(e) - base, 0)] for s, e in offsets[plen:]]

    def __call__(self, sample: dict, ctx: Any) -> dict[str, Any]:
        _, ps, _, pt = unpack(sample)
        s_off, t_off = sample["student.offsets"], sample["teacher.offsets"]
        s_base = len(sample["meta.prompt_text"]) if "meta.prompt_text" in sample else int(s_off[ps][0])
        t_text = sample.get("meta.teacher_prompt_text", sample.get("meta.prompt_text"))
        t_base = len(t_text) if t_text is not None else int(t_off[pt][0])
        s_groups, t_groups = common_parent_groups(self._rebased(s_off, ps, s_base), self._rebased(t_off, pt, t_base))
        return {"align.parent_groups_s": pad_rows([[ps + i for i in g] for g in s_groups]),
                "align.parent_groups_t": pad_rows([[pt + i for i in g] for g in t_groups])}
