"""``exact_token`` aligner (DSKDv2 exact token alignment, ETA)."""
from __future__ import annotations

from typing import Any

import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ALIGNERS
from lmkd.data.aligners.base import IDS, Aligner, neg_ones, reg_field, unpack

reg_field("align.eta_t_for_s", "token", torch.long, -1, "student", "teacher target position aligned to student t")
reg_field("align.eta_s_for_t", "token", torch.long, -1, "teacher", "student target position aligned to teacher t")


def align_sequences(tea_seq, stu_seq, student_tokenizer, teacher_tokenizer):
    """Two-pointer exact alignment of two token-string sequences.

    Verbatim port of DSKDv2-main/code/criterions/dual_space_kd_v2_with_exact_token_alignment.py:8-41 (spaces markers
    ``▁``/``Ġ`` are dropped; ``eos`` strings of both tokenizers match each other)."""
    i, j = 0, 0
    t2s_align, s2t_align = [], []
    history_tea_seq, history_stu_seq = "", ""

    tea_seq = [token.replace('▁', '').replace('Ġ', '') for token in tea_seq]
    stu_seq = [token.replace('▁', '').replace('Ġ', '') for token in stu_seq]

    while i < len(tea_seq) and j < len(stu_seq):
        if history_tea_seq == history_stu_seq and (
            tea_seq[i] == stu_seq[j] or (
                tea_seq[i] == teacher_tokenizer.eos_token and
                stu_seq[j] == student_tokenizer.eos_token
            )
        ):
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            t2s_align.append(i)
            s2t_align.append(j)
            i += 1
            j += 1
        elif len(history_tea_seq) > len(history_stu_seq):
            history_stu_seq += stu_seq[j]
            j += 1
        elif len(history_tea_seq) < len(history_stu_seq):
            history_tea_seq += tea_seq[i]
            i += 1
        else:
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            i += 1
            j += 1

    return t2s_align, s2t_align


@ALIGNERS.register("exact_token", source="white-box-distillation/cross-tokenizer/DSKDv2-main/code/criterions/"
                   "dual_space_kd_v2_with_exact_token_alignment.py:8-41,138-150")
class ExactToken(Aligner):
    """1:1 exact token pairs of the response (ETA), as target positions of both sequences.

    Runs ``align_sequences`` on the token strings of ``ids[prompt_len:]`` and writes
    ``eta_t_for_s[plen_s + sj] = plen_t + ti`` and ``eta_s_for_t[plen_t + ti] = plen_s + sj`` (-1 elsewhere).
    Position type: target (logits based).
    """
    requires = IDS

    provides = ("align.eta_t_for_s", "align.eta_s_for_t")
    class Config(BaseConfig):
        pass

    def empty(self, sample: dict) -> dict[str, Any]:
        s_ids, _, t_ids, _ = unpack(sample)
        return {"align.eta_t_for_s": neg_ones(len(s_ids)), "align.eta_s_for_t": neg_ones(len(t_ids))}

    def __call__(self, sample: dict, ctx: Any) -> dict[str, Any]:
        s_ids, ps, t_ids, pt = unpack(sample)
        stu, tea = ctx.student_tokenizer, ctx.teacher_tokenizer
        out = self.empty(sample)
        if ps >= len(s_ids) or pt >= len(t_ids):
            return out
        t_al, s_al = align_sequences(tea.convert_ids_to_tokens(t_ids[pt:]), stu.convert_ids_to_tokens(s_ids[ps:]),
                                     stu, tea)
        for ti, sj in zip(t_al, s_al):
            out["align.eta_t_for_s"][ps + sj] = pt + ti
            out["align.eta_s_for_t"][pt + ti] = ps + sj
        return out
