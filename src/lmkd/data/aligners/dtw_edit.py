"""``dtw_edit`` aligner (MinED / MCW-KD dynamic time warping with edit-distance cost)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ALIGNERS
from lmkd.data.aligners.base import IDS, Aligner, neg_ones, reg_field, unpack

reg_field("align.dtw_t_for_s", "token", torch.long, -1, "student", "MinED: teacher target aligned 1:1 to student t")
reg_field("align.dtw_s2t", "ragged", torch.long, 0, None, "MCW: [P,2] (student pos, teacher pos) DTW path")
reg_field("align.dtw_s2t_cost", "ragged", torch.float32, 0.0, None, "MCW: [P] edit distance along dtw_s2t")
reg_field("align.dtw_t2s", "ragged", torch.long, 0, None, "MCW: [P,2] (teacher pos, student pos) DTW path")
reg_field("align.dtw_t2s_cost", "ragged", torch.float32, 0.0, None, "MCW: [P] edit distance along dtw_t2s")


def dtw(series_1, series_2, norm_func=np.linalg.norm):
    """Dynamic time warping used by MinED (modified from talcs/simpledtw).

    Verbatim port of IMPACT/code/criterions/MinED.py:368-411. Returns ``(matches, cost, mappings_series_1,
    mappings_series_2, matrix)``; ``mappings_series_2[j]`` lists the ``series_1`` indices matched to ``series_2[j]``."""
    matrix: np.ndarray = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
            )
    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1: list[list[int]] = [[] for v in range(matrix.shape[0])]
    mappings_series_2: list[list[int]] = [[] for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix


def dtw_alignment(series_1, series_2, norm_func):
    """Simple DTW path (FUSELLM style), list of ``(i, j)`` from the end of both series to ``(0, 0)``.

    Verbatim port of IMPACT/code/criterions/MCW_KD.py:433-470."""
    matrix: np.ndarray = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
            )
    matrix = matrix[1:, 1:]
    i, j = len(series_1) - 1, len(series_2) - 1
    aligned = []
    while i > 0 or j > 0:
        aligned.append((i, j))
        options = [
            matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
            matrix[i - 1, j] if i > 0 else np.inf,
            matrix[i, j - 1] if j > 0 else np.inf,
        ]
        move = np.argmin(options)
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    aligned.append((0, 0))
    return aligned


def space_marker(tokenizer: Any) -> str:
    """``"Ġ"`` if the vocab has a token starting with it (byte-level BPE), else ``"▁"``.

    Detected from the vocab instead of MinED's ``TOKENIZER_TO_SPECIAL_TOKEN[tokenizer.__class__]``, which has no entry
    for other tokenizer classes."""
    return "Ġ" if any(t.startswith("Ġ") for t in tokenizer.get_vocab()) else "▁"


@ALIGNERS.register("dtw_edit", source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/MinED.py:107-182,368-411",
                   extras=("ot",))
class DtwEdit(Aligner):
    """DTW alignment of two token sequences with edit-distance cost (MinED; ``full_sequence`` = MCW-KD).

    MinED (``dtw_t_for_s``, target positions): DTW over the CONTEXT tokens of the response targets
    (``ids[plen-1 : L-1]``, since target ``t`` is predicted from token ``t-1``); a student token matched to exactly one
    teacher token keeps the pair if ``require_id_map`` allows it (``student_id == vocab_map.tea2stu[teacher_id]``,
    MinED.py:182). MCW (``full_sequence``): two DTWs over ALL positions (positional), ``dtw_s2t``/``dtw_t2s`` paths and
    their edit costs. Fixed vs the original: special tokens are NOT skipped (``skip_special_tokens=True`` shortened
    the token lists and shifted every index).
    """
    requires = IDS

    class Config(BaseConfig):
        full_sequence: bool = False
        require_id_map: bool = True

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        keys: tuple[str, ...] = ("align.dtw_t_for_s",)
        if cfg is not None and cfg.full_sequence:
            keys += ("align.dtw_s2t", "align.dtw_s2t_cost", "align.dtw_t2s", "align.dtw_t2s_cost")
        return keys

    def __init__(self, cfg: Any = None) -> None:
        super().__init__(cfg)
        self._markers: dict[tuple, str] = {}

    def _marker(self, tok: Any) -> str:
        key = (getattr(tok, "name_or_path", ""), len(tok))
        if key not in self._markers:
            self._markers[key] = space_marker(tok)
        return self._markers[key]

    def empty(self, sample: dict) -> dict[str, Any]:
        s_ids, _, t_ids, _ = unpack(sample)
        out: dict[str, Any] = {"align.dtw_t_for_s": neg_ones(len(s_ids))}
        if self.cfg.full_sequence:
            out.update({"align.dtw_s2t": [], "align.dtw_s2t_cost": [], "align.dtw_t2s": [], "align.dtw_t2s_cost": []})
        return out

    def _minED(self, sample: dict, ctx: Any) -> list[int]:
        import editdistance

        s_ids, ps, t_ids, pt = unpack(sample)
        out = neg_ones(len(s_ids))
        a_s, a_t = max(ps - 1, 0), max(pt - 1, 0)       # context index of the first response target
        ctx_s, ctx_t = s_ids[a_s: len(s_ids) - 1], t_ids[a_t: len(t_ids) - 1]
        if not ctx_s or not ctx_t:
            return out
        stu, tea = ctx.student_tokenizer, ctx.teacher_tokenizer
        sp_s, sp_t = self._marker(stu), self._marker(tea)

        def dist_fn(a, b):  # a: teacher (blending) token, b: student (base) token
            return editdistance.eval(a.replace(sp_t, ""), b.replace(sp_s, ""))

        _, _, _, base_to_blend, _ = dtw(tea.convert_ids_to_tokens(ctx_t), stu.convert_ids_to_tokens(ctx_s),
                                        norm_func=dist_fn)
        tea2stu = self._tea2stu(ctx) if self.cfg.require_id_map else None
        for k, js in enumerate(base_to_blend):
            if len(js) != 1:
                continue
            if tea2stu is not None and ctx_s[k] != int(tea2stu[ctx_t[js[0]]]):
                continue
            out[a_s + 1 + k] = a_t + 1 + js[0]
        return out

    @staticmethod
    def _tea2stu(ctx: Any) -> torch.Tensor:
        if ctx.vocab_map is None:
            raise RuntimeError("dtw_edit(require_id_map=true) needs a vocab map: add the `vocab_map` stage or the "
                               "`load_vocab_map` transform before `align`")
        return ctx.vocab_map.tea2stu

    def _mcw(self, sample: dict, ctx: Any) -> dict[str, Any]:
        import editdistance

        s_ids, _, t_ids, _ = unpack(sample)
        stu_tok = ctx.student_tokenizer.convert_ids_to_tokens(s_ids)
        tea_tok = ctx.teacher_tokenizer.convert_ids_to_tokens(t_ids)
        s2t = dtw_alignment(stu_tok, tea_tok, editdistance.eval)
        t2s = dtw_alignment(tea_tok, stu_tok, editdistance.eval)
        return {"align.dtw_s2t": [list(p) for p in s2t],
                "align.dtw_s2t_cost": [float(editdistance.eval(stu_tok[i], tea_tok[j])) for i, j in s2t],
                "align.dtw_t2s": [list(p) for p in t2s],
                "align.dtw_t2s_cost": [float(editdistance.eval(tea_tok[j], stu_tok[i])) for j, i in t2s]}

    def __call__(self, sample: dict, ctx: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"align.dtw_t_for_s": self._minED(sample, ctx)}
        if self.cfg.full_sequence:
            out.update(self._mcw(sample, ctx))
        return out
