"""``tokenkit_chunk`` aligner (ALM chunk alignment via tokenkit)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ALIGNERS
from lmkd.data.aligners.base import IDS, Aligner, neg_ones, reg_field, unpack

reg_field("align.chunk_s", "token", torch.long, -1, "student", "chunk id of student position p (-1 = none)")
reg_field("align.chunk_t", "token", torch.long, -1, "teacher", "chunk id of teacher position p (-1 = none)")


def load_pair_data(path: str) -> tuple[Any, Any, None, None]:
    """Bias matrices of the unbiased alignment (``bias1_matrix.npz`` / ``bias2_matrix.npz``, IMPACT
    distillation_alm.py:128-143)."""
    from scipy import sparse  # extra eval-emb

    base = Path(path)
    return (sparse.load_npz(base / "bias1_matrix.npz").todok(), sparse.load_npz(base / "bias2_matrix.npz").todok(),
            None, None)


def chunk_ids(matrix: np.ndarray, length: int) -> list[int]:
    """``[L]`` chunk id per position from a ``[1, L, C]`` bool matrix: argmax of its row, -1 if the row is empty.

    Each token belongs to at most one chunk (tokenkit alignment indices are disjoint ranges)."""
    rows = matrix[0, :length]
    if (rows.sum(-1) > 1).any():
        raise ValueError("tokenkit alignment put a token into more than one chunk")
    return [int(r.argmax()) if r.any() else -1 for r in rows]


@ALIGNERS.register("tokenkit_chunk", source="white-box-distillation/cross-tokenizer/IMPACT/code/distillation_alm.py:184-276",
                   extras=("tokenkit",))
class TokenkitChunk(Aligner):
    """Chunk id of every token from tokenkit's (un)biased alignment, for the ALM criterion.

    Calls ``tokenkit.align.get_unconstrained_alignments`` / ``get_unbiased_alignments`` on ONE sample and keeps, per
    position, the single chunk it belongs to (-1 if none). ``exclude_last`` (default) runs tokenkit on ``ids[:-1]``
    like ALMCollator (the last token, usually EOS, is never an input) and sets ``chunk_*[L-1] = -1``, so every
    positional ``p`` in a chunk has its logits row ``p+1``. Position type: positional.
    Needs tokenkit (``pip install -e IMPACT/tokenkit-main``) and byteify specs such as ``"gpt2:source=gpt2"``.
    """
    requires = IDS

    provides = ("align.chunk_s", "align.chunk_t")
    class Config(BaseConfig):
        alignment: Literal["unconstrained", "unbiased"] = "unconstrained"
        student_byteify: str = ""
        teacher_byteify: str = ""
        pair_data_path: str = ""
        bias_threshold: float = 0.1
        exclude_last: bool = True

    def __init__(self, cfg: Any = None) -> None:
        super().__init__(cfg)
        self._tok: tuple[Any, Any] | None = None
        self._pair_data: Any = None

    def empty(self, sample: dict) -> dict[str, Any]:
        s_ids, _, t_ids, _ = unpack(sample)
        return {"align.chunk_s": neg_ones(len(s_ids)), "align.chunk_t": neg_ones(len(t_ids))}

    def byteify_tokenizers(self) -> tuple[Any, Any]:
        """(student, teacher) tokenkit byteified tokenizers, loaded once per process (override in tests)."""
        if self._tok is None:
            from tokenkit.byteify import load_byteify_tokenizer

            if not (self.cfg.student_byteify and self.cfg.teacher_byteify):
                raise RuntimeError("tokenkit_chunk needs student_byteify and teacher_byteify specs")
            self._tok = (load_byteify_tokenizer(self.cfg.student_byteify),
                         load_byteify_tokenizer(self.cfg.teacher_byteify))
        return self._tok

    def _matrices(self, s_ids: list[int], t_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        from tokenkit import align

        stu, tea = self.byteify_tokenizers()
        args = {"input_ids_teacher": np.asarray([t_ids], dtype=np.int64),
                "input_ids_student": np.asarray([s_ids], dtype=np.int64),
                "attention_mask_teacher": np.ones((1, len(t_ids)), dtype=np.int64),
                "attention_mask_student": np.ones((1, len(s_ids)), dtype=np.int64),
                "tokenizer_teacher": tea, "tokenizer_student": stu}
        if self.cfg.alignment == "unbiased":
            if self._pair_data is None:
                self._pair_data = load_pair_data(self.cfg.pair_data_path)
            return align.get_unbiased_alignments(**args, pair_data=self._pair_data,
                                                 bias_threshold=self.cfg.bias_threshold)
        return align.get_unconstrained_alignments(**args)

    def __call__(self, sample: dict, ctx: Any) -> dict[str, Any]:
        s_ids, _, t_ids, _ = unpack(sample)
        out = self.empty(sample)
        cut = 1 if self.cfg.exclude_last else 0
        if len(s_ids) - cut <= 0 or len(t_ids) - cut <= 0:
            return out
        m_s, m_t = self._matrices(s_ids[: len(s_ids) - cut], t_ids[: len(t_ids) - cut])
        out["align.chunk_s"][: len(s_ids) - cut] = chunk_ids(m_s, len(s_ids) - cut)
        out["align.chunk_t"][: len(t_ids) - cut] = chunk_ids(m_t, len(t_ids) - cut)
        return out
