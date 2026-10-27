"""Vocabulary map between two tokenizers: overlap, id mappings, space masks (DESIGN §11.4)."""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import Transform


def normalize_token(tok: str) -> str:
    """GPT-2 ``Ġ`` -> SentencePiece ``▁`` (DSKDv2 distiller.py:130, IMPACT impact_utils.py:19)."""
    return tok.replace("Ġ", "▁")


def space_mask(tokenizer: Any, mode: str, size: int) -> torch.Tensor:
    """Bool mask ``[size]`` of tokens by the ``+``-joined ``mode`` (space, tab, newline, special).

    Port of tokenkit/utils.py:76-99 ``get_space_mask`` (keeps tokenkit's names: ``tab`` <-> ``Ċ``, ``newline`` <->
    ``ĉ``) on ``convert_ids_to_tokens``; the tail past ``len(tokenizer)`` (padded embeddings) is False.
    """
    n = len(tokenizer)
    out = np.zeros(max(size, n), dtype=bool)
    tokens = tokenizer.convert_ids_to_tokens(list(range(n)))
    special = set(tokenizer.all_special_ids)
    modes = set(mode.split("+"))
    first = {"space": "Ġ", "tab": "Ċ", "newline": "ĉ"}
    for i, tok in enumerate(tokens):
        if not tok:
            continue
        if any(m in modes and tok[0] == ch for m, ch in first.items()) or ("special" in modes and i in special):
            out[i] = True
    return torch.from_numpy(out[:size] if size else out)


def _byteify(tokenizer: Any, spec: str) -> Any:
    """tokenkit's byteified tokenizer for ``spec`` when tokenkit is installed, else the plain tokenizer."""
    if not spec:
        return tokenizer
    try:
        from tokenkit.byteify import load_byteify_tokenizer
    except ImportError:
        return tokenizer
    return load_byteify_tokenizer(spec)


def _normalized_vocab(tokenizer: Any) -> dict[str, int]:
    """``{normalized token: id}`` iterated in id order (a fast tokenizer's ``get_vocab()`` order is not stable)."""
    return {normalize_token(k): v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])}


def _first_inverse(tea2stu: torch.Tensor, n_stu: int) -> torch.Tensor:
    """student id -> first (lowest) teacher id mapping to it; -1 if none (``max_align_num = 1``)."""
    stu2tea = torch.full((n_stu,), -1, dtype=torch.long)
    for tid in range(len(tea2stu)):
        sid = int(tea2stu[tid])
        if 0 <= sid < n_stu and stu2tea[sid] < 0:
            stu2tea[sid] = tid
    return stu2tea


@dataclass
class VocabMap:
    """Overlap / id mapping between student and teacher vocabularies.

    ``overlap_s/t`` follow the student's vocab in id order (DSKDv2 ``distiller.py:130-133`` iterates ``get_vocab()``,
    whose order is only stable for slow tokenizers; id order is the same there; the list is truncated by
    ``topk_vocab``); ``overlap_sorted_*`` are sorted by normalized token string (IMPACT ``impact_utils.py:121``).
    ``tea2stu`` comes from a JSON file (``{"teacher_id": student_id}``, built by id, not by dict order) or from exact
    normalized-string match; ``stu2tea`` keeps the first teacher id (``IMPACT distiller.py:41-60``);
    ``tea2stu_exact`` is ``student.convert_tokens_to_ids(teacher_token)`` (ETA; a fast tokenizer returns its unk id
    for unknown tokens, kept as in the original).
    """
    student_vocab_size: int
    teacher_vocab_size: int
    overlap_s: torch.Tensor
    overlap_t: torch.Tensor
    overlap_sorted_s: torch.Tensor
    overlap_sorted_t: torch.Tensor
    tea2stu: torch.Tensor
    stu2tea: torch.Tensor
    tea2stu_exact: torch.Tensor
    same_vocab: bool
    space_mask_s: torch.Tensor | None = None
    space_mask_t: torch.Tensor | None = None

    def save(self, path: Path | str) -> None:
        """``torch.save`` of every field."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({f.name: getattr(self, f.name) for f in fields(self)}, p)

    @classmethod
    def load(cls, path: Path | str) -> VocabMap:
        """Inverse of :meth:`save`."""
        return cls(**torch.load(Path(path), map_location="cpu", weights_only=True))

    @classmethod
    def build(cls, stu_tok: Any, tea_tok: Any, *, mapping_file: str = "", space_mask_mode: str = "",
              student_vocab_size: int = 0, teacher_vocab_size: int = 0, student_byteify: str = "",
              teacher_byteify: str = "") -> VocabMap:
        """Build from two tokenizers. ``*_vocab_size`` = model vocab size (0 -> ``len(tokenizer)``); the byteify specs
        are only used for the space masks when tokenkit is installed."""
        n_s = student_vocab_size or len(stu_tok)
        n_t = teacher_vocab_size or len(tea_tok)
        sv = _normalized_vocab(stu_tok)
        tv = _normalized_vocab(tea_tok)
        tokens = [k for k in sv if k in tv]
        srt = sorted(set(sv) & set(tv))
        tea2stu = cls._tea2stu(tea_tok, sv, n_t, mapping_file)
        sm = tm = None
        if space_mask_mode:
            sm = space_mask(_byteify(stu_tok, student_byteify), space_mask_mode, n_s)
            tm = space_mask(_byteify(tea_tok, teacher_byteify), space_mask_mode, n_t)
        return cls(
            n_s, n_t,
            torch.tensor([sv[k] for k in tokens], dtype=torch.long), torch.tensor([tv[k] for k in tokens], dtype=torch.long),
            torch.tensor([sv[k] for k in srt], dtype=torch.long), torch.tensor([tv[k] for k in srt], dtype=torch.long),
            tea2stu, _first_inverse(tea2stu, n_s), cls._tea2stu_exact(stu_tok, tea_tok, n_t),
            stu_tok.get_vocab() == tea_tok.get_vocab(), sm, tm)

    @staticmethod
    def _tea2stu(tea_tok: Any, sv: dict[str, int], n_t: int, mapping_file: str) -> torch.Tensor:
        out = torch.full((n_t,), -1, dtype=torch.long)
        if mapping_file:
            for tid, sid in json.loads(Path(mapping_file).read_text()).items():
                if 0 <= int(tid) < n_t:
                    out[int(tid)] = int(sid)
            return out
        for tok, tid in sorted(tea_tok.get_vocab().items(), key=lambda kv: kv[1]):
            if tid < n_t and normalize_token(tok) in sv:
                out[tid] = sv[normalize_token(tok)]
        return out

    @staticmethod
    def _tea2stu_exact(stu_tok: Any, tea_tok: Any, n_t: int) -> torch.Tensor:
        toks = tea_tok.convert_ids_to_tokens(list(range(min(n_t, len(tea_tok)))))
        ids = stu_tok.convert_tokens_to_ids(toks)
        out = torch.full((n_t,), -1, dtype=torch.long)
        for tid, sid in enumerate(ids):
            out[tid] = -1 if sid is None else int(sid)
        return out


def vocab_map_kwargs(cfg: Any) -> dict[str, Any]:
    """``VocabMap.build`` keyword arguments from a ``vocab_map`` stage / ``load_vocab_map`` transform Config."""
    return {"mapping_file": cfg.mapping_file, "space_mask_mode": cfg.space_mask_mode,
            "student_vocab_size": cfg.student_vocab_size, "teacher_vocab_size": cfg.teacher_vocab_size,
            "student_byteify": cfg.student_byteify, "teacher_byteify": cfg.teacher_byteify}


@TRANSFORMS.register("load_vocab_map", source="IMPACT/code/distiller.py:41-60")
class LoadVocabMap(Transform):
    """Load (or build and cache) the student/teacher :class:`VocabMap` into ``rt.data_ctx.vocab_map`` at setup;
    samples pass through unchanged. Put it before any ``align`` transform (no pipeline needed).
    """

    class Config(BaseConfig):
        kind: str = "overlap"
        mapping_file: str = ""
        space_mask_mode: str = ""
        student_vocab_size: int = 0
        teacher_vocab_size: int = 0
        student_byteify: str = ""
        teacher_byteify: str = ""
        cache_path: str | None = None

    def setup(self, rt: Any) -> None:
        if rt.data_ctx.vocab_map is not None:
            return
        path = Path(self.cfg.cache_path) if self.cfg.cache_path else None
        if path is not None and path.exists():
            rt.data_ctx.vocab_map = VocabMap.load(path)
            return
        if rt.teacher_tokenizer is None:
            raise ValueError("load_vocab_map needs models.teacher (a teacher tokenizer)")
        vm = VocabMap.build(rt.student_tokenizer, rt.teacher_tokenizer, **vocab_map_kwargs(self.cfg))
        if path is not None:
            vm.save(path)
        rt.data_ctx.vocab_map = vm

    def __call__(self, sample: dict, ctx: DataContext) -> dict | None:
        return sample
