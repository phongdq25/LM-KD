"""Aligner base class and the ``align`` transform (DESIGN §11.4).

POSITION CONVENTIONS used by the aligners (each docstring says which one it uses):
  * target position ``t``: signals computed from logits (the distribution predicting token ``t`` is the
    LogitsView row ``t``);
  * positional ``p``: signals computed from hidden states (``hidden[:, p]`` is the state after reading token ``p``).
The response targets are ``t in [prompt_len, len(ids))``.
"""
from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np

from lmkd.core.batch import FIELDS, FieldSpec
from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import ALIGNERS, TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import Transform

IDS = ("student.input_ids", "student.prompt_len", "teacher.input_ids", "teacher.prompt_len")
IDS_OFFSETS = (*IDS, "student.offsets", "teacher.offsets")


class Aligner(Plugin):
    """``sample -> dict`` of new ``align.*`` keys. Pure CPU, deterministic, picklable, never touches a model."""

    requires: ClassVar[tuple[str, ...]] = IDS

    def __call__(self, sample: dict, ctx: DataContext) -> dict[str, Any]:
        raise NotImplementedError

    def fingerprint(self) -> dict:
        """Content of the aligner for the ``align_precompute`` cache key."""
        return {"type": self.registry_name, **self.cfg.model_dump()}

    def empty(self, sample: dict) -> dict[str, Any]:
        """Values meaning "nothing aligned" (``on_fail='empty'``); same keys as ``__call__``."""
        raise NotImplementedError


def reg_field(key: str, kind: Literal["token", "token_k", "ragged"], dtype: Any, pad: float = 0,
              side: Literal["student", "teacher"] | None = None, doc: str = "") -> None:
    """``FIELDS.register`` shortcut for ``align.*`` keys."""
    FIELDS.register(FieldSpec(key, kind, dtype, pad, side, doc))


def neg_ones(n: int) -> list[int]:
    """``[-1] * n`` (the "not aligned" row of a token field)."""
    return [-1] * n


def as_int_array(x: Any) -> np.ndarray:
    """Sample value (list / array / tensor) -> int64 ndarray."""
    return np.asarray(x, dtype=np.int64)


def unpack(sample: dict) -> tuple[list[int], int, list[int], int]:
    """``(student_ids, student_prompt_len, teacher_ids, teacher_prompt_len)`` as plain python values."""
    return ([int(i) for i in sample["student.input_ids"]], int(sample["student.prompt_len"]),
            [int(i) for i in sample["teacher.input_ids"]], int(sample["teacher.prompt_len"]))


@TRANSFORMS.register("align", source="DESIGN §11.4")
class AlignTransform(Transform):
    """Run one :class:`Aligner` on every sample and merge its ``align.*`` keys into it (in memory or offline).

    ``on_fail='drop'`` removes a sample whose alignment raised ``ValueError``; ``'empty'`` keeps it with the aligner's
    "nothing aligned" values.
    """

    class Config(BaseConfig):
        aligner: dict | str
        on_fail: Literal["drop", "empty"] = "drop"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.aligner: Aligner = ALIGNERS.build(cfg.aligner)

    @classmethod
    def required_keys(cls, cfg: Any) -> tuple[str, ...]:
        meta, acfg = ALIGNERS.validate(cfg.aligner)
        return tuple(meta.obj.required_keys(acfg))

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        meta, acfg = ALIGNERS.validate(cfg.aligner)
        return tuple(meta.obj.provided_keys(acfg))

    def setup(self, rt: Any) -> None:
        self.aligner.setup(rt)

    def __call__(self, sample: dict, ctx: DataContext) -> dict | None:
        try:
            new = self.aligner(sample, ctx)
        except ValueError as err:
            if self.cfg.on_fail == "drop":
                ctx.extra["align_failed"] = ctx.extra.get("align_failed", 0) + 1
                ctx.extra["align_last_error"] = str(err)
                return None
            new = self.aligner.empty(sample)
        out = dict(sample)
        out.update(new)
        return out
