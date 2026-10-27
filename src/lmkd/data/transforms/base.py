"""Transform = ``sample -> sample | None`` (None drops the sample). DESIGN §4.4.

Offline transforms run once (in-memory now; Arrow-cached by the pipeline later, task T17).
Transforms flagged ``online = True`` run in ``Dataset.__getitem__`` (cheap / random ones only).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from lmkd.core.plugin import Plugin
from lmkd.data.context import DataContext


class Transform(Plugin):
    """``sample -> sample | None`` step of the data chain (``None`` drops the sample); declare ``requires`` /
    ``provides`` batch keys.
    """
    online: ClassVar[bool] = False

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        raise NotImplementedError


class CorpusTransform(Transform):
    """Transform that fits statistics on the entire collection of samples before per-sample processing."""

    def fit(self, samples: list[dict[str, Any]], ctx: DataContext) -> None:
        pass


def run_transforms(records: Iterable[dict], transforms: list[Transform], ctx: DataContext) -> list[dict]:
    current: list[dict] = list(records)
    for t in transforms:
        if t.online:
            continue
        if isinstance(t, CorpusTransform):
            t.fit(current, ctx)
        next_samples = []
        for s in current:
            res = t(s, ctx)
            if res is not None:
                next_samples.append(res)
        current = next_samples
    return current
