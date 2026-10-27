"""In-memory prepared dataset. (Arrow-backed cached version: task T17.)"""
from __future__ import annotations

from typing import Any

import torch

from lmkd.data.context import DataContext
from lmkd.data.transforms.base import Transform


class PreparedDataset(torch.utils.data.Dataset):
    """Map-style dataset over already transformed samples; applies only ``online`` transforms per item."""
    def __init__(self, samples: list[dict[str, Any]], online: list[Transform] | None = None,
                 ctx: DataContext | None = None):
        self.samples = samples
        self.online = [t for t in (online or []) if t.online]
        self.ctx = ctx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = dict(self.samples[i])
        for t in self.online:
            assert self.ctx is not None, "online transforms need a DataContext"
            out = t(s, self.ctx)
            if out is None:
                raise ValueError(f"online transform {type(t).__name__} dropped a sample (only offline transforms may)")
            s = out
        return s
