"""Replay buffer for off-policy/on-policy samples (DESIGN §7.2, §12.1).

Stores unpadded sample dictionaries (CPU Python structures/tensors) to remain model-agnostic.
Sampling returns unpadded samples which are then collated by the runtime collator.
"""
from __future__ import annotations

import collections
import random
from typing import Any

import torch

from lmkd.core.batch import FIELDS, Batch


class ReplayBuffer:
    """FIFO replay buffer for unpadded sample dicts with deterministic sampling."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.seed = seed
        self.rng = random.Random(seed)
        self.buffer: collections.deque[dict[str, Any]] = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_samples(self, samples: list[dict[str, Any]]) -> None:
        """Add unpadded sample dicts to buffer with FIFO eviction."""
        for s in samples:
            self.buffer.append(s)

    def sample(self, k: int) -> list[dict[str, Any]]:
        """Sample k items uniformly at random without replacement."""
        if k > len(self.buffer):
            raise ValueError(f"cannot sample {k} items from buffer of size {len(self.buffer)}")
        return self.rng.sample(list(self.buffer), k)

    def state_dict(self, include_data: bool = False) -> dict[str, Any]:
        """Serialize buffer state, including RNG state and optionally buffer items."""
        d: dict[str, Any] = {
            "capacity": self.capacity,
            "seed": self.seed,
            "rng": self.rng.getstate(),
        }
        if include_data:
            d["data"] = list(self.buffer)
        return d

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore buffer state from serialized dict."""
        self.capacity = d.get("capacity", self.capacity)
        self.seed = d.get("seed", self.seed)
        if "rng" in d:
            self.rng.setstate(d["rng"])
        self.buffer = collections.deque(maxlen=self.capacity)
        if "data" in d:
            for s in d["data"]:
                self.buffer.append(s)


def batch_to_samples(batch: Batch) -> list[dict[str, Any]]:
    """Unpack a collated Batch into unpadded per-sample dictionaries.

    Token and token_k fields are sliced by the corresponding side's attention mask sum,
    scalars converted via .item(), vectors detached to CPU, and metadata copied per sample.
    """
    n = batch.batch_size
    if n == 0:
        return []

    side_lengths: dict[str, list[int]] = {}
    for side in ("student", "teacher"):
        mask_key = f"{side}.attention_mask"
        if mask_key in batch:
            mask = batch[mask_key]
            if isinstance(mask, torch.Tensor):
                side_lengths[side] = [int(mask[i].sum().item()) for i in range(n)]

    samples: list[dict[str, Any]] = [{} for _ in range(n)]

    for key, val in batch.items():
        if key.startswith("meta."):
            for i in range(n):
                samples[i][key] = val[i]
            continue
        try:
            spec = FIELDS.get(key)
            kind = spec.kind
            field_side = spec.side
        except KeyError:
            kind = "meta"
            field_side = None

        if kind in ("token", "token_k"):
            for i in range(n):
                seq = val[i]
                if field_side is not None and field_side in side_lengths:
                    real_len = side_lengths[field_side][i]
                else:
                    real_len = seq.shape[0] if isinstance(seq, torch.Tensor) else len(seq)
                sliced = seq[:real_len]
                if isinstance(sliced, torch.Tensor):
                    samples[i][key] = sliced.detach().cpu().tolist()
                else:
                    samples[i][key] = list(sliced)
        elif kind == "scalar":
            for i in range(n):
                v = val[i]
                samples[i][key] = v.item() if isinstance(v, torch.Tensor) else v
        elif kind in ("vector", "ragged"):
            for i in range(n):
                v = val[i]
                samples[i][key] = v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
        elif kind == "meta":
            for i in range(n):
                samples[i][key] = val[i]

    for key, vals in batch.meta.items():
        for i in range(n):
            samples[i][key] = vals[i]

    return samples
