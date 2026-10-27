"""Sequence packing and deterministic length-aware samplers (DESIGN §11.5)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from lmkd.core.batch import FIELDS, Batch, FieldSpec
from lmkd.data.collator import Collator

FIELDS.register(FieldSpec("pack.seq_id", "token", torch.long, -1, "student",
                          "index of the original sample inside the pack; -1 = pad"))
FIELDS.register(FieldSpec("pack.num_seqs", "scalar", torch.long, 0, None,
                          "number of original samples in this packed row"))

# Per-sample scalar fields have no meaning for a packed row. Future scalar fields that are
# useful for inspection can be mapped here to a metadata key without entering Batch data.
KEEP_AS_LIST: dict[str, str] = {}


def _as_list(value: Any) -> list:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def pack_samples(samples: list[dict], max_tokens: int) -> dict:
    """Concatenate unpadded student token fields into one row of at most ``max_tokens``.

    ``position_ids`` restart at zero for every original sequence. Target-aligned labels are
    concatenated unchanged, including the ignored first target of each sequence. Per-sample
    scalars are dropped; metadata is retained as lists. Source: DESIGN §4.2 and §11.5.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not samples:
        raise ValueError("cannot pack an empty sample list")
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples[1:]):
        raise ValueError("samples in a pack must have the same fields")
    if "student.input_ids" not in keys:
        raise KeyError("packing needs student.input_ids")
    if any(key.startswith("teacher.") for key in keys):
        raise ValueError("packing with separate teacher inputs not supported")
    lengths = [len(sample["student.input_ids"]) for sample in samples]
    if any(length <= 0 for length in lengths):
        raise ValueError("cannot pack an empty student sequence")
    if sum(lengths) > max_tokens:
        raise ValueError(f"packed sample has {sum(lengths)} tokens, exceeds max_tokens={max_tokens}")

    packed: dict[str, Any] = {}
    for key in samples[0]:
        if key in ("student.position_ids", "student.attention_mask", "pack.seq_id", "pack.num_seqs"):
            continue
        if key.startswith("meta."):
            packed[key] = [sample[key] for sample in samples]
            continue
        spec = FIELDS.resolve(key, [sample[key] for sample in samples], {"student": lengths})
        if spec.kind == "ragged":
            raise ValueError(f"packing does not support ragged field {key!r}")
        if spec.kind == "meta":
            packed[key] = [sample[key] for sample in samples]
        elif spec.kind in ("token", "token_k"):
            if spec.side != "student":
                raise ValueError("packing with separate teacher inputs not supported")
            values = [_as_list(sample[key]) for sample in samples]
            if any(len(value) != length for value, length in zip(values, lengths)):
                raise ValueError(f"{key}: token field length differs from student.input_ids")
            packed[key] = [item for value in values for item in value]
        elif key in KEEP_AS_LIST:
            packed[KEEP_AS_LIST[key]] = [sample[key] for sample in samples]
        # Scalars and vectors describe one original sequence, not the packed row.

    packed["student.position_ids"] = [position for length in lengths for position in range(length)]
    packed["student.attention_mask"] = [1] * sum(lengths)
    packed["pack.seq_id"] = [i for i, length in enumerate(lengths) for _ in range(length)]
    packed["pack.num_seqs"] = len(samples)
    return packed


class PackedCollator:
    """Pack each bin, then let the existing Collator pad and stack packed rows.

    Source: DESIGN §4.3 and §11.5. No changes to the generic collator are needed.
    """

    def __init__(self, collator: Collator, max_tokens: int):
        self.collator = collator
        self.max_tokens = max_tokens

    def __call__(self, bins: list[list[dict]]) -> Batch:
        return self.collator([pack_samples(samples, self.max_tokens) for samples in bins])


def ffd_bins(lengths: list[int], capacity: int) -> list[list[int]]:
    """First-fit-decreasing bins of sample indices, each with total length ≤ capacity.

    Stable sorting preserves the input order among equal-length samples. Source: T02 §5.3.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if any(length <= 0 for length in lengths):
        raise ValueError("sample lengths must be positive")
    if any(length > capacity for length in lengths):
        raise ValueError("sample length exceeds packing capacity")
    bins: list[list[int]] = []
    remaining: list[int] = []
    for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        length = lengths[i]
        for j, free in enumerate(remaining):
            if length <= free:
                bins[j].append(i)
                remaining[j] -= length
                break
        else:
            bins.append([i])
            remaining.append(capacity - length)
    return bins


class MultipackBatchSampler:
    """Yield fixed-size groups of FFD packs, one group per optimizer step.

    A rank receives every ``world_size``-th bin after the epoch shuffle. The final incomplete
    group is dropped, so every rank takes the same number of steps. Source: T02 §5.3.
    """

    def __init__(self, lengths, capacity, packs_per_step, seed, lookahead=10000,
                 rank=0, world_size=1, shuffle=True):
        self.lengths = [int(length) for length in lengths]
        self.capacity = int(capacity)
        self.packs_per_step = int(packs_per_step)
        self.seed = int(seed)
        self.lookahead = int(lookahead)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        if self.packs_per_step <= 0 or self.lookahead <= 0:
            raise ValueError("packs_per_step and lookahead must be positive")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid rank or world_size")
        if self.capacity <= 0 or any(length <= 0 or length > self.capacity for length in self.lengths):
            raise ValueError("sample lengths must be positive and no greater than capacity")
        self.epoch = 0
        self.start = 0
        self._epoch0_steps: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start(self, k: int) -> None:
        if k < 0:
            raise ValueError("start must be nonnegative")
        self.start = int(k)

    def _all_bins(self, epoch: int) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + epoch)
        indices = (torch.randperm(len(self.lengths), generator=generator).tolist()
                   if self.shuffle else list(range(len(self.lengths))))
        bins: list[list[int]] = []
        for offset in range(0, len(indices), self.lookahead):
            chunk = indices[offset:offset + self.lookahead]
            bins.extend([[chunk[i] for i in bin_] for bin_ in
                         ffd_bins([self.lengths[i] for i in chunk], self.capacity)])
        if self.shuffle and bins:
            order = torch.randperm(len(bins), generator=generator).tolist()
            bins = [bins[i] for i in order]
        return bins

    def __iter__(self) -> Iterator[list[list[int]]]:
        bins = self._all_bins(self.epoch)
        # Drop surplus bins before grouping, to keep equal work on all ranks.
        rank_bins = bins[self.rank::self.world_size][:len(bins) // self.world_size]
        steps = len(rank_bins) // self.packs_per_step
        for step in range(self.start, steps):
            start = step * self.packs_per_step
            yield rank_bins[start:start + self.packs_per_step]

    def __len__(self) -> int:
        if self._epoch0_steps is None:
            self._epoch0_steps = len(self._all_bins(0)) // (self.world_size * self.packs_per_step)
        return self._epoch0_steps


class LengthGroupedSampler(torch.utils.data.Sampler[int]):
    """EpochSampler-compatible length buckets that keep full local batches together.

    Shuffle samples, sort each mega-batch by length, then shuffle the resulting batches.
    Complete batches are sharded across ranks; their final tail is divided evenly so each
    rank exposes ``floor(n / world_size)`` indices. Source: T02 §5.7.
    """

    def __init__(self, lengths, batch_size: int, seed: int, length_bucket_mult: int = 50,
                 rank: int = 0, world_size: int = 1, shuffle: bool = True):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.length_bucket_mult = int(length_bucket_mult)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        if self.batch_size <= 0 or self.length_bucket_mult <= 0:
            raise ValueError("batch_size and length_bucket_mult must be positive")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid rank or world_size")
        if any(length <= 0 for length in self.lengths):
            raise ValueError("sample lengths must be positive")
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start(self, k: int) -> None:
        if k < 0:
            raise ValueError("start must be nonnegative")
        self.start = int(k)

    def per_rank(self) -> int:
        return len(self.lengths) // self.world_size

    def _indices(self) -> list[int]:
        n = len(self.lengths)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = (torch.randperm(n, generator=generator).tolist() if self.shuffle else list(range(n)))
        mega_size = self.batch_size * self.length_bucket_mult
        batches: list[list[int]] = []
        for offset in range(0, n, mega_size):
            mega = indices[offset:offset + mega_size]
            mega.sort(key=lambda i: -self.lengths[i])
            batches.extend(mega[j:j + self.batch_size] for j in range(0, len(mega), self.batch_size))
        if self.shuffle and batches:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]

        full = [batch for batch in batches if len(batch) == self.batch_size]
        partial = [i for batch in batches if len(batch) < self.batch_size for i in batch]
        full_rounds = len(full) // self.world_size
        chosen = [i for j in range(full_rounds) for i in full[j * self.world_size + self.rank]]
        tail = [i for batch in full[full_rounds * self.world_size:] for i in batch] + partial
        chosen.extend(tail[self.rank::self.world_size][:len(tail) // self.world_size])
        return chosen

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices()[self.start:])

    def __len__(self) -> int:
        return max(0, self.per_rank() - self.start)
