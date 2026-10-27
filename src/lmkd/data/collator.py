"""Generic collator driven by FieldSpec (DESIGN §4.3). No ``if key == ...`` branches here.

Sample format (output of transforms): ``dict[str, Any]`` with namespaced keys. Values may be
python lists, numpy arrays or tensors. Keys whose FieldSpec kind is ``meta`` (or any key under
the ``meta.`` namespace) go to ``batch.meta`` as lists.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lmkd.core.batch import FIELDS, Batch, FieldRegistry


def _to_tensor(v: Any, dtype: torch.dtype | None) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.to(dtype) if dtype is not None else v
    if isinstance(v, np.ndarray):
        t = torch.from_numpy(v)
        return t.to(dtype) if dtype is not None else t
    return torch.as_tensor(v, dtype=dtype)


class Collator:
    """Generic collator driven by FieldSpec: pads/stacks every key of a list of samples into a :class:`Batch` (no
    per-key branches).
    """
    def __init__(
        self,
        pad_token_ids: dict[str, int],
        *,
        fields: FieldRegistry = FIELDS,
        pad_to_multiple_of: int | None = 8,
        fixed_length: dict[str, int] | None = None,
    ):
        """
        Args:
            pad_token_ids: {"student": id, "teacher": id} used for ``<side>.input_ids``.
            pad_to_multiple_of: round padded length up (tensor cores).
            fixed_length: {"student": L} to pad to a fixed length (padding: max_length).
        """
        self.pad_token_ids = pad_token_ids
        self.fields = fields
        self.mult = pad_to_multiple_of
        self.fixed = fixed_length or {}

    def _target_len(self, lens: list[int], side: str) -> int:
        n = self.fixed.get(side, max(lens) if lens else 0)
        if self.mult:
            n = ((n + self.mult - 1) // self.mult) * self.mult
        return n

    def __call__(self, samples: list[dict[str, Any]]) -> Batch:
        if not samples:
            raise ValueError("empty sample list")
        keys = list(samples[0].keys())
        for s in samples[1:]:
            if set(s.keys()) != set(keys):
                raise KeyError(f"samples have different keys: {sorted(set(keys) ^ set(s.keys()))}")

        # sequence length per side = max length of <side>.input_ids
        side_len: dict[str, int] = {}
        lengths: dict[str, list[int]] = {}
        for side in ("student", "teacher"):
            k = f"{side}.input_ids"
            if k in samples[0]:
                lengths[side] = [len(s[k]) for s in samples]
                side_len[side] = self._target_len(lengths[side], side)

        data: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        for key in keys:
            vals = [s[key] for s in samples]
            if key.startswith("meta."):
                meta[key] = vals
                continue
            spec = self.fields.resolve(key, vals, lengths)   # unregistered key -> inferred (warning)
            if spec.kind == "meta":
                meta[key] = vals
                continue
            if spec.kind == "scalar":
                data[key] = _to_tensor(vals, spec.dtype)
            elif spec.kind == "vector":
                data[key] = torch.stack([_to_tensor(v, spec.dtype) for v in vals])
            elif spec.kind == "ragged":
                data[key] = [_to_tensor(v, spec.dtype) for v in vals]
            elif spec.kind in ("token", "token_k"):
                if spec.side not in side_len:
                    raise KeyError(f"{key} follows side '{spec.side}' but '{spec.side}.input_ids' is absent")
                T = side_len[spec.side]
                pad = spec.pad_value
                if key.endswith(".input_ids"):
                    pad = self.pad_token_ids[spec.side]
                ts = [_to_tensor(v, spec.dtype) for v in vals]
                if any(t.shape[0] > T for t in ts):
                    raise ValueError(f"{key}: sequence longer than padded length {T}")
                shape = (len(ts), T) + tuple(ts[0].shape[1:])
                out = torch.full(shape, pad, dtype=ts[0].dtype)
                for i, t in enumerate(ts):
                    out[i, : t.shape[0]] = t
                data[key] = out
            else:  # pragma: no cover
                raise ValueError(f"unknown field kind {spec.kind}")
        return Batch(data, meta)
