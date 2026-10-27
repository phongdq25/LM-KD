"""The universal batch contract (DESIGN §4).

* :class:`Batch` is a flat mapping with namespaced keys ``"<ns>.<name>"`` (e.g.
  ``"student.input_ids"``, ``"seq.token_weights"``) plus a ``meta`` dict for non-tensor data.
* Every key that a collator must pad/stack is described by a :class:`FieldSpec` registered in
  :data:`FIELDS`. Plugins that introduce new keys should register their FieldSpec at import
  time; an unregistered key is inferred from its values (:func:`infer_field_spec`, with a warning).

TARGET-ALIGNED CONVENTION (DESIGN §4.2, mandatory):
  every per-token tensor has length T == len(input_ids) of its ``side`` and position ``t``
  describes the *target token* ``input_ids[t]``. Position 0 is never supervised.
  Only :class:`lmkd.models.logits_view.LogitsView` shifts logits; nobody else does ``[:, :-1]``.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

FieldKind = Literal["token", "token_k", "scalar", "vector", "ragged", "meta"]


@dataclass(frozen=True)
class FieldSpec:
    """How the collator pads/stacks one batch key: ``kind`` (token | token_k | scalar | vector | ragged | meta),
    dtype, pad value and, for token fields, which side's length it follows.
    """
    key: str
    kind: FieldKind
    dtype: torch.dtype | None = None
    pad_value: float | int = 0
    side: Literal["student", "teacher"] | None = None  # which sequence length a token field follows
    doc: str = ""

    def __post_init__(self) -> None:
        if self.kind in ("token", "token_k") and self.side is None:
            raise ValueError(f"token field {self.key!r} must declare side='student'|'teacher'")


class FieldRegistry:
    """Registry of :class:`FieldSpec` by batch key (``FIELDS``). Unknown keys can be inferred from values with
    ``resolve``.
    """
    def __init__(self) -> None:
        self._specs: dict[str, FieldSpec] = {}

    def register(self, spec: FieldSpec, *, force: bool = False) -> FieldSpec:
        old = self._specs.get(spec.key)
        if old is not None and old != spec and not force:
            raise KeyError(f"field {spec.key!r} already registered with a different spec: {old}")
        self._specs[spec.key] = spec
        return spec

    def get(self, key: str) -> FieldSpec:
        if key not in self._specs:
            raise KeyError(
                f"no FieldSpec registered for batch key {key!r}. Register it with "
                f"FIELDS.register(FieldSpec(...)) in the plugin module that produces it."
            )
        return self._specs[key]

    def resolve(self, key: str, values: list[Any], lengths: dict[str, list[int]]) -> FieldSpec:
        """Registered spec, else infer one from the values (and register it, warning once).

        ``lengths`` = per-sample ``len(<side>.input_ids)`` for each side present. Explicit
        registration wins; register when the default pad value (0) or the guess is wrong."""
        if key in self._specs:
            return self._specs[key]
        spec = infer_field_spec(key, values, lengths)
        warnings.warn(f"batch key {key!r} has no FieldSpec; inferred {spec}. Register it explicitly with "
                      f"FIELDS.register(FieldSpec(...)) if the kind, side or pad value is wrong.", stacklevel=2)
        return self.register(spec)

    def __contains__(self, key: str) -> bool:
        return key in self._specs

    def keys(self) -> list[str]:
        return sorted(self._specs)


def _infer_dtype(t: torch.Tensor) -> torch.dtype:
    if t.dtype == torch.bool:
        return torch.bool
    return torch.long if not t.is_floating_point() else torch.float32


def infer_field_spec(key: str, values: list[Any], lengths: dict[str, list[int]]) -> FieldSpec:
    """Guess a FieldSpec from one batch worth of values.

    meta.* / str / dict / None -> meta; python or 0-d numbers -> scalar; arrays whose first dim
    equals ``len(<side>.input_ids)`` for every sample -> token (1-D) / token_k (>1-D) on that side
    (student first); equal shapes -> vector; otherwise ragged."""
    doc = "inferred"
    if key.startswith("meta.") or any(v is None or isinstance(v, (str, bytes, dict)) for v in values):
        return FieldSpec(key, "meta", None, 0, None, doc)
    try:
        ts = [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in values]
    except (TypeError, ValueError, RuntimeError):
        return FieldSpec(key, "meta", None, 0, None, doc)
    dtype = _infer_dtype(ts[0])
    if all(t.dim() == 0 for t in ts):
        return FieldSpec(key, "scalar", dtype, 0, None, doc)
    if all(t.dim() >= 1 for t in ts):
        for side in ("student", "teacher"):
            lens = lengths.get(side)
            if lens and len(lens) == len(ts) and all(t.shape[0] == n for t, n in zip(ts, lens)):
                kind: FieldKind = "token" if ts[0].dim() == 1 else "token_k"
                return FieldSpec(key, kind, dtype, 0, side, doc)
        if all(t.shape == ts[0].shape for t in ts):
            return FieldSpec(key, "vector", dtype, 0, None, doc)
    return FieldSpec(key, "ragged", dtype, 0, None, doc)


FIELDS = FieldRegistry()

_L = torch.long
_F = torch.float32
_B = torch.bool


def _reg(key: str, kind: FieldKind, dtype: torch.dtype | None = None, pad: float = 0,
         side: Literal["student", "teacher"] | None = None, doc: str = "") -> None:
    FIELDS.register(FieldSpec(key, kind, dtype, pad, side, doc))


# --- core fields (other namespaces are registered by the plugins that produce them) ---------
for _side in ("student", "teacher"):
    _reg(f"{_side}.input_ids", "token", _L, 0, _side, "token ids (pad value replaced by collator pad id)")
    _reg(f"{_side}.attention_mask", "token", _L, 0, _side)
    _reg(f"{_side}.position_ids", "token", _L, 0, _side)
    _reg(f"{_side}.labels", "token", _L, -100, _side, "target-aligned; -100 = ignore")
    _reg(f"{_side}.prompt_len", "scalar", _L, 0, None, "number of prompt tokens")
_reg("seq.loss_mask", "token", _F, 0.0, "student", "1 = supervised target token")
_reg("seq.token_weights", "token", _F, 1.0, "student", "per-target weight (IWC, region...)")
_reg("seq.token_source", "token", _L, 0, "student", "0 prompt 1 prefix 2 cont 3 junction")
_reg("seq.segment_id", "token", _L, -1, "student", "CoT step index per token, -1 = none")
_reg("group.uid", "meta", None, 0, None, "group id (listwise / RL groups)")
_reg("group.index", "scalar", _L, 0, None, "index within group")
_reg("meta.id", "meta", None, 0, None, "stable record id")
_reg("meta.raw", "meta", None, 0, None, "raw record (debug)")
# Common benchmark and prompt dataset text fields
for _meta_text in ("problem", "question", "solution", "answer", "input"):
    _reg(_meta_text, "meta", None, 0, None, f"raw {_meta_text} text")


class Batch(MutableMapping):
    """Flat namespaced mapping. Tensors live in the mapping, non-tensor data in ``meta``."""

    def __init__(self, data: dict[str, Any] | None = None, meta: dict[str, Any] | None = None):
        self._d: dict[str, Any] = dict(data or {})
        self.meta: dict[str, Any] = dict(meta or {})

    # MutableMapping
    def __getitem__(self, k: str) -> Any:
        if k not in self._d:
            raise KeyError(
                f"batch key {k!r} missing. Present: {sorted(self._d)}. "
                f"Some plugin must 'provide' it (see `lmkd check`)."
            )
        return self._d[k]

    def __setitem__(self, k: str, v: Any) -> None:
        if "." not in k:
            raise KeyError(f"batch keys must be namespaced '<ns>.<name>', got {k!r}")
        self._d[k] = v

    def __delitem__(self, k: str) -> None:
        del self._d[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        parts = []
        for k, v in sorted(self._d.items()):
            parts.append(f"{k}={tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__}")
        return f"Batch({', '.join(parts)}; meta={sorted(self.meta)})"

    # helpers
    def require(self, *keys: str) -> None:
        missing = [k for k in keys if k not in self._d]
        if missing:
            raise KeyError(f"batch is missing required keys {missing}; present: {sorted(self._d)}")

    def ns(self, prefix: str) -> dict[str, Any]:
        """``batch.ns("student")`` -> ``{"input_ids": ..., "attention_mask": ...}``"""
        p = prefix + "."
        return {k[len(p):]: v for k, v in self._d.items() if k.startswith(p)}

    @property
    def batch_size(self) -> int:
        for v in self._d.values():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                return v.shape[0]
        for v in self.meta.values():
            if isinstance(v, list):
                return len(v)
        return 0

    def to(self, device: torch.device | str, non_blocking: bool = True) -> Batch:
        out: dict[str, Any] = {}
        for k, v in self._d.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=non_blocking)
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                out[k] = [t.to(device, non_blocking=non_blocking) for t in v]
            else:
                out[k] = v
        return Batch(out, self.meta)

    def select(self, idx: torch.Tensor | list[int]) -> Batch:
        idx_list = idx.tolist() if isinstance(idx, torch.Tensor) else list(idx)
        idx_t = torch.as_tensor(idx_list, dtype=torch.long)
        out: dict[str, Any] = {}
        for k, v in self._d.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                out[k] = v.index_select(0, idx_t.to(v.device))
            elif isinstance(v, list):
                out[k] = [v[i] for i in idx_list]
            else:
                out[k] = v
        meta = {k: ([v[i] for i in idx_list] if isinstance(v, list) else v) for k, v in self.meta.items()}
        return Batch(out, meta)

    def split(self, n_micro: int) -> list[Batch]:
        """Split along dim 0 into ``n_micro`` contiguous chunks.

        Listwise/RL groups (K13): rows of one group must be contiguous and carry ``group.index``
        (0..N-1). A split boundary that would cut a group raises (choose micro_batch_size as a
        multiple of the group size)."""
        bs = self.batch_size
        if n_micro <= 1 or bs == 0:
            return [self]
        if bs % n_micro:
            raise ValueError(f"batch size {bs} not divisible by n_micro={n_micro}")
        step = bs // n_micro
        if "group.index" in self._d:
            gi = self._d["group.index"]
            for start in range(step, bs, step):
                if int(gi[start]) != 0:
                    raise ValueError(
                        f"micro-batch boundary at row {start} cuts a group (group.index={int(gi[start])}); "
                        "make train.micro_batch_size a multiple of the group size"
                    )
        return [self.select(list(range(i, i + step))) for i in range(0, bs, step)]

    def copy(self) -> Batch:
        return Batch(dict(self._d), dict(self.meta))
