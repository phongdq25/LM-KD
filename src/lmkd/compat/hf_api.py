"""transformers glue (see ``lmkd.compat``)."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import torch

from lmkd.compat import has

# model_type values whose attention uses ``apply_rotary_pos_emb`` from their own modeling module.
ROTARY_MODEL_TYPES = ("llama", "qwen2", "qwen3", "mistral")


def rotary_apply_fn(attention_module: Any) -> Callable[..., tuple[Any, Any]]:
    """``apply_rotary_pos_emb(q, k, cos, sin)`` of the modeling file that defines this attention
    class (each architecture ships its own copy, e.g. ``transformers.models.qwen2.modeling_qwen2``)."""
    module = importlib.import_module(type(attention_module).__module__)
    return module.apply_rotary_pos_emb


def dtype_kwarg(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """``from_pretrained(path, **dtype_kwarg(dt))``: transformers renamed ``torch_dtype`` to ``dtype`` in 4.56."""
    return {"dtype": dtype} if has("transformers", "4.56") else {"torch_dtype": dtype}
