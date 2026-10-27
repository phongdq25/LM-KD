"""DeepSpeed glue (see ``lmkd.compat``): ZeRO-3 parameters, engine step bookkeeping, gathering."""
from __future__ import annotations

from typing import Any

import torch

# ZeRO-3 marks every partitioned parameter with this attribute and keeps it while temporarily gathered.
PARTITIONED_ATTR = "ds_id"


def is_partitioned_param(p: Any) -> bool:
    """True for a ZeRO-3 parameter (sharded, or gathered at the moment)."""
    return hasattr(p, PARTITIONED_ATTR)


def initialize(**kwargs: Any) -> Any:
    """``deepspeed.initialize`` returning only the engine."""
    import deepspeed

    engine, _, _, _ = deepspeed.initialize(**kwargs)
    return engine


def gathered_parameters(params: list[torch.nn.Parameter], modifier_rank: int | None = None) -> Any:
    """Context manager that materializes the full ZeRO-3 parameters (collective)."""
    import deepspeed

    return deepspeed.zero.GatheredParameters(params, modifier_rank=modifier_rank)


def set_accumulation_boundary(ds_engine: Any, is_boundary: bool) -> None:
    """Tell the engine whether the next backward ends a gradient-accumulation window
    (``False`` = skip the gradient all-reduce / optimizer work for this micro-batch)."""
    ds_engine.set_gradient_accumulation_boundary(is_boundary)


def skipped_steps(ds_engine: Any) -> int:
    return int(getattr(ds_engine, "skipped_steps", 0))


def step_was_applied(ds_engine: Any, skipped_before: int) -> bool:
    """After ``ds_engine.step()``: did the optimizer really step (False on an fp16 overflow skip)?"""
    if hasattr(ds_engine, "was_step_applied"):
        return bool(ds_engine.was_step_applied())
    return skipped_steps(ds_engine) == skipped_before
