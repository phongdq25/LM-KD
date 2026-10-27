"""Pure schedule functions and parameter schedule evaluators (DESIGN §6.3, §8.1).

Supported schedule types for ``schedule_value``:
| type               | parameters                        | formula / behavior                                                |
|--------------------|-----------------------------------|-------------------------------------------------------------------|
| constant           | value                             | value                                                             |
| linear             | start, end, begin=0.0, until=1.0  | linear interpolation on progress in [begin, until], clamped       |
| linear_warmup      | start=0.0, end, until             | = linear(begin=0.0)                                               |
| warmup_steps       | start=0.0, end, steps             | linear interpolation on step in [0, steps], clamped               |
| cosine             | start, end, begin=0.0, until=1.0  | end + (start - end) * 0.5 * (1 + cos(pi * u)) on progress        |
| step               | values, boundaries                | piecewise constant by progress                                    |
| follow_lr_warmup   | end                               | end * min(1, step / warmup_steps) from first optim scheduler      |
"""
from __future__ import annotations

import math
from typing import Any


def polynomial(
    s: int,
    total: int,
    warmup: int = 0,
    power: float = 0.5,
    min_lr_ratio: float = 0.0,
) -> float:
    """Polynomial decay with warmup matching HF get_polynomial_decay_schedule_with_warmup."""
    if warmup > 0 and s < warmup:
        return float(s) / float(max(1, warmup))
    if s >= total:
        return min_lr_ratio
    p = (s - warmup) / max(1, total - warmup)
    p = min(1.0, max(0.0, p))
    return min_lr_ratio + (1.0 - min_lr_ratio) * ((1.0 - p) ** power)


def cosine_annealing(s: int, total: int, min_lr_ratio: float = 0.0) -> float:
    """Cosine annealing without warmup (CosineAnnealingLR relative to group lr)."""
    p = min(1.0, max(0.0, s / max(1, total)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))


def wsd(
    s: int,
    total: int,
    warmup: int = 0,
    decay: int = 0,
    min_lr_ratio: float = 0.0,
    decay_type: str = "linear",
) -> float:
    """Warmup (linear) - Stable - Decay (linear | cosine | 1-sqrt) factor in [min_lr_ratio, 1].

    Same shape as ``transformers.get_wsd_schedule`` (checked by tests/unit/test_compat_canary.py);
    written here because transformers only exposes it through a private lambda."""
    stable_end = max(0, total - decay)
    if s < warmup:
        return min_lr_ratio + (1.0 - min_lr_ratio) * s / max(1, warmup)
    if s < stable_end:
        return 1.0
    if s < stable_end + decay:
        p = (s - stable_end) / max(1, decay)
        if decay_type == "linear":
            f = 1.0 - p
        elif decay_type == "cosine":
            f = 0.5 * (1.0 + math.cos(math.pi * p))
        elif decay_type == "1-sqrt":
            f = 1.0 - math.sqrt(p)
        else:
            raise ValueError(f"unknown decay_type {decay_type!r} (linear | cosine | 1-sqrt)")
        return min_lr_ratio + (1.0 - min_lr_ratio) * f
    return min_lr_ratio


def _f(spec: dict, key: str, default: float | None = None) -> float:
    return float(spec[key]) if default is None else float(spec.get(key, default))


def _constant(spec: dict, step: int, total: int, progress: float, rt: Any) -> float:
    return float(spec["value"])


def _linear(spec: dict, step: int, total: int, progress: float, rt: Any) -> float:
    """start -> end between progress ``begin`` and ``until``."""
    start, end, begin, until = _f(spec, "start", 0.0), _f(spec, "end"), _f(spec, "begin", 0.0), _f(spec, "until", 1.0)
    if progress <= begin:
        return start
    if progress >= until:
        return end
    return start + (progress - begin) / max(1e-12, until - begin) * (end - start)


def _linear_warmup(spec: dict, step: int, total: int, progress: float, rt: Any) -> float:
    """start -> end between progress 0 and ``until``."""
    start, end, until = _f(spec, "start", 0.0), _f(spec, "end"), _f(spec, "until", 1.0)
    if progress <= 0.0:
        return start
    if progress >= until:
        return end
    return start + progress / max(1e-12, until) * (end - start)


def _warmup_steps(spec: dict, step: int, total: int, progress: float, rt: Any) -> float:
    """start -> end over an absolute number of ``steps``."""
    start, end, steps = _f(spec, "start", 0.0), _f(spec, "end"), int(spec["steps"])
    if step <= 0:
        return start
    if step >= steps:
        return end
    return start + (step / max(1, steps)) * (end - start)


def _cosine(spec: dict, step: int, total: int, progress: float, rt: Any) -> float:
    start, end, begin, until = _f(spec, "start", 0.0), _f(spec, "end"), _f(spec, "begin", 0.0), _f(spec, "until", 1.0)
    if progress <= begin:
        return start
    if progress >= until:
        return end
    u = (progress - begin) / max(1e-12, until - begin)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * u))


def _piecewise(spec: dict, step: int, total: int, progress: float, rt: Any) -> float:
    values = spec["values"]
    for boundary, value in zip(spec.get("boundaries", []), values):
        if progress < boundary:
            return float(value)
    return float(values[-1])


def _follow_lr_warmup(spec: dict, step: int, total: int, progress: float, rt: Any) -> float:
    """0 -> end over the same warmup as the first LR scheduler of the recipe."""
    warmup_steps = None
    if rt is not None and getattr(rt, "config", None) is not None:
        scheds = getattr(rt.config.optim, "schedulers", [])
        if scheds:
            s0 = scheds[0]
            warmup_steps = s0.get("warmup_steps") if isinstance(s0, dict) else getattr(s0, "warmup_steps", None)
            if warmup_steps is None:
                ratio = s0.get("warmup_ratio", 0.0) if isinstance(s0, dict) else getattr(s0, "warmup_ratio", 0.0)
                warmup_steps = int(ratio * total)
    if warmup_steps is None:
        warmup_steps = int(spec.get("warmup_steps", 0))
    p = 1.0 if warmup_steps <= 0 else min(1.0, step / warmup_steps)
    return _f(spec, "end") * p


_SCHEDULE_TYPES = {
    "constant": _constant, "linear": _linear, "linear_warmup": _linear_warmup, "warmup_steps": _warmup_steps,
    "cosine": _cosine, "step": _piecewise, "follow_lr_warmup": _follow_lr_warmup,
}


def schedule_value(spec: dict, step: int, total: int, rt: Any = None) -> float:
    """Value of a parameter schedule ``spec`` (``{"type": ..., ...}``) at ``step`` of ``total``.
    Types: constant, linear, linear_warmup, warmup_steps, cosine, step, follow_lr_warmup."""
    typ = spec.get("type", "constant")
    fn = _SCHEDULE_TYPES.get(typ)
    if fn is None:
        raise ValueError(f"unknown schedule type '{typ}' (available: {sorted(_SCHEDULE_TYPES)})")
    return fn(spec, step, total, step / max(1, total), rt)
