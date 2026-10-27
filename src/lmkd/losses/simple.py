"""Easy loss APIs (tiers 0 and 1) on top of the Criterion contract (tier 2).

Tier 0 - ``compute_loss`` (like ``transformers.Trainer.compute_loss``): a function
``fn(batch, rt, **kwargs) -> Tensor | dict[str, Tensor] | LossOutput`` that may run the models
itself through ``rt.forward(batch)`` / ``rt.teacher_forward(batch)``. Use it from a recipe::

    imports: [my_loss.py]
    algorithm: {type: compute_loss, fn: my_kd, kwargs: {temperature: 2.0}}

or from a script: ``rt.train_step(loss_fn=my_kd)``.

Tier 1 - ``register_loss``: a function on HF-style model outputs, registered as a normal
criterion (composable in ``loss.terms`` with any other criterion, weights, schedules)::

    @register_loss("my_kd", teacher=True)
    def my_kd(s, t, batch, temperature: float = 2.0):
        ...   # s.logits / t.logits: [B,T,V] HF convention; batch["student.labels"] target-aligned

Tier 2 - subclass :class:`lmkd.losses.base.Criterion` (FeatureRequest + LogitsView chunked):
memory-efficient, the form expected for the official baselines.

Returned values with the default reduction ``mean`` are means over the micro-batch; the
aggregator averages them over micro-batches x ranks, so gradient accumulation is exact for
per-micro-batch means. Use ``reduction="token_sum"`` (and return a sum) for a token-exact mean.
"""
from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from pydantic import create_model

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS, LOSS_FNS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, Reduction
from lmkd.models.forward import FeatureRequest, ForwardRunner, SideOutput, SideRequest, side_inputs
from lmkd.teachers.signals import TeacherSignalSpec


@dataclass
class ModelView:
    """HF-style outputs of one model on one batch.

    ``logits[:, t]`` predicts ``input_ids[:, t+1]`` (HF convention, NOT target-aligned) - shift it
    yourself as in the original code. ``side`` gives the efficient target-aligned API
    (``side.logits.target_logp(labels)``, chunked) when you want to optimize later."""
    logits: torch.Tensor | None              # [B,T,V] fp32 (None when logits=False)
    hidden_states: tuple[torch.Tensor, ...] | None   # (embeddings, block 1..L) when hidden=True
    last_hidden_state: torch.Tensor           # [B,T,D]
    attention_mask: torch.Tensor | None
    side: SideOutput

    @classmethod
    def from_side(cls, so: SideOutput, *, hidden: bool, logits: bool) -> ModelView:
        hs = tuple(so.hidden(i) for i in range(so.num_layers + 1)) if hidden else None
        lg = so.logits.positional() if (logits and so.logits is not None) else None
        return cls(lg, hs, so.last_hidden, so.attention_mask, so)


def _side_request(*, hidden: bool, logits: bool) -> SideRequest:
    return SideRequest(logits="view" if logits else "none",
                       hidden="all" if hidden else ("none" if logits else "last"))


def run_side(rt: Any, batch: Any, side: str, *, hidden: bool, logits: bool, grad: bool) -> ModelView:
    """Forward one model on ``batch`` (student through the engine wrapper; teacher frozen)."""
    req = _side_request(hidden=hidden, logits=logits)
    if side == "student":
        outs = ForwardRunner.run(rt.student_model, rt.student_adapter, side_inputs(batch, "student"), req,
                                 grad=grad, module=rt.student_module)
    elif side == "teacher":
        if rt.teacher_model is None:
            raise ValueError("rt.teacher_forward: no teacher model (set models.teacher or teacher_model=)")
        src = "teacher" if "teacher.input_ids" in batch else "student"
        outs = ForwardRunner.run(rt.teacher_model, rt.teacher_adapter, side_inputs(batch, src), req, grad=False)
    else:
        raise ValueError(f"side must be 'student' or 'teacher', got {side!r}")
    return ModelView.from_side(outs[0], hidden=hidden, logits=logits)


def as_loss_output(value: Any, reduction: Reduction = "mean", name: str = "loss") -> LossOutput:
    """Tensor -> one term; dict[str, Tensor] -> one term per key; LossOutput -> unchanged."""
    if isinstance(value, LossOutput):
        return value
    if isinstance(value, torch.Tensor):
        value = {name: value}
    if not isinstance(value, dict) or not value:
        raise TypeError(f"loss function must return a Tensor, a dict of Tensors or a LossOutput, got {type(value)}")
    terms = {}
    for k, v in value.items():
        if not isinstance(v, torch.Tensor) or v.dim() != 0:
            raise ValueError(f"loss term {k!r} must be a scalar Tensor, got {getattr(v, 'shape', type(v))}")
        terms[k] = LossTerm(v, reduction)
    return LossOutput(terms=terms)


def _same_origin(registry: Any, name: str, fn: Callable) -> bool:
    """True when ``name`` is already registered from the same function (file imported twice via
    ``imports:`` and a plain import, or a re-run notebook cell): re-registering is then allowed."""
    if name not in registry:
        return False
    old = getattr(registry.get(name), "_lmkd_fn", registry.get(name))
    code_old, code_new = getattr(old, "__code__", None), fn.__code__
    return (code_old is not None and code_old.co_filename == code_new.co_filename
            and getattr(old, "__qualname__", None) == fn.__qualname__) or (
        getattr(old, "__module__", None) == fn.__module__ and getattr(old, "__qualname__", None) == fn.__qualname__)


def _params_config(fn: Callable, skip: int, cls_name: str) -> type[BaseConfig]:
    """Pydantic Config from the keyword parameters after the first ``skip`` positional ones."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - unresolved annotations -> Any
        hints = {}
    fields: dict[str, Any] = {}
    for p in list(inspect.signature(fn).parameters.values())[skip:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(f"{fn.__name__}: *args/**kwargs are not allowed; declare each hyperparameter")
        fields[p.name] = (hints.get(p.name, Any), ... if p.default is p.empty else p.default)
    return create_model(f"{cls_name}Config", __base__=BaseConfig, **fields)


# ============================================================================ tier 1
def register_loss(
    name: str,
    *,
    teacher: bool = False,
    logits: bool = True,
    hidden: bool = False,
    reduction: Reduction = "mean",
    paper: str = "",
    source: str = "",
    status: str = "experimental",
) -> Callable[[Callable], Callable]:
    """Register ``fn(student: ModelView, teacher: ModelView | None, batch, **hyperparams)`` as
    criterion ``name``. Hyperparameters = keyword parameters with defaults (strict Config,
    set them in ``loss.terms[].params``). ``teacher=True`` runs the configured teacher provider
    (``hf_frozen``) online; ``hidden=True`` exposes ``hidden_states`` of every layer.
    Full ``[B,T,V]`` logits are materialized: move to a tier-2 Criterion when memory matters."""

    def deco(fn: Callable) -> Callable:
        cls_name = "".join(p.capitalize() for p in name.split("_")) or "Fn"
        cfg_cls = _params_config(fn, 3, cls_name)
        req = _side_request(hidden=hidden, logits=logits)

        class FnCriterion(Criterion):
            """Criterion generated by ``register_loss`` around a user function."""
            needs_teacher = teacher
            supports_packing = False   # user code sees padded rows

            Config = cfg_cls  # type: ignore[assignment]  # built from fn's signature

            def features(self) -> FeatureRequest:
                if not teacher:
                    return FeatureRequest(student=req)
                return FeatureRequest(student=req, teacher=req,
                                      teacher_signal=TeacherSignalSpec(kind="hidden" if hidden else "full"))

            def forward(self, ctx: LossContext) -> LossOutput:
                s = ModelView.from_side(ctx.student, hidden=hidden, logits=logits)
                t = None
                if teacher:
                    if ctx.teacher is None or ctx.teacher.side is None:
                        raise ValueError(f"'{name}' needs an online teacher (teacher.provider: hf_frozen)")
                    t = ModelView.from_side(ctx.teacher.side, hidden=hidden, logits=logits)
                return as_loss_output(fn(s, t, ctx.batch, **self.cfg.model_dump()), reduction, self.alias)

        FnCriterion.__name__ = FnCriterion.__qualname__ = cls_name
        FnCriterion.__doc__ = fn.__doc__ or f"tier-1 loss function {fn.__module__}.{fn.__name__}"
        FnCriterion.__module__ = fn.__module__
        FnCriterion._lmkd_fn = fn  # type: ignore[attr-defined]
        CRITERIONS.register(name, paper=paper, source=source, status=status,
                            force=_same_origin(CRITERIONS, name, fn))(FnCriterion)
        fn.criterion_class = FnCriterion  # type: ignore[attr-defined]
        fn.registry_name = name  # type: ignore[attr-defined]
        return fn

    return deco


# ============================================================================ tier 0
def compute_loss(name: str, *, paper: str = "", source: str = "") -> Callable[[Callable], Callable]:
    """Register ``fn(batch, rt, **kwargs)`` so a recipe can select it with
    ``algorithm: {type: compute_loss, fn: <name>, kwargs: {...}}``."""

    def deco(fn: Callable) -> Callable:
        fn.__doc__ = fn.__doc__ or f"tier-0 loss {fn.__module__}.{fn.__name__}"
        LOSS_FNS.register(name, paper=paper, source=source, force=_same_origin(LOSS_FNS, name, fn))(fn)
        return fn

    return deco
