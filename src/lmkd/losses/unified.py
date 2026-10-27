"""Target-aligned, lazy loss inputs and semantic reductions for user-defined criteria.

Token-local callbacks may recompute linear-head chunks in backward (first-order,
 deterministic callbacks only). Sequence/scalar/dict objectives use full inputs.
"""
from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from pydantic import create_model

from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.simple import _same_origin
from lmkd.models.forward import FeatureRequest, SideOutput, SideRequest
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherSignalSpec


def _masked(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values
    if values.dim() == mask.dim() + 1 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.shape != mask.shape:
        raise ValueError(f"TokenLoss values {tuple(values.shape)} must match mask {tuple(mask.shape)}")
    return values * mask.to(values.device, values.dtype)


@dataclass(frozen=True)
class TokenLoss:
    """Per-target values [B,T], optional selection/weight mask and diagnostic values.

    Custom masks cannot enable unsupervised targets. The denominator remains the
    step's supervised-token count, not the number selected by a custom mask.
    """
    values: torch.Tensor
    mask: torch.Tensor | None = None
    log_values: torch.Tensor | None = None

    def effective_mask(self, default_mask: torch.Tensor | None) -> torch.Tensor | None:
        if self.mask is None:
            return default_mask
        if default_mask is None:
            return self.mask
        return self.mask * default_mask.ne(0).to(self.mask.device)

    def as_loss_term(self, default_mask: torch.Tensor | None = None) -> LossTerm:
        mask = self.effective_mask(default_mask)
        log_value = _masked(self.log_values, mask).sum() if self.log_values is not None else None
        return LossTerm(_masked(self.values, mask).sum(), "token_sum", log_value=log_value)


@dataclass(frozen=True)
class SeqLoss:
    """Per-sequence values [B], divided by the step's global sequence count."""
    values: torch.Tensor
    log_values: torch.Tensor | None = None

    def as_loss_term(self) -> LossTerm:
        if self.values.dim() != 1:
            raise ValueError("SeqLoss values must have shape [B]")
        log_value = self.log_values.sum() if self.log_values is not None else None
        return LossTerm(self.values.sum(), "seq_sum", log_value=log_value)


@dataclass(frozen=True)
class ScalarLoss:
    """A scalar micro-batch objective, averaged over micro-batches and ranks."""
    value: torch.Tensor
    log_value: torch.Tensor | None = None

    def as_loss_term(self) -> LossTerm:
        if self.value.dim() != 0:
            raise ValueError(f"ScalarLoss value must be scalar (0-dim), got shape {self.value.shape}")
        return LossTerm(self.value, "mean", log_value=self.log_value)


class Inputs:
    """Lazy target-aligned model features for a loss callback.

    ``logits`` materializes only on access. ``labels`` and ``mask`` follow [start,end).
    Hidden states retain HF positional convention; reading student hidden/side
    selects the full graph so both hidden and head gradients remain connected.
    Hidden capture uses HF indices (0=embeddings, 1..L=blocks).
    """

    def __init__(self, logits: torch.Tensor | None = None, hidden: tuple[torch.Tensor, ...] | None = None,
                 last_hidden: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                 batch: Batch | None = None, side: SideOutput | None = None,
                 logits_view: LogitsView | None = None, *, start: int = 0, end: int | None = None,
                 labels: torch.Tensor | None = None, chunked_student: bool = False):
        self._logits = logits
        self.hidden, self._last_hidden = hidden, last_hidden
        self.mask, self.batch, self._side = mask, batch, side
        self._chunked_student = chunked_student
        self.logits_view = logits_view
        self.start = start
        self.end = end if end is not None else (mask.shape[1] if mask is not None else 0)
        self.labels = labels

    @property
    def last_hidden(self) -> torch.Tensor | None:
        if self._chunked_student:
            raise _FullInputsRequired
        return self._last_hidden

    @property
    def side(self) -> SideOutput | None:
        if self._chunked_student:
            raise _FullInputsRequired
        return self._side

    @property
    def logits(self) -> torch.Tensor | None:
        if self._logits is None and self.logits_view is not None:
            # Explicit access to logits requests the full vocabulary; hidden-only callbacks do not.
            self._logits = self.logits_view.full()
        return self._logits

    def token_logp(self, labels: torch.Tensor) -> torch.Tensor:
        """Target log probabilities, zero on ignored labels, without full logits when possible."""
        if self._logits is None and self.logits_view is not None:
            return self.logits_view.target_logp(labels)
        logits = self.logits
        if logits is None:
            raise ValueError("Inputs has no logits; declare logits=True")
        if labels.shape != logits.shape[:2]:
            raise ValueError("labels must match the current chunk; use Inputs.labels")
        logp = logits.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return logp.masked_fill(labels.eq(-100), 0)


def _build_pydantic_config(fn: Callable, skip_args: int, name: str) -> type[BaseConfig]:
    try:
        hints = typing.get_type_hints(fn)
    except (NameError, TypeError):
        hints = {}
    fields: dict[str, Any] = {}
    for p in list(inspect.signature(fn).parameters.values())[skip_args:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(f"{fn.__name__}: declare each loss hyperparameter instead of *args/**kwargs")
        fields[p.name] = (hints.get(p.name, Any), ... if p.default is p.empty else p.default)
    return create_model(f"{name}Config", __base__=BaseConfig, **fields)


def _inputs(ctx: LossContext, side: str, start: int, end: int, *, logits: torch.Tensor | None = None,
            capture: tuple[int, ...] = (), use_logits: bool = True, chunk: bool = False) -> Inputs:
    output = ctx.student if side == "student" else (ctx.teacher.side if ctx.teacher else None)
    view = ctx.student.logits if side == "student" else (ctx.teacher.logits if ctx.teacher else None)
    mask = ctx.supervised_mask("student")[:, start:end]
    labels = ctx.batch.get(f"{side}.labels", ctx.batch["student.labels"])[:, start:end]
    hidden = tuple(output.hidden(i) for i in capture) if output is not None and capture else None
    last_hidden = output.last_hidden[:, start:end] if output is not None else None
    return Inputs(logits, hidden, last_hidden, mask, ctx.batch, output,
                  view if use_logits and not chunk else None, start=start, end=end, labels=labels,
                  chunked_student=chunk and side == "student")


def _as_output(res: Any, mask: torch.Tensor, name: str) -> LossOutput:
    if isinstance(res, LossOutput):
        return res
    values = res if isinstance(res, dict) else {"loss": res}
    terms = {}
    for key, value in values.items():
        if isinstance(value, TokenLoss):
            terms[key] = value.as_loss_term(mask)
        elif isinstance(value, (SeqLoss, ScalarLoss)):
            terms[key] = value.as_loss_term()
        elif isinstance(value, torch.Tensor):
            terms[key] = (ScalarLoss(value).as_loss_term() if value.dim() == 0
                          else TokenLoss(value).as_loss_term(mask))
        else:
            raise TypeError(f"[{name}] expected TokenLoss, SeqLoss, ScalarLoss or LossOutput; got {type(value)}")
    if not terms:
        raise ValueError(f"[{name}] loss returned no terms")
    return LossOutput(terms)


class _FullInputsRequired(Exception):
    """Internal signal: a callback returned a sequence/global objective."""


def _token_result(res: Any, mask: torch.Tensor) -> TokenLoss:
    if isinstance(res, torch.Tensor) and res.dim() > 0:
        res = TokenLoss(res)
    if not isinstance(res, TokenLoss):
        raise _FullInputsRequired
    # Validate callback shape without permitting accidental [B] -> [B,t] broadcasts.
    _masked(res.values, res.effective_mask(mask))
    return res


def _forward_chunked(ctx: LossContext, fn: Callable, kwargs: dict[str, Any], teacher: bool,
                     chunk_size: int) -> LossOutput:
    s_view = ctx.student.logits
    t_view = ctx.teacher.logits if ctx.teacher is not None else None
    if s_view is None or (teacher and t_view is None):
        raise ValueError("chunked loss needs online logits views for every requested side")
    if teacher and t_view is not None and t_view.shape_bt != s_view.shape_bt:
        raise ValueError("chunked KD requires aligned sequence shapes; use a cross-tokenizer Criterion")
    mask = ctx.supervised_mask("student")
    logs: dict[tuple[int, int], torch.Tensor] = {}

    def callback(logits: torch.Tensor, start: int, end: int) -> torch.Tensor:
        s = _inputs(ctx, "student", start, end, logits=logits, chunk=True)
        if teacher:
            assert t_view is not None
            with torch.no_grad():
                t_logits = t_view._target_slice(start, end)
            if t_logits.shape != logits.shape:
                raise ValueError("chunked KD requires matching vocabularies; use a cross-tokenizer Criterion")
            res = fn(s, _inputs(ctx, "teacher", start, end, logits=t_logits, chunk=True), **kwargs)
        else:
            res = fn(s, **kwargs)
        assert s.mask is not None
        result = _token_result(res, s.mask)
        selected = result.effective_mask(s.mask)
        if result.log_values is not None:
            logs[start, end] = _masked(result.log_values, selected).sum().detach()
        return _masked(result.values, selected)

    if s_view._efficient_ok():
        values = s_view._apply_chunked(callback, chunk_size, mask)
    else:
        parts = []
        for start in range(0, s_view.shape_bt[1], chunk_size):
            end = min(start + chunk_size, s_view.shape_bt[1])
            # Keep the graph even when a micro-batch has no supervised tokens.
            parts.append(callback(s_view._target_slice(start, end), start, end))
        values = torch.cat(parts, 1)
    log_value = torch.stack(list(logs.values())).sum() if logs else None
    return LossOutput({"loss": LossTerm(values.sum(), "token_sum", log_value=log_value)})


def _forward_full(ctx: LossContext, fn: Callable, kwargs: dict[str, Any], teacher: bool,
                  hidden_s: tuple[int, ...], hidden_t: tuple[int, ...], use_logits: bool, name: str) -> LossOutput:
    end = ctx.batch["student.labels"].shape[1]
    s = _inputs(ctx, "student", 0, end, capture=hidden_s, use_logits=use_logits)
    if teacher:
        if ctx.teacher is None:
            raise ValueError(f"[{name}] teacher output is missing")
        res = fn(s, _inputs(ctx, "teacher", 0, end, capture=hidden_t, use_logits=use_logits), **kwargs)
    else:
        res = fn(s, **kwargs)
    return _as_output(res, ctx.supervised_mask("student"), name)


def _capture_layers(capture: dict[str, list[str | int]], short: str, long: str) -> tuple[int, ...]:
    values = capture.get(short, capture.get(long, []))
    if any(isinstance(v, bool) or not str(v).lstrip("-").isdigit() for v in values):
        raise ValueError("capture must contain integer HF hidden indices (0=embedding, 1..L=blocks)")
    return tuple(int(v) for v in values)


def _build_criterion_class(fn, name, teacher, chunked, hidden_s, hidden_t, cfg_cls,
                           use_logits, chunk_size, supports_packing):
    class DecoratedCriterion(Criterion):
        """Criterion generated for a target-aligned user loss."""
        needs_teacher = teacher
        Config = cfg_cls

        def features(self) -> FeatureRequest:
            req = SideRequest(logits="view" if use_logits else "none", hidden=hidden_s or "last")
            t_req = SideRequest(logits="view" if use_logits else "none", hidden=hidden_t or "last") if teacher else None
            signal = TeacherSignalSpec(kind="full" if use_logits else "hidden") if teacher else None
            return FeatureRequest(student=req, teacher=t_req, teacher_signal=signal)

        def forward(self, ctx: LossContext) -> LossOutput:
            kwargs = self.cfg.model_dump()
            can_chunk = chunked is not False and use_logits and not hidden_s and not hidden_t
            if can_chunk and (chunked is True or bool(ctx.supervised_mask("student").any())):
                try:
                    return _forward_chunked(ctx, fn, kwargs, teacher, chunk_size)
                except _FullInputsRequired:
                    if chunked is True:
                        raise TypeError("chunked=True needs a token-local TokenLoss using logits; "
                                        "use chunked=False for hidden/sequence/global losses") from None
            return _forward_full(ctx, fn, kwargs, teacher, hidden_s, hidden_t, use_logits, name)

    DecoratedCriterion.supports_packing = supports_packing
    return DecoratedCriterion


def loss(name: str, *, teacher: bool = False, chunked: bool | None = None,
         capture: dict[str, list[str | int]] | None = None, logits: bool = True,
         chunk_size: int = 1024, supports_packing: bool = False, paper: str = "", source: str = "",
         status: str = "experimental") -> Callable[[Callable], Callable]:
    """Register a target-aligned loss; default auto-selects token chunks or full global inputs.

    Set chunked=False for sequence/batch interactions or higher-order gradients;
    chunked=True requires a deterministic, per-token callback. Logits remain lazy
    on the full path; logits=False also avoids requesting the head/teacher logits.
    Packing is opt-in because arbitrary callbacks may operate on whole sequences.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    capture = capture or {}
    if set(capture) - {"s", "student", "t", "teacher"}:
        raise ValueError("capture sides must be s/student or t/teacher")
    hidden_s, hidden_t = _capture_layers(capture, "s", "student"), _capture_layers(capture, "t", "teacher")
    if hidden_t and not teacher:
        raise ValueError("teacher capture requires teacher=True")

    def deco(fn: Callable) -> Callable:
        cls_name = "".join(p.capitalize() for p in name.split("_")) or "LossFn"
        cfg_cls = _build_pydantic_config(fn, 2 if teacher else 1, cls_name)
        cls = _build_criterion_class(fn, name, teacher, chunked, hidden_s, hidden_t, cfg_cls,
                                     logits, chunk_size, supports_packing)
        cls.__name__ = cls.__qualname__ = cls_name
        cls.__doc__ = fn.__doc__ or f"loss {fn.__module__}.{fn.__name__}"
        cls.__module__ = fn.__module__
        cls._lmkd_fn = fn
        CRITERIONS.register(name, paper=paper, source=source, status=status,
                            force=_same_origin(CRITERIONS, name, fn))(cls)
        fn.criterion_class = cls  # type: ignore[attr-defined]
        fn.registry_name = name  # type: ignore[attr-defined]
        return fn

    return deco
