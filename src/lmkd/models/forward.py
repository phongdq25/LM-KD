"""FeatureRequest (criteria declare needs) + ForwardRunner (engine satisfies them once). DESIGN §5.2.

Selected hidden layers use temporary hooks; final attention rows are recomputed without
requesting full attention matrices. Forward always respects the engine wrapper.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Union

import torch
import torch.nn as nn

from lmkd.core.batch import Batch
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.logits_view import LogitsView

LayerSpec = Union[tuple[int, ...], Literal["all", "last", "none"]]
_LOGITS_ORDER = {"none": 0, "view": 1, "full": 2}


def _merge_layers(a: LayerSpec, b: LayerSpec) -> LayerSpec:
    if a == "all" or b == "all":
        return "all"
    if a == "none":
        return b
    if b == "none":
        return a
    sa = {-1} if a == "last" else set(a)
    sb = {-1} if b == "last" else set(b)
    return tuple(sorted(sa | sb))


@dataclass
class SideRequest:
    """What one model must compute: logits (none | view | full), hidden/attention layers, pooled vectors, number
    of stochastic views.
    """
    logits: Literal["none", "view", "full"] = "none"
    hidden: LayerSpec = "none"
    attentions: LayerSpec = "none"
    input_embeds: bool = False
    pooled: tuple[tuple[int, str], ...] = ()      # (layer, pooler name)
    n_views: int = 1                               # >1: several stochastic forwards (SimCSE)

    attention_last_row: LayerSpec = "none"

    def merge(self, o: SideRequest) -> SideRequest:
        return SideRequest(
            logits=max(self.logits, o.logits, key=_LOGITS_ORDER.__getitem__),
            hidden=_merge_layers(self.hidden, o.hidden),
            attentions=_merge_layers(self.attentions, o.attentions),
            input_embeds=self.input_embeds or o.input_embeds,
            pooled=tuple(sorted(set(self.pooled) | set(o.pooled))),
            n_views=max(self.n_views, o.n_views),
            attention_last_row=_merge_layers(self.attention_last_row, o.attention_last_row),
        )

    @property
    def needs_forward(self) -> bool:
        return (self.logits != "none" or self.hidden != "none" or self.attentions != "none"
                or self.input_embeds or bool(self.pooled) or self.attention_last_row != "none")


@dataclass
class FeatureRequest:
    """Union of what criteria need from student, teacher and teacher signal; ``merge`` combines requests so the
    forward runs once.
    """
    student: SideRequest = field(default_factory=SideRequest)
    teacher: SideRequest | None = None
    teacher_signal: Any | None = None   # lmkd.teachers.signals.TeacherSignalSpec

    def merge(self, o: FeatureRequest) -> FeatureRequest:
        t = self.teacher
        if o.teacher is not None:
            t = o.teacher if t is None else t.merge(o.teacher)
        sig = self.teacher_signal
        if o.teacher_signal is not None:
            sig = o.teacher_signal if sig is None else sig.merge(o.teacher_signal)
        return FeatureRequest(self.student.merge(o.student), t, sig)


class SideOutput:
    """Lazy, memoized access to one model's features for one batch."""

    def __init__(self, model: nn.Module, adapter: HFAutoAdapter, hf_out: Any, attention_mask: torch.Tensor | None,
                 input_embeds: torch.Tensor | None = None,
                 captured: dict[int, torch.Tensor] | None = None,
                 captured_attn_last: dict[int, torch.Tensor] | None = None):
        self._captured = captured or {}
        self._captured_attn_last = captured_attn_last or {}
        self.model = model
        self.adapter = adapter
        self._out = hf_out
        self.attention_mask = attention_mask
        self.input_embeds = input_embeds
        self._pooled: dict[tuple[int, str], torch.Tensor] = {}
        head = adapter.lm_head(model)
        self.logits: LogitsView | None = (
            LogitsView(hf_out.last_hidden_state, head) if head is not None else None
        )

    @property
    def last_hidden(self) -> torch.Tensor:
        return self._out.last_hidden_state

    @property
    def num_layers(self) -> int:
        return self.adapter.num_layers(self.model)

    def hidden(self, layer: int) -> torch.Tensor:
        """[B,T,D] hidden state of layer (0 = embeddings, L = last block, negatives allowed).
        NOTE: HF's last entry of hidden_states is already final-normed for most models."""
        if layer < 0:
            layer += self.num_layers + 1
        if layer in self._captured:
            return self._captured[layer]
        hs = self._out.hidden_states
        if hs is None:
            if layer in (-1, self.num_layers):
                return self.last_hidden
            raise KeyError(f"hidden({layer}) not captured: declare it in FeatureRequest.hidden")
        return hs[layer]

    def attention(self, layer: int) -> torch.Tensor:
        att = self._out.attentions
        if att is None:
            raise KeyError("attentions not captured: declare them in FeatureRequest.attentions")
        # HF attentions tuple is indexed by block (0..L-1); we use block numbering 1..L
        return att[layer - 1 if layer > 0 else layer]

    def attention_last_row(self, layer: int) -> torch.Tensor:
        if layer < 0:
            layer += self.num_layers + 1
        if layer not in self._captured_attn_last:
            raise KeyError(f"attention_last_row({layer}) not requested")
        return self._captured_attn_last[layer]

    def pooled(self, layer: int, pooler: str) -> torch.Tensor:
        key = (layer, pooler)
        if key not in self._pooled:
            from lmkd.core.registry import POOLERS

            self._pooled[key] = POOLERS.build(pooler)(self.hidden(layer), self.attention_mask)
        return self._pooled[key]

    def resolve(self, target: str) -> torch.Tensor:
        """Tensor addressed by a target string (used by TensorGradHook):
        ``hidden[i]`` | ``last_hidden`` | ``input_embeds`` | ``attention[i]``."""
        kind, idx = _parse_target_body(target)
        if kind == "hidden":
            return self.hidden(_need_index(kind, idx, target))
        if kind == "last_hidden":
            return self.last_hidden
        if kind == "input_embeds":
            if self.input_embeds is None:
                raise KeyError("input_embeds not captured: declare SideRequest(input_embeds=True)")
            return self.input_embeds
        if kind == "attention":
            return self.attention(_need_index(kind, idx, target))
        raise KeyError(f"unknown target {target!r}")


def _parse_target_body(body: str) -> tuple[str, int | None]:
    body = body.strip()
    if "[" in body:
        kind, rest = body.split("[", 1)
        return kind, int(rest.rstrip("]"))
    return body, None


def _need_index(kind: str, idx: int | None, target: str) -> int:
    if idx is None:
        raise ValueError(f"target {target!r}: '{kind}' needs a layer index, e.g. '{kind}[-1]'")
    return idx


def parse_target(target: str) -> tuple[str, str, SideRequest]:
    """``"student.hidden[-8]"`` -> ("student", "hidden[-8]", SideRequest needed to capture it)."""
    if "." not in target:
        raise ValueError(f"target must look like '<side>.<tensor>', got {target!r}")
    side, body = target.split(".", 1)
    if side not in ("student", "teacher"):
        raise ValueError(f"unknown side {side!r} in target {target!r}")
    kind, idx = _parse_target_body(body)
    if kind == "hidden":
        req = SideRequest(hidden=(_need_index(kind, idx, target),))
    elif kind == "last_hidden":
        req = SideRequest(hidden=(-1,))
    elif kind == "input_embeds":
        req = SideRequest(input_embeds=True, logits="view")
    elif kind == "attention":
        req = SideRequest(attentions=(_need_index(kind, idx, target),))
    else:
        raise ValueError(f"unknown tensor kind {kind!r} in target {target!r}")
    return side, body, req


def _layer_indices(spec: LayerSpec, num_layers: int, *, lowest: int, what: str) -> list[int]:
    """Absolute block indices (lowest..num_layers) selected by a LayerSpec (negatives allowed)."""
    if spec == "all":
        return list(range(lowest, num_layers + 1))
    if spec == "none":
        return []
    out = []
    for index in set((-1,) if spec == "last" else spec):
        i = index + num_layers + 1 if index < 0 else index
        if not lowest <= i <= num_layers:
            raise IndexError(f"{what} layer {index} outside {lowest}..{num_layers}")
        out.append(i)
    return out


class _FeatureCapture:
    """Temporary forward hooks that collect selected hidden states / last-row attentions during
    ONE forward pass. Use as a context manager; ``hidden`` / ``attn_last`` are filled by the pass."""

    def __init__(self, model: nn.Module, adapter: HFAutoAdapter, inputs: dict[str, torch.Tensor],
                 req: SideRequest, *, need_all_hidden: bool):
        self.model, self.adapter, self.inputs = model, adapter, inputs
        self.layers = adapter.layers(model)
        self.num_layers = len(self.layers)
        # HF already returns every layer when output_hidden_states=True: no hooks needed then.
        self.hidden_spec: LayerSpec = "none" if need_all_hidden else req.hidden
        self.attn_spec: LayerSpec = req.attention_last_row
        self.hidden: dict[int, torch.Tensor] = {}
        self.attn_last: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    # -- hooks (one factory per layer so each closure binds its own index)
    def _pre_hidden(self, i: int) -> Callable[..., None]:
        def hook(block, args, kwargs):
            if i not in self.hidden:
                self.hidden[i] = kwargs.get("hidden_states", args[0] if args else None)
        return hook

    def _post_hidden(self, i: int) -> Callable[..., None]:
        def hook(block, args, output):
            if i not in self.hidden:
                self.hidden[i] = output[0] if isinstance(output, tuple) else output
        return hook

    def _pre_attention(self, i: int) -> Callable[..., None]:
        def hook(block, args, kwargs):
            if i not in self.attn_last:
                h = kwargs.get("hidden_states", args[0] if args else None)
                self.attn_last[i] = self.adapter.last_row_attention(
                    self.model, i, h, self.inputs.get("attention_mask"), self.inputs.get("position_ids")).detach()
        return hook

    def __enter__(self) -> _FeatureCapture:
        L = self.num_layers
        for i in _layer_indices(self.hidden_spec, L, lowest=0, what="hidden"):
            if i == 0:   # embeddings: input of the first block
                self._handles.append(self.layers[0].register_forward_pre_hook(self._pre_hidden(i), with_kwargs=True))
            elif i != L:  # last block's output is ``last_hidden_state`` (already final-normed)
                self._handles.append(self.layers[i - 1].register_forward_hook(self._post_hidden(i)))
        for i in _layer_indices(self.attn_spec, L, lowest=1, what="attention"):
            self._handles.append(self.layers[i - 1].register_forward_pre_hook(self._pre_attention(i), with_kwargs=True))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class ForwardRunner:
    """Runs a model once per view with exactly the features requested."""

    @staticmethod
    def run(
        model: nn.Module, adapter: HFAutoAdapter, inputs: dict[str, torch.Tensor], req: SideRequest,
        *, grad: bool, module: nn.Module | None = None,
    ) -> list[SideOutput]:
        """``module`` is the student forward wrapper; always call it when supplied (K1).

        Local uses HiddenForward. T05's DeepSpeed wrapper forwards through engine -> bundle
        -> student; never unwrap it here. The raw model is only used to locate features.
        """
        if not req.needs_forward:
            return []
        need_all_hidden = req.hidden == "all" or bool(req.pooled)
        outs = []
        with torch.enable_grad() if grad else torch.no_grad():
            for _ in range(req.n_views):
                embeds = adapter.embed_tokens(model)(inputs["input_ids"]) if req.input_embeds else None
                kw: dict[str, Any] = {"attention_mask": inputs.get("attention_mask"), "position_ids": inputs.get("position_ids"),
                      "output_hidden_states": need_all_hidden, "output_attentions": req.attentions != "none",
                      "inputs_embeds": embeds}
                with _FeatureCapture(model, adapter, inputs, req, need_all_hidden=need_all_hidden) as cap:
                    if module is not None:
                        hf_out = module(input_ids=inputs["input_ids"], **kw)
                    else:
                        hf_out = adapter.forward_hidden(model, inputs["input_ids"], **kw)
                outs.append(SideOutput(model, adapter, hf_out, inputs.get("attention_mask"), embeds,
                                       cap.hidden, cap.attn_last))
        return outs


def side_inputs(batch: Batch, side: str) -> dict[str, torch.Tensor]:
    d = batch.ns(side)
    return {k: d[k] for k in ("input_ids", "attention_mask", "position_ids")
            if k in d and (k != "attention_mask" or "pack.seq_id" not in batch)}


def without_logits(req: SideRequest) -> SideRequest:
    return replace(req, logits="none")
