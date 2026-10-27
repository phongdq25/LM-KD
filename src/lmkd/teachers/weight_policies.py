"""WeightPolicy: how a self-teacher's weights evolve (DESIGN §5.5)."""
from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from pydantic import Field

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import WEIGHT_POLICIES
from lmkd.models.adapters import unwrap


class WeightPolicy(Plugin):
    """How a self-teacher's weights evolve relative to the student."""

    def setup(self, rt: Any) -> None:
        self.rt = rt

    def context(self, rt: Any) -> contextlib.AbstractContextManager[nn.Module]:
        """Context manager under which the student model acts as the teacher."""
        raise NotImplementedError

    def on_optimizer_step(self, rt: Any) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


@WEIGHT_POLICIES.register("live")
class Live(WeightPolicy):
    """Teacher = the current student (live weights, 0 memory overhead)."""

    class Config(BaseConfig):
        pass

    @contextlib.contextmanager
    def context(self, rt: Any) -> Iterator[nn.Module]:
        yield rt.student_model


@WEIGHT_POLICIES.register("lora_disabled")
class LoraDisabled(WeightPolicy):
    """Teacher = the student with LoRA adapters switched off, i.e. the frozen base."""

    class Config(BaseConfig):
        pass

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        m = unwrap(rt.student_model)
        if not hasattr(m, "disable_adapter"):
            raise ValueError("lora_disabled weight policy requires a PEFT model with disable_adapter")

    @contextlib.contextmanager
    def context(self, rt: Any) -> Iterator[nn.Module]:
        m = rt.student_model
        with rt.student_adapter.adapters_disabled(m):
            yield m


@WEIGHT_POLICIES.register("ema", source="Self-Distill/OPSD-main/opsd_trainer.py:481-624")
class EMA(WeightPolicy):
    """Teacher = exponential moving average of the trainable student parameters.

    Update rule on optimizer step:
      * First step (lazy init): ema[n] = p.data.clone() (no decay update).
      * Subsequent steps: ema[n].mul_(1 - rate).add_(p.data, alpha=rate).

    Rate conversion table:
      * OPSD: ema_decay = 0.999 -> rate = 1 - ema_decay = 0.001
      * SDPO: teacher_update_rate = 0.05 -> rate = 0.05

    Autograd safety:
      Teacher forward runs after the student forward pass has created its computation graph.
      Assigning p.data temporarily modifies parameter storage. To preserve autograd correctness,
      the student weights are strictly restored in the contextmanager's finally block before backward.
    """

    class Config(BaseConfig):
        rate: float = 0.05

    def __init__(self, cfg: Any = None) -> None:
        super().__init__(cfg)
        self.ema: dict[str, torch.Tensor] = {}
        self._cached_head: nn.Module | None = None

    def get_head(self, rt: Any) -> nn.Module | None:
        head = rt.student_adapter.lm_head(rt.student_model)
        if head is None or not any(p.requires_grad for p in head.parameters()):
            return head
        if self._cached_head is None:
            self._cached_head = copy.deepcopy(head).eval().requires_grad_(False)
            self._sync_cached_head(rt)
        return self._cached_head

    def _sync_cached_head(self, rt: Any) -> None:
        if self._cached_head is not None:
            m = unwrap(rt.student_model)
            id_to_name = {id(p): n for n, p in m.named_parameters()}
            head = rt.student_adapter.lm_head(rt.student_model)
            for p_live, p_cached in zip(head.parameters(), self._cached_head.parameters()):
                name = id_to_name.get(id(p_live))
                if name is not None and name in self.ema:
                    p_cached.data.copy_(self.ema[name].data)

    @contextlib.contextmanager
    def context(self, rt: Any) -> Iterator[nn.Module]:
        m = rt.student_model
        if not self.ema:
            yield m
            return

        unwrapped = unwrap(m)
        saved: dict[str, torch.Tensor] = {}
        for name, p in unwrapped.named_parameters():
            if p.requires_grad and name in self.ema:
                ema_t = self.ema[name]
                if ema_t.device != p.data.device:
                    ema_t = ema_t.to(p.data.device)
                    self.ema[name] = ema_t
                saved[name] = p.data
                p.data = ema_t
        try:
            yield m
        finally:
            for name, p in unwrapped.named_parameters():
                if name in saved:
                    p.data = saved[name]

    def on_optimizer_step(self, rt: Any) -> None:
        m = unwrap(rt.student_model)
        first = len(self.ema) == 0
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if first or name not in self.ema:
                self.ema[name] = p.data.detach().clone()
            else:
                ema_t = self.ema[name]
                if ema_t.device != p.data.device:
                    ema_t = ema_t.to(p.data.device)
                    self.ema[name] = ema_t
                ema_t.mul_(1.0 - self.cfg.rate).add_(p.data, alpha=self.cfg.rate)
        self._sync_cached_head(rt)

    def state_dict(self) -> dict[str, Any]:
        return {"ema": {n: t.detach().cpu() for n, t in self.ema.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "ema" in state:
            self.ema = {n: t.clone() for n, t in state["ema"].items()}
            self._cached_head = None


@WEIGHT_POLICIES.register("periodic_sync")
class PeriodicSync(WeightPolicy):
    """Teacher = a frozen copy refreshed every N steps.

    Checkpoints retain the frozen copy, including every=0's initial-policy reference.
    Refresh intervals use the trainer/rollout step index. Refresh only after a successful rollout,
    once after its PPO mini-batches; skipped steps still advance the trainer index.
    """

    class Config(BaseConfig):
        every: int = 20

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        m = unwrap(rt.student_model)
        self.copy = copy.deepcopy(m).eval().requires_grad_(False)

    @contextlib.contextmanager
    def context(self, rt: Any) -> Iterator[nn.Module]:
        yield self.copy

    def on_optimizer_step(self, rt: Any) -> None:
        if self.cfg.every > 0 and rt.state.step % self.cfg.every == 0:
            m = unwrap(rt.student_model)
            self.copy.load_state_dict(m.state_dict())

    def state_dict(self) -> dict[str, Any]:
        return {"copy": {n: t.detach().cpu().clone() for n, t in self.copy.state_dict().items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "copy" in state:
            self.copy.load_state_dict(state["copy"])


@WEIGHT_POLICIES.register("trust_region", source="Self-Distill/SDPO-main/verl/workers/actor/dp_actor.py:51-64")
class TrustRegion(WeightPolicy):
    """Teacher logits = lerp(frozen reference logits, live student logits, mix)."""

    class Config(BaseConfig):
        mix: float = Field(default=.5, ge=0, le=1)

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        model = unwrap(rt.student_model)
        self.ref = None if hasattr(model, "disable_adapter") else copy.deepcopy(model).eval().requires_grad_(False)
        self.warned_full = False

    @contextlib.contextmanager
    def context(self, rt: Any) -> Iterator[nn.Module]:
        if self.ref is None:
            with rt.student_adapter.adapters_disabled(rt.student_model):
                yield rt.student_model
        else:
            yield self.ref

    def state_dict(self) -> dict[str, Any]:
        return {"ref": {n: t.detach().cpu().clone() for n, t in self.ref.state_dict().items()}} if self.ref is not None else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.ref is not None and "ref" in state:
            self.ref.load_state_dict(state["ref"])
