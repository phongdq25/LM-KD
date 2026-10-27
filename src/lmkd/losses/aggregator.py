"""LossAggregator: combine weighted criterion outputs with GLOBAL normalization (DESIGN §6.2-6.3)."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import AGGREGATORS
from lmkd.losses.base import LossOutput


@dataclass
class Norms:
    """Global denominators for ONE optimizer step (all micro-batches x all ranks)."""
    z_tok: float
    z_seq: float
    n_micro: int

    def divide(self, value: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "token_sum":
            return value / max(self.z_tok, 1.0)
        if reduction == "seq_sum":
            return value / max(self.z_seq, 1.0)
        if reduction == "mean":
            return value / max(self.n_micro, 1)
        raise ValueError(f"unknown reduction {reduction!r}")


@dataclass
class WeightedOutput:
    """One criterion's :class:`LossOutput` with its current weight and alias."""
    alias: str
    weight: float
    out: LossOutput


class LossAggregator(Plugin):
    """Combines weighted criterion outputs into the scalar loss, dividing each term by its GLOBAL denominator
    (``Norms``).
    """
    def combine(self, outputs: list[WeightedOutput], norms: Norms) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    @staticmethod
    def _normalized_terms(o: WeightedOutput, norms: Norms) -> Iterator[tuple[str, torch.Tensor, torch.Tensor | None]]:
        for name, term in o.out.terms.items():
            yield name, norms.divide(term.value, term.reduction), (
                norms.divide(term.log_value, term.reduction) if term.log_value is not None else None
            )


@AGGREGATORS.register("weighted_sum")
class WeightedSum(LossAggregator):
    """L = sum_i w_i * sum_terms(normalized term)."""

    class Config(BaseConfig):
        pass

    def combine(self, outputs, norms):
        total = None
        metrics: dict[str, torch.Tensor] = {}
        for o in outputs:
            for name, val, logv in self._normalized_terms(o, norms):
                contrib = o.weight * val
                total = contrib if total is None else total + contrib
                metrics[f"loss/{o.alias}/{name}"] = (logv if logv is not None else val).detach()
            for k, v in o.out.metrics.items():
                metrics[f"metric/{o.alias}/{k}"] = v.detach() if isinstance(v, torch.Tensor) else torch.tensor(v)
        if total is None:
            raise ValueError("no loss terms produced")
        metrics["loss/total"] = total.detach()
        return total, metrics


@AGGREGATORS.register("kd_mix", source="DistiLLM / DSKD: (1-r)*CE + r*KD")
class KDMix(WeightedSum):
    """Rescales weights: the term aliased ``ce_alias`` gets (1-kd_ratio)*w, all others kd_ratio*w."""

    class Config(BaseConfig):
        kd_ratio: float = 0.5
        ce_alias: str = "nll"

    def combine(self, outputs, norms):
        r = self.cfg.kd_ratio
        scaled = [
            WeightedOutput(o.alias, o.weight * ((1 - r) if o.alias == self.cfg.ce_alias else r), o.out)
            for o in outputs
        ]
        return super().combine(scaled, norms)


@AGGREGATORS.register("gradmag", source="white-box-distillation/cross-tokenizer/IMPACT/code/alm_multitask.py")
class GradMag(LossAggregator):
    """L = sum_i inverse-gradient-norm weight_i * configured weight_i * normalized task_i.

    Source: IMPACT/code/alm_multitask.py:68-116. Requires step: multi_objective_grad.
    """

    class Config(BaseConfig):
        mode: str = "approx_gradmag_preserve_mag"
        epsilon: float = 1e-8

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._weights: torch.Tensor | None = None
        self._grad_norms: torch.Tensor | None = None
        self._warned_weight = False

    def setup(self, rt) -> None:
        step = rt.config.step
        kind = step if isinstance(step, str) else step.get("type")
        combine = "gradmag" if isinstance(step, str) else step.get("combine", "gradmag")
        if kind != "multi_objective_grad" or combine != "gradmag":
            raise ValueError("aggregator 'gradmag' requires step: multi_objective_grad, combine: gradmag")

    def task_losses(self, outputs: list[WeightedOutput], norms: Norms) -> list[tuple[str, torch.Tensor]]:
        """One normalized unweighted task loss per criterion alias."""
        tasks = []
        for o in outputs:
            values = [val for _, val, _ in self._normalized_terms(o, norms)]
            if not values:
                raise ValueError(f"criterion {o.alias!r} produced no loss terms")
            loss = values[0]
            for value in values[1:]:
                loss = loss + value
            tasks.append((o.alias, loss))
        return tasks

    def set_weights(self, weights: torch.Tensor, grad_norms: torch.Tensor) -> None:
        """Set detached weights for exactly the next micro-batch."""
        self._weights = weights.detach()
        self._grad_norms = grad_norms.detach()

    def combine(self, outputs: list[WeightedOutput], norms: Norms) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import logging

        if self._weights is None or self._grad_norms is None:
            if not torch.is_grad_enabled():
                tasks = self.task_losses(outputs, norms)
                device = tasks[0][1].device if tasks else torch.device("cpu")
                w = torch.ones(len(outputs), device=device) / max(1, len(outputs))
                gn = torch.zeros(len(outputs), device=device)
                self.set_weights(w, gn)
            else:
                raise RuntimeError("aggregator 'gradmag' requires step: multi_objective_grad")
        assert self._weights is not None and self._grad_norms is not None
        weights, grad_norms = self._weights, self._grad_norms
        self._weights = self._grad_norms = None
        if len(weights) != len(outputs):
            raise ValueError("GradMag weights must match the number of criterion outputs")
        metrics: dict[str, torch.Tensor] = {}
        total = None
        for i, o in enumerate(outputs):
            if o.weight != 1.0 and not self._warned_weight:
                logging.getLogger(__name__).warning("GradMag applies configured term weights after gradient weights")
                self._warned_weight = True
            contrib = weights[i] * o.weight * self.task_losses([o], norms)[0][1]
            total = contrib if total is None else total + contrib
            for name, val, logv in self._normalized_terms(o, norms):
                metrics[f"loss/{o.alias}/{name}"] = (logv if logv is not None else val).detach()
            for key, value in o.out.metrics.items():
                metrics[f"metric/{o.alias}/{key}"] = value.detach() if isinstance(value, torch.Tensor) else torch.tensor(value)
            metrics[f"gradmag/weight_{o.alias}"] = weights[i]
            metrics[f"gradmag/norm_{o.alias}"] = grad_norms[i]
        if total is None:
            raise ValueError("no loss terms produced")
        metrics["loss/total"] = total.detach()
        return total, metrics
