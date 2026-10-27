"""Criterion contract (DESIGN §6.1-6.2).

A criterion:
  * declares ``features()`` (what logits/hidden it needs) and ``requires`` (batch keys),
  * may register learnable modules in ``build_modules`` and do one-time work in ``prepare``,
  * returns :class:`LossOutput` whose terms carry a *numerator* and a ``reduction`` type.
    The aggregator divides by the GLOBAL denominator (all micro-batches x all ranks):
      token_sum -> / Z_tok   (number of supervised target tokens in the optimizer step)
      seq_sum   -> / Z_seq   (number of sequences in the optimizer step)
      mean      -> / n_micro (already normalized inside the micro-batch)
  * never calls the model itself, never shifts logits, never divides by batch size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn

from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig
from lmkd.core.state import TrainState
from lmkd.models.forward import FeatureRequest, SideOutput
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherOutput

Reduction = Literal["token_sum", "seq_sum", "mean"]


@dataclass
class LossTerm:
    """One named loss term: a NUMERATOR plus how to normalize it (token_sum | seq_sum | mean); the aggregator
    divides.
    """
    value: torch.Tensor                    # numerator (scalar tensor with grad)
    reduction: Reduction
    log_value: torch.Tensor | None = None  # what to log if different (e.g. CSD surrogate)


@dataclass
class LossOutput:
    """What a criterion returns: named :class:`LossTerm`s plus detached diagnostic metrics."""
    terms: dict[str, LossTerm]
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)  # detached diagnostics (summed over micro)


@dataclass
class ModelPairInfo:
    """Static info for build_modules (dims, layers, vocab)."""
    student_hidden: int
    student_layers: int
    student_vocab: int
    teacher_hidden: int | None = None
    teacher_layers: int | None = None
    teacher_vocab: int | None = None


@dataclass
class LossContext:
    """Everything a criterion may read in ``forward``: batch, student/teacher outputs, aux modules, state, runtime."""
    batch: Batch
    student: SideOutput                       # first view (grad enabled)
    student_views: list[SideOutput]
    teacher: TeacherOutput | None
    modules: Any                              # lmkd.models.modules.ModuleRegistry
    state: TrainState
    rt: Any                                   # Runtime (adapters, models) - read only
    shared: dict = field(default_factory=dict)  # per-step cache shared between criteria

    def student_logits(self) -> LogitsView:
        """The student's :class:`LogitsView`; raises a clear error for models without an LM head."""
        view = self.student.logits
        if view is None:
            raise ValueError("this loss needs student logits but the model has no lm_head (encoder model?)")
        return view

    def supervised_mask(self, side: str = "student") -> torch.Tensor:
        """(labels != -100) * seq.loss_mask (if present), float [B,T]."""
        from lmkd.data.labels import supervised_mask

        lm = self.batch.get("seq.loss_mask") if side == "student" else None
        return supervised_mask(self.batch[f"{side}.labels"], lm)


@dataclass
class PrepareContext:
    """Context of ``Criterion.prepare``: the runtime and a few sample batches for one-time calibration."""
    rt: Any
    sample_batches: list[Batch] = field(default_factory=list)


class Criterion(nn.Module):
    """A loss (axis D). Declares ``features()`` (what to compute), ``requires`` (batch keys), builds aux modules,
    returns :class:`LossOutput`. Never calls the model.
    """
    requires: ClassVar[tuple[str, ...]] = ()
    supports_packing: ClassVar[bool] = True
    needs_teacher: ClassVar[bool] = False

    class Config(BaseConfig):
        pass

    provides: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        return tuple(cls.provides)

    @classmethod
    def required_keys(cls, cfg: Any) -> tuple[str, ...]:
        return tuple(cls.requires)

    cfg: Any   # the plugin's own nested Config (see core.plugin.Plugin)

    def __init__(self, cfg: BaseConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else self.Config()
        self.alias: str = getattr(type(self), "registry_name", type(self).__name__)

    # ---- declarations
    def features(self) -> FeatureRequest:
        return FeatureRequest()

    # ---- lifecycle
    def build_modules(self, info: ModelPairInfo, modules: Any) -> None:
        return None

    def prepare(self, ctx: PrepareContext) -> None:
        return None

    def forward(self, ctx: LossContext) -> LossOutput:
        raise NotImplementedError

    def on_step_end(self, state: TrainState) -> None:
        return None

    def extra_state(self) -> dict:
        return {}

    def load_extra_state(self, state: dict) -> None:
        return None
