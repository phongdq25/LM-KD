"""Runtime = every built component of a run, wired together. Plugins receive it in ``setup(rt)``
and through contexts; they must treat it as READ-ONLY except for their own state.

Lifecycle: ``build_runtime`` (train/builder.py) fills the fields in a fixed order; the lightweight
pipeline-only runtime (``pipeline.runner.build_runtime_for_pipeline``) fills only the tokenizers,
``artifacts`` and ``data_ctx``. Fields typed without ``| None`` are therefore always set once a
training runtime is built (they hold ``_UNSET`` until then); ``| None`` fields are genuinely optional.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from lmkd.core.config import RecipeConfig
from lmkd.core.state import TrainState

if TYPE_CHECKING:  # type-only imports: avoid import cycles at runtime
    from transformers import PreTrainedTokenizerBase

    from lmkd.core.batch import Batch
    from lmkd.data.collator import Collator
    from lmkd.data.context import DataContext
    from lmkd.data.dataset import PreparedDataset
    from lmkd.data.producers import BatchProducer
    from lmkd.eval.base import Evaluator
    from lmkd.losses.aggregator import LossAggregator
    from lmkd.losses.base import Criterion, LossOutput
    from lmkd.losses.simple import ModelView
    from lmkd.models.adapters import HFAutoAdapter
    from lmkd.models.modules import ModuleRegistry
    from lmkd.pipeline.artifacts import ArtifactStore
    from lmkd.rollout.base import RolloutEngine
    from lmkd.teachers.input_builders import TeacherInputBuilder
    from lmkd.teachers.providers import TeacherProvider
    from lmkd.train.algorithms import Algorithm
    from lmkd.train.callbacks import CallbackList
    from lmkd.train.engine import Engine
    from lmkd.train.grad import GradTransform, ParamWithGroup, TensorGradHook
    from lmkd.train.step_strategies import StepOutput, StepStrategy

_UNSET: Any = None   # placeholder for "built later by build_runtime" (see module docstring)

LossFn = Callable[["Batch", "Runtime"], "torch.Tensor | dict[str, torch.Tensor] | LossOutput"]
EvalFn = Callable[["Runtime"], dict[str, float]]


@dataclass
class CriterionEntry:
    """A built criterion with its alias and base weight."""
    alias: str
    criterion: Criterion
    base_weight: float


@dataclass
class Runtime:
    """Every built component of a run (models, data, objective, optimization, callbacks) plus the library-mode
    helpers: ``begin / train_step / end / forward / evaluate / save``.
    """
    config: RecipeConfig
    engine: Engine
    state: TrainState = field(default_factory=TrainState)
    # models
    student_model: nn.Module = _UNSET         # raw HF/PEFT model (save, lm_head, adapter queries)
    student_module: nn.Module = _UNSET        # engine-wrapped HiddenForward: the ONLY forward path (K1)
    student_adapter: HFAutoAdapter = _UNSET
    student_tokenizer: PreTrainedTokenizerBase = _UNSET
    teacher_model: nn.Module | None = None
    teacher_adapter: HFAutoAdapter | None = None
    teacher_tokenizer: PreTrainedTokenizerBase | None = None
    modules: ModuleRegistry = _UNSET          # learnable aux modules (projectors...) owned by criteria
    # data
    data_ctx: DataContext = _UNSET
    train_dataset: PreparedDataset = _UNSET
    eval_dataset: Any = None
    collator: Collator = _UNSET
    producer: BatchProducer = _UNSET
    artifacts: ArtifactStore = _UNSET
    rollout: RolloutEngine | None = None
    # objective
    teacher: TeacherProvider = _UNSET
    teacher_input: TeacherInputBuilder = _UNSET
    criteria: list[CriterionEntry] = field(default_factory=list)
    loss_weights: dict[str, float] = field(default_factory=dict)  # mutable by schedules
    aggregator: LossAggregator = _UNSET
    algorithm: Algorithm = _UNSET
    # optimization
    step_strategy: StepStrategy = _UNSET
    optimizers: dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    schedulers: list[Any] = field(default_factory=list)   # torch LRScheduler instances
    grad_transforms: list[GradTransform] = field(default_factory=list)
    grad_hooks: list[TensorGradHook] = field(default_factory=list)
    trainable_params: list[ParamWithGroup] = field(default_factory=list)
    callbacks: CallbackList = _UNSET
    evaluators: list[Evaluator | EvalFn] = field(default_factory=list)   # plugins or callables fn(rt) -> dict
    loss_fn: LossFn | None = None             # tier-0 loss fn(batch, rt) -> Tensor | dict | LossOutput
    extra: dict[str, Any] = field(default_factory=dict)

    # ================================================================== library mode (lmkd.api)
    # The building blocks of Trainer.fit, usable from a hand-written loop:
    #   rt.begin(); while not rt.done: rt.train_step(loss_fn=...); rt.end()

    @property
    def done(self) -> bool:
        return self.state.done

    def begin(self) -> None:
        """Train mode + ``callbacks.on_train_begin``. Call once before the first step."""
        if self.config.models.student.trainable:
            self.student_model.train()
        self.modules.train()
        self.callbacks.on_train_begin(self)

    def next_batch(self) -> Batch:
        """Global batch for the next optimizer step (from the configured producer)."""
        return self.producer.next(self.state)

    def train_step(self, batch: Batch | None = None, loss_fn: LossFn | None = None) -> StepOutput:
        """ONE optimizer step with the full machinery: callbacks (schedules, log, ckpt),
        teacher-input augment, micro-batching, global loss normalization, no_sync/autocast, grad
        transforms, optimizers, schedulers, state update, ``criterion.on_step_end``.

        ``batch``: global batch for this step; None = take it from the producer (after
        ``on_step_begin``, as Trainer.fit does). ``loss_fn(micro_batch, rt)`` (optional, tier 0) is
        ADDED to the recipe criteria for this step; it returns a scalar Tensor (mean over the
        micro-batch), a dict of them, or a LossOutput. Returns the StepOutput (``.metrics``)."""
        self.callbacks.on_step_begin(self)
        if batch is None:
            batch = self.next_batch()
        batch = self.algorithm.augment(batch, self)
        prev = self.loss_fn
        if loss_fn is not None:
            self.loss_fn = loss_fn
        try:
            out = self.step_strategy.run(batch, self)
        finally:
            self.loss_fn = prev
        st = self.state
        st.step += 1
        st.tokens_seen += out.tokens
        st.samples_seen += out.samples
        st.done = st.step >= st.total_steps
        # PPO holds the teacher fixed across mini-batches; update once per successful rollout.
        if self.teacher is not None and not out.skipped:
            self.teacher.on_optimizer_step(st)
        for e in self.criteria:
            e.criterion.on_step_end(st)
        self.callbacks.on_step_end(self, out)
        return out

    def end(self) -> None:
        self.callbacks.on_train_end(self)

    def fit(self) -> Runtime:
        self.begin()
        while not self.done:
            self.train_step()
        self.end()
        return self

    def forward(self, batch: Batch, *, hidden: bool = False, logits: bool = True, grad: bool = True) -> ModelView:
        """Student forward on ``student.*`` of ``batch`` -> :class:`lmkd.losses.simple.ModelView`
        (HF convention: ``logits[:, t]`` predicts ``input_ids[:, t+1]``). Goes through the engine
        wrapper, so it is safe under DeepSpeed; use it instead of ``rt.student_model(...)``."""
        from lmkd.losses.simple import run_side

        return run_side(self, batch, "student", hidden=hidden, logits=logits, grad=grad)

    def teacher_forward(self, batch: Batch, *, hidden: bool = False, logits: bool = True) -> ModelView:
        """Teacher forward (no grad) on ``teacher.*`` if present, else ``student.*``."""
        from lmkd.losses.simple import run_side

        return run_side(self, batch, "teacher", hidden=hidden, logits=logits, grad=False)

    def evaluate(self) -> dict[str, float]:
        """Run every evaluator (plugin ``.evaluate(rt)`` or callable ``fn(rt)``), store the merged
        metrics in ``state.extra["eval"]`` (keep-best checkpoints read it), fire ``on_eval``."""
        was_training = self.student_model.training
        self.student_model.eval()
        metrics: dict[str, float] = {}
        try:
            for ev in self.evaluators:
                metrics.update(ev.evaluate(self) if hasattr(ev, "evaluate") else ev(self))
        finally:
            self.student_model.train(was_training)
        self.state.extra["eval"] = metrics
        self.callbacks.on_eval(self, metrics)
        if hasattr(self.producer, "on_eval"):
            self.producer.on_eval(metrics)
        return metrics

    def save(self, tag: str) -> Path:
        """Full checkpoint ``<output_dir>/checkpoint-<tag>`` (collective: call on every rank)."""
        from lmkd.train.checkpointing import save_checkpoint

        return save_checkpoint(self, tag)

    def unwrap_model(self) -> nn.Module:
        """The raw HF/PEFT student (for generation, save_pretrained, custom eval)."""
        return self.student_model
