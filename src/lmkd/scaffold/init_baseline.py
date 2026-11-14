"""Template generator for initializing a full baseline/method package (``lmkd init``)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_PLUGIN_FILES = ("config.py", "loss.py", "data.py", "teacher.py", "train.py", "eval.py")


def _to_class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part)


def _resolve_root(root: Path | None = None) -> Path:
    if root is not None:
        return root
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists() or (cwd / "src" / "lmkd").exists() or (cwd / "recipes").exists():
        return cwd
    if (cwd / "lm-kd" / "pyproject.toml").exists():
        return cwd / "lm-kd"
    return cwd


def parse_init_target(
    raw_name: str,
    target: str = "dev",
    domain: str | None = None,
) -> tuple[str, str, str | None]:
    """Parse ``raw_name`` (which may include path prefixes like ``dev/foo`` or ``baselines/bar``)
    into ``(clean_name, normalized_target, domain)``.
    """
    parts = [p for p in re.split(r"[/\\]+", raw_name.strip()) if p and p != "."]
    if not parts:
        raise ValueError("Method name cannot be empty.")

    if parts[0] == "src" and len(parts) > 1:
        parts = parts[1:]

    if parts[0] in ("dev", "baseline", "baselines"):
        target = "baseline" if parts[0] in ("baseline", "baselines") else "dev"
        parts = parts[1:]

    if not parts:
        raise ValueError("Method name cannot be empty after target prefix.")

    if len(parts) >= 2 and domain is None:
        domain = "/".join(parts[:-1])
        clean = parts[-1]
    else:
        clean = parts[-1]

    clean = clean.replace("-", "_").strip()
    if not clean.isidentifier():
        raise ValueError(f"Invalid method name '{clean}': must be a valid Python identifier (e.g. 'my_method').")

    norm_target = "baseline" if target in ("baseline", "baselines") else "dev"
    return clean, norm_target, domain


def _render_init_py(name: str, cls_name: str) -> str:
    return f'''"""{cls_name} baseline/method package for LM-KD.

Unmodified components delegate to canonical framework defaults.
Edit only the files you need:
  - loss.py    : Criterion / custom loss function (Axis D)
  - data.py    : Transform, BatchProducer, offline Stage (Axis A & B)
  - teacher.py : TeacherInputBuilder & TeacherProvider (Axis C)
  - train.py   : Algorithm, StepStrategy, Callback, custom Trainer loop (Axis E & F)
  - eval.py    : Evaluator and pure metric function
  - config.py  : Method-level default hyperparameters
"""

from __future__ import annotations

from pathlib import Path

from lmkd.plugins import load_plugins

if __package__:
    from .config import {cls_name}Config
    from .data import {cls_name}Producer, {cls_name}Stage, {cls_name}Transform
    from .eval import {cls_name}Evaluator, {name}_metric
    from .loss import {cls_name}Loss
    from .teacher import {cls_name}Teacher, {cls_name}TeacherInput
    from .train import (
        {cls_name}Algorithm,
        {cls_name}Callback,
        {cls_name}Step,
        {cls_name}Trainer,
        run_training,
    )

    __all__ = [
        "{cls_name}Algorithm",
        "{cls_name}Callback",
        "{cls_name}Config",
        "{cls_name}Evaluator",
        "{cls_name}Loss",
        "{cls_name}Producer",
        "{cls_name}Stage",
        "{cls_name}Step",
        "{cls_name}Teacher",
        "{cls_name}TeacherInput",
        "{cls_name}Trainer",
        "{cls_name}Transform",
        "{name}_metric",
        "run_training",
    ]
else:
    _HERE = Path(__file__).resolve().parent
    load_plugins({list(_PLUGIN_FILES)!r}, base_dir=_HERE)
'''


def _render_config_py(name: str, cls_name: str) -> str:
    return f'''"""Configuration schema and default hyperparameters for {name}."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class {cls_name}Config:
    """Hyperparameter specification for {name}."""

    temperature: float = 1.0
    alpha: float = 0.5
    chunk_size: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)
'''


def _render_loss_py(name: str, cls_name: str) -> str:
    return f'''"""Loss / Criterion definitions for {name} (Axis D).

If left unmodified, ``{cls_name}Loss`` calls ``super().forward(ctx)`` (canonical NLL loss).
Override ``features()``, ``build_modules()``, ``prepare()``, or ``forward()`` to implement your objective.
"""

from __future__ import annotations

from typing import ClassVar

from lmkd.core.registry import CRITERIONS
from lmkd.core.state import TrainState
from lmkd.losses.base import (
    Criterion,
    LossContext,
    LossOutput,
    LossTerm,
    ModelPairInfo,
    PrepareContext,
)
from lmkd.losses.sft import NLL
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherSignalSpec


@CRITERIONS.register("{name}", status="experimental", force=True)
class {cls_name}Loss(NLL):
    """{cls_name} distillation criterion (defaults to standard NLL if unmodified)."""

    # Set needs_teacher = True when your objective requires teacher signals.
    needs_teacher: ClassVar[bool] = False
    supports_packing: ClassVar[bool] = True

    class Config(NLL.Config):
        temperature: float = 1.0
        alpha: float = 0.5

    def features(self) -> FeatureRequest:
        """Declare student/teacher representations needed for forward().

        Unmodified: requests student logits view via ``super().features()``.
        Example for teacher + student logits & last hidden layer using ``self.make_features``:
            return self.make_features(
                student_logits="view",
                teacher_logits="view",
                student_hidden=(-1,),
                teacher_hidden=(-1,),
            )
        """
        _ = (SideRequest, TeacherSignalSpec)
        return super().features()

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        """Register learnable auxiliary modules (e.g., hidden-state projectors).

        Example:
            import torch.nn as nn
            t_dim = info.teacher_hidden or info.student_hidden
            self.register_aux_module(modules, "proj", nn.Linear(info.student_hidden, t_dim))
        """
        super().build_modules(info, modules)

    def prepare(self, ctx: PrepareContext) -> None:
        """Optional one-time initialization or calibration before training starts."""
        super().prepare(ctx)

    def forward(self, ctx: LossContext) -> LossOutput:
        """Compute loss numerators and diagnostic metrics for one micro-batch.

        Unmodified: delegates to ``NLL.forward(ctx)``.
        Example custom implementation using ``LossContext`` shortcuts:
            logp = ctx.student_logits().with_temperature(self.cfg.temperature).target_logp(ctx.labels)
            loss_num = -(logp * ctx.mask).sum()
            return LossOutput(
                terms={{"{name}": LossTerm(loss_num, "token_sum")}},
                metrics={{"{name}/tokens": ctx.mask.sum().detach()}},
            )
        """
        _ = (Criterion, LossTerm)
        return super().forward(ctx)

    def on_step_end(self, state: TrainState) -> None:
        """Optional hook called at the end of every optimizer step."""
        super().on_step_end(state)


__all__ = ["{cls_name}Loss"]
'''


def _render_data_py(name: str, cls_name: str) -> str:
    return f'''"""Data processing, batch production, and offline stage templates for {name} (Axis A & B).

If left unmodified:
  - ``{cls_name}Transform`` passes each sample through unchanged.
  - ``{cls_name}Producer`` delegates to ``DatasetProducer`` (standard off-policy DataLoader).
  - ``{cls_name}Stage`` provides a no-op offline pipeline stage skeleton.
"""

from __future__ import annotations

from typing import Any, ClassVar

from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import PRODUCERS, STAGES, TRANSFORMS
from lmkd.core.state import TrainState
from lmkd.data.context import DataContext
from lmkd.data.producers import DatasetProducer
from lmkd.data.transforms.base import Transform
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.stage import Stage, StageContext


@TRANSFORMS.register("{name}_transform", status="experimental", force=True)
class {cls_name}Transform(Transform):
    """Per-sample data transform for {name} (no-op pass-through by default)."""

    requires: ClassVar[tuple[str, ...]] = ("student.input_ids",)
    provides: ClassVar[tuple[str, ...]] = ()

    class Config(BaseConfig):
        pass

    def setup(self, rt: Any) -> None:
        super().setup(rt)

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        """Modify ``sample`` in-place or return ``None`` to drop the sample."""
        _ = ctx
        return sample


@PRODUCERS.register("{name}_producer", status="experimental", force=True)
class {cls_name}Producer(DatasetProducer):
    """Batch producer for {name} (delegates to standard DatasetProducer by default)."""

    class Config(DatasetProducer.Config):
        pass

    def setup(self, rt: Any) -> None:
        super().setup(rt)

    def next(self, state: TrainState) -> Batch:
        """Return the global Batch for one optimizer step.

        Override to implement on-policy rollout generation (e.g. ``make_on_policy_batch``),
        curriculum sampling, or replay buffer mixing.
        """
        return super().next(state)

    def on_eval(self, metrics: dict[str, Any]) -> None:
        super().on_eval(metrics)

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)


@STAGES.register("{name}_stage", status="experimental", force=True)
class {cls_name}Stage(Stage):
    """Offline pipeline stage for {name} (cached, content-addressed preprocessing)."""

    code_version: ClassVar[str] = "1"

    class Config(BaseConfig):
        pass

    def run(self, ctx: StageContext) -> dict[str, Any]:
        """Execute offline computation and write artifacts to ``ctx.output.path``."""
        if ctx.output is not None:
            ctx.output.path.mkdir(parents=True, exist_ok=True)
        return {{"status": "ok"}}

    def attach(self, art: Artifact, rt: Any) -> None:
        """Expose cached stage artifacts to the runtime before training starts."""
        super().attach(art, rt)


__all__ = ["{cls_name}Producer", "{cls_name}Stage", "{cls_name}Transform"]
'''


def _render_teacher_py(name: str, cls_name: str) -> str:
    return f'''"""Teacher input builder and teacher provider templates for {name} (Axis C).

If left unmodified:
  - ``{cls_name}TeacherInput`` delegates to ``SameInput`` (teacher sees the student's batch).
  - ``{cls_name}Teacher`` delegates to ``HFFrozenTeacher`` (online frozen teacher model).
"""

from __future__ import annotations

from typing import Any

from lmkd.core.batch import Batch
from lmkd.core.registry import TEACHER_INPUTS, TEACHERS
from lmkd.core.state import TrainState
from lmkd.models.forward import SideOutput, SideRequest
from lmkd.teachers.input_builders import SameInput
from lmkd.teachers.providers import HFFrozenTeacher
from lmkd.teachers.signals import TeacherOutput, TeacherSignalSpec


@TEACHER_INPUTS.register("{name}_teacher_input", status="experimental", force=True)
class {cls_name}TeacherInput(SameInput):
    """Teacher input builder for {name} (defaults to SameInput if unmodified)."""

    class Config(SameInput.Config):
        pass

    def setup(self, rt: Any) -> None:
        super().setup(rt)

    def build(self, batch: Batch, rt: Any) -> Batch:
        """Construct ``teacher.*`` batch keys from the student batch.

        Override to build privileged prompts, hindsight context, or cross-tokenizer inputs.
        """
        return super().build(batch, rt)


@TEACHERS.register("{name}_teacher", status="experimental", force=True)
class {cls_name}Teacher(HFFrozenTeacher):
    """Teacher signal provider for {name} (defaults to HFFrozenTeacher if unmodified)."""

    class Config(HFFrozenTeacher.Config):
        pass

    def setup(self, rt: Any) -> None:
        super().setup(rt)

    def signals(
        self,
        batch: Batch,
        spec: TeacherSignalSpec | None,
        req: SideRequest | None,
        student: SideOutput | None,
    ) -> TeacherOutput | None:
        """Compute teacher signals (logits, top-k, logp, or hidden states)."""
        return super().signals(batch, spec, req, student)

    def on_optimizer_step(self, state: TrainState) -> None:
        """Hook called after each successful optimizer update (e.g. for EMA teacher updates)."""
        super().on_optimizer_step(state)


__all__ = ["{cls_name}Teacher", "{cls_name}TeacherInput"]
'''


def _render_train_py(name: str, cls_name: str) -> str:
    return f'''"""Training loop, Algorithm, StepStrategy, and Callback templates for {name} (Axis E & F).

If left unmodified:
  - ``{cls_name}Algorithm`` delegates to ``Standard`` (forward + teacher signals + criteria).
  - ``{cls_name}Step`` delegates to ``StandardStep`` (micro-batch accumulation, global normalization, optimizer step).
  - ``{cls_name}Callback`` delegates to ``Callback`` (lifecycle hooks).
  - ``{cls_name}Trainer`` runs the canonical ``rt.begin() -> while not rt.done: rt.train_step() -> rt.end()`` loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

import lmkd
from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ALGORITHMS, CALLBACKS, STEP_STRATEGIES
from lmkd.losses.aggregator import Norms, WeightedOutput
from lmkd.train.algorithms import Standard
from lmkd.train.callbacks import Callback
from lmkd.train.runtime import Runtime
from lmkd.train.step_strategies import StandardStep, StepOutput
from lmkd.train.trainer import Trainer


@ALGORITHMS.register("{name}_algorithm", status="experimental", force=True)
class {cls_name}Algorithm(Standard):
    """Algorithm orchestration for {name} (batch augmentation and micro-batch loss evaluation)."""

    class Config(Standard.Config):
        pass

    def setup(self, rt: Any) -> None:
        super().setup(rt)

    def augment(self, batch: Batch, rt: Any) -> Batch:
        """Prepare step-level batch (e.g. teacher inputs, advantages) before micro-batch splitting."""
        return super().augment(batch, rt)

    def loss(self, micro: Batch, rt: Any) -> list[WeightedOutput]:
        """Run forward passes and evaluate active criteria on one micro-batch."""
        return super().loss(micro, rt)


@STEP_STRATEGIES.register("{name}_step", status="experimental", force=True)
class {cls_name}Step(StandardStep):
    """Optimizer step strategy for {name} (micro-batching, backward, grad transforms, optimizer step)."""

    class Config(StandardStep.Config):
        pass

    def setup(self, rt: Any) -> None:
        super().setup(rt)

    def micro_batches(self, batch: Batch) -> list[Batch]:
        return super().micro_batches(batch)

    def global_norms(self, micros: list[Batch]) -> Norms:
        return super().global_norms(micros)

    def accumulate(self, micros: list[Batch], norms: Norms) -> dict[str, torch.Tensor]:
        """Forward and backward across micro-batches with global loss normalization."""
        return super().accumulate(micros, norms)

    def apply_grad_transforms(self, metrics: dict[str, Any]) -> None:
        """Unscale and apply gradient transforms (e.g. gradient clipping)."""
        super().apply_grad_transforms(metrics)

    def optimizer_step(self) -> bool:
        """Step optimizers and LR schedulers, then zero gradients."""
        return super().optimizer_step()

    def run(self, batch: Batch, rt: Any) -> StepOutput:
        """Execute one complete optimization step."""
        return super().run(batch, rt)


@CALLBACKS.register("{name}_callback", status="experimental", force=True)
class {cls_name}Callback(Callback):
    """Lifecycle callback hooks for {name}."""

    class Config(BaseConfig):
        pass

    def on_train_begin(self, rt: Any) -> None:
        super().on_train_begin(rt)

    def on_step_begin(self, rt: Any) -> None:
        super().on_step_begin(rt)

    def on_step_end(self, rt: Any, out: Any) -> None:
        super().on_step_end(rt, out)

    def on_eval(self, rt: Any, metrics: dict[str, Any]) -> None:
        super().on_eval(rt, metrics)

    def on_save(self, rt: Any, path: Path) -> None:
        super().on_save(rt, path)

    def on_train_end(self, rt: Any) -> None:
        super().on_train_end(rt)


class {cls_name}Trainer(Trainer):
    """Custom outer training loop for {name}.

    Unmodified: inherits the canonical ``Trainer.fit()`` (``begin -> train_step -> end``) loop.
    Override ``train_step`` or ``fit`` to customize outer-loop behavior while keeping ``self.rt.train_step()``.
    """


def run_training(
    recipe: str | Path | dict[str, Any],
    overrides: list[str] | None = None,
    imports: list[str] | None = None,
) -> Runtime:
    """Build the runtime from ``recipe`` and execute ``{cls_name}Trainer.fit()``."""
    here = Path(__file__).resolve().parent
    default_imports = [str(here / "__init__.py")]
    all_imports = list(dict.fromkeys([*default_imports, *(imports or [])]))
    rt = lmkd.build(recipe, overrides=overrides, imports=all_imports)
    return {cls_name}Trainer(rt).fit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train {name} baseline")
    default_recipe = Path(__file__).resolve().parent / "recipe.yaml"
    parser.add_argument("-c", "--config", default=str(default_recipe), help="Path to YAML recipe")
    parser.add_argument("--set", nargs="*", default=[], action="extend", help="Config overrides key=value")
    args = parser.parse_args(argv)
    run_training(args.config, overrides=args.set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "{cls_name}Algorithm",
    "{cls_name}Callback",
    "{cls_name}Step",
    "{cls_name}Trainer",
    "run_training",
]
'''


def _render_eval_py(name: str, cls_name: str) -> str:
    return f'''"""Evaluation plugin and pure metric function templates for {name}.

If left unmodified, ``{cls_name}Evaluator`` delegates to ``DevLoss`` (teacher-forced dev loss).
"""

from __future__ import annotations

from typing import Any

from lmkd.core.registry import EVALUATORS
from lmkd.eval.evaluators import DevLoss


def {name}_metric(model: Any, tokenizer: Any, records: list[dict[str, Any]]) -> dict[str, float]:
    """Pure evaluation function usable standalone or inside an Evaluator plugin."""
    _ = (model, tokenizer)
    return {{"num_samples": float(len(records))}}


@EVALUATORS.register("{name}_eval", status="experimental", force=True)
class {cls_name}Evaluator(DevLoss):
    """Evaluator plugin for {name} (defaults to DevLoss if unmodified)."""

    name_prefix = "{name}_eval"

    class Config(DevLoss.Config):
        pass

    def setup(self, rt: Any) -> None:
        super().setup(rt)

    def evaluate(self, rt: Any) -> dict[str, float]:
        """Evaluate student model on dev records and return metric dictionary."""
        return super().evaluate(rt)


__all__ = ["{cls_name}Evaluator", "{name}_metric"]
'''


def _render_recipe_yaml(name: str, import_prefix: str = "") -> str:
    imports_list = [f"{import_prefix}{fname}" for fname in _PLUGIN_FILES]
    imports_str = ", ".join(imports_list)
    return f"""name: {name}
imports: [{imports_str}]

engine:
  type: local
  device: cpu
  mixed_precision: "no"

models:
  student:
    path: sshleifer/tiny-gpt2
  # Uncomment below to enable a teacher model:
  # teacher:
  #   path: sshleifer/tiny-gpt2

data:
  train:
    type: jsonl
    path: examples/data/train_sample.jsonl
  transforms:
    - type: tokenize_sft
      use_chat_template: false
    - type: {name}_transform
  max_length: 256

producer:
  type: {name}_producer

teacher:
  provider: none
  # When using a teacher model, change provider to:
  # provider:
  #   type: {name}_teacher
  input:
    type: {name}_teacher_input

algorithm:
  type: {name}_algorithm

step:
  type: {name}_step

loss:
  aggregator: weighted_sum
  terms:
    - criterion: {name}
      weight: 1.0
      params:
        temperature: 1.0
        alpha: 0.5

optim:
  lr: 2e-5
  weight_decay: 0.0
  optimizers:
    default:
      type: adamw
  schedulers:
    - type: constant

grad:
  clip_norm: 1.0

callbacks:
  - type: {name}_callback

train:
  output_dir: outputs/{name}
  max_steps: 2
  batch_size: 2
  micro_batch_size: 1
  seed: 42
  log_every: 1
"""


def _render_readme_md(name: str, cls_name: str, rel_pkg_dir: str, rel_recipe_file: str) -> str:
    return f"""# {cls_name} (`{name}`)

Template generated by `lmkd init {name}`.
All components are pre-wired into `recipe.yaml` and inherit from LM-KD's canonical defaults.
**If you do not modify a file, it automatically uses the framework's default behavior.**

## Directory Structure (`{rel_pkg_dir}/`)

| File | Responsibility | Registered Plugins / Classes | Default Behavior (Unmodified) |
|---|---|---|---|
| `loss.py` | Objective / loss function (Axis D) | `@CRITERIONS.register("{name}")` (`{cls_name}Loss`) | Calls `NLL.forward(ctx)` |
| `data.py` | Data preprocessing, batching & offline stage (Axis A & B) | `{name}_transform`, `{name}_producer`, `{name}_stage` | Pass-through transform + `DatasetProducer` |
| `teacher.py` | Teacher input & signal provider (Axis C) | `{name}_teacher_input`, `{name}_teacher` | `SameInput` + `HFFrozenTeacher` |
| `train.py` | Training loop, algorithm, step strategy & callbacks (Axis E & F) | `{name}_algorithm`, `{name}_step`, `{name}_callback`, `{cls_name}Trainer` | `Standard` + `StandardStep` + `Trainer.fit()` |
| `eval.py` | Evaluation metric & plugin | `{name}_eval` (`{cls_name}Evaluator`), `{name}_metric` | `DevLoss.evaluate(rt)` |
| `config.py` | Method hyperparameter schema | `{cls_name}Config` | Default hyperparameter values |
| `recipe.yaml` | Self-contained experiment config | Wires all `{name}_*` plugins | Ready for `lmkd check` / `lmkd train` |

## Quick Commands

```bash
# 1. Fast synthetic gradient probe on the loss
lmkd try {name} --plugins {rel_pkg_dir}/__init__.py

# 2. Validate recipe configuration and plugin graph
lmkd check -c {rel_pkg_dir}/recipe.yaml
lmkd check -c {rel_recipe_file}

# 3. Train using standard CLI or custom train.py script
lmkd train -c {rel_pkg_dir}/recipe.yaml
python {rel_pkg_dir}/train.py -c {rel_pkg_dir}/recipe.yaml

# 4. Run unit tests for this method
pytest -q tests/unit/test_{name}.py
```
"""


def _render_test_py(name: str, rel_pkg_dir: str, rel_recipe_path: str) -> str:
    return f'''"""Unit and smoke tests for initialized method '{name}'."""

from __future__ import annotations

from pathlib import Path

from lmkd.core.config import load_recipe
from lmkd.core.registry import (
    ALGORITHMS,
    CALLBACKS,
    CRITERIONS,
    EVALUATORS,
    PRODUCERS,
    STAGES,
    STEP_STRATEGIES,
    TEACHER_INPUTS,
    TEACHERS,
    TRANSFORMS,
)
from lmkd.core.validate import check_recipe
from lmkd.plugins import load_plugins
from lmkd.scaffold.probe import probe_gradient

ROOT = Path(__file__).resolve().parents[2]
PKG_INIT = ROOT / "{rel_pkg_dir}" / "__init__.py"
RECIPE_PATH = ROOT / "{rel_recipe_path}"

load_plugins([str(PKG_INIT)])


def test_{name}_plugins_registered() -> None:
    assert CRITERIONS.build("{name}") is not None
    assert TRANSFORMS.build("{name}_transform") is not None
    assert PRODUCERS.build("{name}_producer") is not None
    assert STAGES.build("{name}_stage") is not None
    assert TEACHER_INPUTS.build("{name}_teacher_input") is not None
    assert TEACHERS.build("{name}_teacher") is not None
    assert ALGORITHMS.build("{name}_algorithm") is not None
    assert STEP_STRATEGIES.build("{name}_step") is not None
    assert CALLBACKS.build("{name}_callback") is not None
    assert EVALUATORS.build({{"type": "{name}_eval", "data": {{"type": "jsonl", "path": "dummy.jsonl"}}}}) is not None


def test_{name}_loss_gradient() -> None:
    crit = CRITERIONS.build("{name}")
    probe_gradient(crit)


def test_{name}_recipe_check() -> None:
    cfg = load_recipe(RECIPE_PATH)
    rep = check_recipe(cfg)
    assert rep.ok, rep.render()
'''


def _render_minimal_init_py(name: str, cls_name: str) -> str:
    return f'''"""Minimal {cls_name} method package for LM-KD (loss-only scaffold)."""

from __future__ import annotations

from pathlib import Path

from lmkd.plugins import load_plugins

if __package__:
    from .loss import {cls_name}Loss

    __all__ = ["{cls_name}Loss"]
else:
    _HERE = Path(__file__).resolve().parent
    load_plugins(["loss.py"], base_dir=_HERE)
'''


def _render_minimal_recipe_yaml(name: str, import_prefix: str = "") -> str:
    return f"""name: {name}
imports: [{import_prefix}loss.py]

engine:
  type: local
  device: cpu
  mixed_precision: "no"

models:
  student:
    path: sshleifer/tiny-gpt2
  # Uncomment below to enable a teacher model:
  # teacher:
  #   path: sshleifer/tiny-gpt2

data:
  train:
    type: jsonl
    path: examples/data/train_sample.jsonl
  transforms:
    - type: tokenize_sft
      use_chat_template: false
  max_length: 256

teacher:
  provider: none  # change to 'hf_frozen' when adding models.teacher
  input: same

loss:
  aggregator: weighted_sum
  terms:
    - criterion: {name}
      weight: 1.0
      params:
        temperature: 1.0
        alpha: 0.5

optim:
  lr: 2e-5
  weight_decay: 0.0
  optimizers:
    default:
      type: adamw
  schedulers:
    - type: constant

grad:
  clip_norm: 1.0

train:
  output_dir: outputs/{name}
  max_steps: 2
  batch_size: 2
  micro_batch_size: 1
  seed: 42
  log_every: 1
"""


def _render_minimal_readme_md(name: str, cls_name: str, rel_pkg_dir: str, rel_recipe_file: str) -> str:
    return f"""# {cls_name} (`{name}`)

Minimal loss-only package generated by `lmkd init {name} --minimal`.
Everything besides the criterion (`loss.py`) uses canonical LM-KD framework defaults.

## Quick Commands

```bash
# 1. Fast synthetic gradient probe on the loss
lmkd try {name}

# 2. Validate recipe configuration and plugin graph
lmkd check -c {rel_pkg_dir}/recipe.yaml
lmkd check -c {rel_recipe_file}

# 3. Train
lmkd train -c {rel_pkg_dir}/recipe.yaml

# 4. Run unit tests for this method
pytest -q tests/unit/test_{name}.py
```
"""


def _render_minimal_test_py(name: str, rel_pkg_dir: str, rel_recipe_path: str) -> str:
    return f'''"""Unit and smoke tests for minimal initialized method '{name}'."""

from __future__ import annotations

from pathlib import Path

from lmkd.core.config import load_recipe
from lmkd.core.registry import CRITERIONS
from lmkd.core.validate import check_recipe
from lmkd.plugins import load_plugins
from lmkd.scaffold.probe import probe_gradient

ROOT = Path(__file__).resolve().parents[2]
PKG_INIT = ROOT / "{rel_pkg_dir}" / "__init__.py"
RECIPE_PATH = ROOT / "{rel_recipe_path}"

load_plugins([str(PKG_INIT)])


def test_{name}_criterion_registered() -> None:
    assert CRITERIONS.build("{name}") is not None


def test_{name}_loss_gradient() -> None:
    crit = CRITERIONS.build("{name}")
    probe_gradient(crit)


def test_{name}_recipe_check() -> None:
    cfg = load_recipe(RECIPE_PATH)
    rep = check_recipe(cfg)
    assert rep.ok, rep.render()
'''


def _compute_paths(
    root: Path,
    clean_name: str,
    norm_target: str,
    domain: str | None,
) -> tuple[Path, Path, Path]:
    if norm_target == "baseline":
        base_dir = root / "src" / "baselines"
        pkg_dir = (base_dir / domain / clean_name) if domain else (base_dir / clean_name)
        recipe_domain = domain or "baselines"
    else:
        base_dir = root / "dev"
        pkg_dir = (base_dir / domain / clean_name) if domain else (base_dir / clean_name)
        recipe_domain = domain or "dev"

    central_recipe = root / "recipes" / recipe_domain / f"{clean_name}.yaml"
    test_file = root / "tests" / "unit" / f"test_{clean_name}.py"
    return pkg_dir, central_recipe, test_file


def _ensure_parent_packages(pkg_dir: Path, stop_at: Path) -> None:
    cur = pkg_dir.parent
    stop_resolved = stop_at.resolve()
    while cur.resolve() != stop_resolved and stop_resolved in cur.resolve().parents:
        init_f = cur / "__init__.py"
        if not init_f.exists():
            init_f.touch()
        cur = cur.parent


def create_baseline_template(
    name: str | None = None,
    *,
    target: str = "dev",
    domain: str | None = None,
    force: bool = False,
    minimal: bool = False,
    root: Path | None = None,
) -> int:
    """Create a ready-to-override baseline/method directory template (full or ``minimal=True``)."""
    if not name or not name.strip():
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                name = input("Enter new baseline/method name: ").strip()
            except EOFError:
                name = ""
        if not name:
            sys.stderr.write("Error: baseline/method name is required (e.g. `lmkd init my_method`).\n")
            return 1

    try:
        clean_name, norm_target, resolved_domain = parse_init_target(name, target=target, domain=domain)
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    proj_root = _resolve_root(root)
    pkg_dir, central_recipe, test_file = _compute_paths(proj_root, clean_name, norm_target, resolved_domain)
    cls_name = _to_class_name(clean_name)

    rel_pkg_from_recipe = os.path.relpath(pkg_dir, central_recipe.parent).replace("\\", "/")
    rel_pkg_dir = pkg_dir.relative_to(proj_root).as_posix()
    rel_central_recipe = central_recipe.relative_to(proj_root).as_posix()

    if minimal:
        files_to_write: dict[Path, str] = {
            pkg_dir / "__init__.py": _render_minimal_init_py(clean_name, cls_name),
            pkg_dir / "loss.py": _render_loss_py(clean_name, cls_name),
            pkg_dir / "recipe.yaml": _render_minimal_recipe_yaml(clean_name, import_prefix=""),
            central_recipe: _render_minimal_recipe_yaml(clean_name, import_prefix=f"{rel_pkg_from_recipe}/"),
            pkg_dir / "README.md": _render_minimal_readme_md(clean_name, cls_name, rel_pkg_dir, rel_central_recipe),
            test_file: _render_minimal_test_py(clean_name, rel_pkg_dir, rel_central_recipe),
        }
    else:
        files_to_write = {
            pkg_dir / "__init__.py": _render_init_py(clean_name, cls_name),
            pkg_dir / "config.py": _render_config_py(clean_name, cls_name),
            pkg_dir / "loss.py": _render_loss_py(clean_name, cls_name),
            pkg_dir / "data.py": _render_data_py(clean_name, cls_name),
            pkg_dir / "teacher.py": _render_teacher_py(clean_name, cls_name),
            pkg_dir / "train.py": _render_train_py(clean_name, cls_name),
            pkg_dir / "eval.py": _render_eval_py(clean_name, cls_name),
            pkg_dir / "recipe.yaml": _render_recipe_yaml(clean_name, import_prefix=""),
            central_recipe: _render_recipe_yaml(clean_name, import_prefix=f"{rel_pkg_from_recipe}/"),
            pkg_dir / "README.md": _render_readme_md(clean_name, cls_name, rel_pkg_dir, rel_central_recipe),
            test_file: _render_test_py(clean_name, rel_pkg_dir, rel_central_recipe),
        }

    if not force:
        existing = [p for p in files_to_write if p.exists()]
        if existing:
            sys.stderr.write(f"File already exists: {existing[0]} (use --force to overwrite)\n")
            return 1

    for path, content in files_to_write.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    if norm_target == "baseline":
        _ensure_parent_packages(pkg_dir, proj_root / "src")
    else:
        _ensure_parent_packages(pkg_dir, proj_root)

    mode_label = "minimal method" if minimal else "baseline"
    file_list = "__init__.py, loss.py" if minimal else "__init__.py, config.py, loss.py, data.py, teacher.py, train.py, eval.py"
    print(f"Initialized {mode_label} template '{clean_name}' in {pkg_dir}")
    print(f"  - Package  : {rel_pkg_dir}/ ({file_list})")
    print(f"  - Recipes  : {rel_pkg_dir}/recipe.yaml, {rel_central_recipe}")
    print(f"  - Unit test: {test_file.relative_to(proj_root).as_posix()}")
    return 0
