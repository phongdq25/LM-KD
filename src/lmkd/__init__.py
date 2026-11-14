"""LM-KD: unified plugin-based LLM distillation framework.

Users: README.md (CLI + Python API). See official_docs/ for full technical documentation.

Python API (imported lazily, ``import lmkd`` stays cheap)::

    lmkd.build(recipe, ...) -> Runtime      lmkd.run(recipe, ...) -> Runtime (= lmkd train)
    lmkd.quick(student, data, loss=...)     lmkd.load_plugins([...])
    @lmkd.loss(name, ...)                   # Level 1: target-aligned functional loss (recommended)
    class MyLoss(lmkd.Criterion): ...       # Level 2: class-based criterion (+ aux modules)
    @lmkd.register_loss / @lmkd.compute_loss  # HF-positional compat for raw Trainer ports
"""

__version__ = "0.1.0.dev0"

import warnings

# Suppress upstream framework deprecations that originate outside LM-KD
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*argument 'device' of Tensor\.(pin_memory|is_pinned)\(\) is deprecated.*",
)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*torch\.jit\.script_method.*")

_EXPORTS = {
    "build": "lmkd.api",
    "run": "lmkd.api",
    "quick": "lmkd.api",
    "load_recipe": "lmkd.core.config",
    "load_plugins": "lmkd.plugins",
    # Level 1: @lmkd.loss API
    "loss": "lmkd.losses.loss_decorator",
    "TokenLoss": "lmkd.losses.loss_decorator",
    "SeqLoss": "lmkd.losses.loss_decorator",
    "ScalarLoss": "lmkd.losses.loss_decorator",
    "Inputs": "lmkd.losses.loss_decorator",
    # Level 2: Criterion contract & feature declarations
    "Criterion": "lmkd.losses.base",
    "LossContext": "lmkd.losses.base",
    "LossOutput": "lmkd.losses.base",
    "LossTerm": "lmkd.losses.base",
    "ModelPairInfo": "lmkd.losses.base",
    "PrepareContext": "lmkd.losses.base",
    "FeatureRequest": "lmkd.models.forward",
    "SideRequest": "lmkd.models.forward",
    "SideOutput": "lmkd.models.forward",
    "TeacherSignalSpec": "lmkd.teachers.signals",
    "TeacherOutput": "lmkd.teachers.signals",
    # Core data, plugin & registry primitives
    "Batch": "lmkd.core.batch",
    "FieldSpec": "lmkd.core.batch",
    "FIELDS": "lmkd.core.batch",
    "BaseConfig": "lmkd.core.plugin",
    "Plugin": "lmkd.core.plugin",
    "CRITERIONS": "lmkd.core.registry",
    "TRANSFORMS": "lmkd.core.registry",
    "ALIGNERS": "lmkd.core.registry",
    "PRODUCERS": "lmkd.core.registry",
    "TEACHERS": "lmkd.core.registry",
    "TEACHER_INPUTS": "lmkd.core.registry",
    "ALGORITHMS": "lmkd.core.registry",
    "STEP_STRATEGIES": "lmkd.core.registry",
    "CALLBACKS": "lmkd.core.registry",
    "EVALUATORS": "lmkd.core.registry",
    "STAGES": "lmkd.core.registry",
    # Legacy HF-positional compat helpers (for porting raw Trainer scripts verbatim)
    "register_loss": "lmkd.losses.hf_compat",
    "compute_loss": "lmkd.losses.hf_compat",
    "ModelView": "lmkd.losses.hf_compat",
    "Runtime": "lmkd.train.runtime",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'lmkd' has no attribute {name!r}")
