"""LM-KD: unified plugin-based LLM distillation framework.

Users: README.md (CLI + Python API). See official_docs/ for full technical documentation.

Python API (imported lazily, ``import lmkd`` stays cheap)::

    lmkd.build(recipe, ...) -> Runtime      lmkd.run(recipe, ...) -> Runtime (= lmkd train)
    lmkd.load_recipe(path, overrides)       lmkd.load_plugins([...])
    @lmkd.register_loss(name, ...)          # tier-1 loss function -> criterion
    @lmkd.compute_loss(name)                # tier-0 loss function -> algorithm compute_loss
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
    "loss": "lmkd.losses.loss_decorator",
    "TokenLoss": "lmkd.losses.loss_decorator",
    "SeqLoss": "lmkd.losses.loss_decorator",
    "ScalarLoss": "lmkd.losses.loss_decorator",
    "Inputs": "lmkd.losses.loss_decorator",
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
