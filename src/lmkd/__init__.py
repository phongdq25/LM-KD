"""LM-KD: unified plugin-based LLM distillation framework.

Users: README.md (CLI + Python API). Contributors: docs/CONTRIBUTING.md, docs/DESIGN.md.

Python API (imported lazily, ``import lmkd`` stays cheap)::

    lmkd.build(recipe, ...) -> Runtime      lmkd.run(recipe, ...) -> Runtime (= lmkd train)
    lmkd.load_recipe(path, overrides)       lmkd.load_plugins([...])
    @lmkd.register_loss(name, ...)          # tier-1 loss function -> criterion
    @lmkd.compute_loss(name)                # tier-0 loss function -> algorithm compute_loss
"""

__version__ = "0.1.0.dev0"

import warnings

# Suppress upstream framework deprecations that originate outside LM-KD
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*argument 'device' of Tensor\.(pin_memory|is_pinned)\(\) is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*torch\.jit\.script_method.*")

_EXPORTS = {
    "build": "lmkd.api",
    "run": "lmkd.api",
    "quick": "lmkd.api",
    "load_recipe": "lmkd.core.config",
    "load_plugins": "lmkd.plugins",
    "loss": "lmkd.losses.unified",
    "TokenLoss": "lmkd.losses.unified",
    "SeqLoss": "lmkd.losses.unified",
    "ScalarLoss": "lmkd.losses.unified",
    "Inputs": "lmkd.losses.unified",
    "register_loss": "lmkd.losses.simple",
    "compute_loss": "lmkd.losses.simple",
    "ModelView": "lmkd.losses.simple",
    "Runtime": "lmkd.train.runtime",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'lmkd' has no attribute {name!r}")
