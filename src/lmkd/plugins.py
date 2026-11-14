"""Plugin loading.

* :func:`load_builtin_plugins` imports EVERY module of the ``lmkd`` package (auto-discovery):
  a new plugin file anywhere under ``src/lmkd/`` is registered without editing any list.
* :func:`load_plugins` imports user code: a ``.py`` file path (``my_loss.py``,
  ``exp/plugins/kd.py``) or a dotted module name (``mypkg.losses``). Recipes list them under
  ``imports:``; the CLI accepts ``--plugins``; third-party packages may use the ``lmkd.plugins``
  entry-point group.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import pkgutil
import sys
from collections.abc import Iterable
from pathlib import Path

# Modules that never register plugins (or must not be imported eagerly).
EXCLUDE = ("lmkd.cli", "lmkd.scaffold")

_LOADED = False
_USER_LOADED: set[str] = set()


def _discover() -> list[str]:
    import lmkd

    names = []
    for m in pkgutil.walk_packages(lmkd.__path__, "lmkd."):
        if m.name.startswith(EXCLUDE) or any(part.startswith("_") for part in m.name.split(".")):
            continue
        names.append(m.name)
    try:
        import baselines

        for m in pkgutil.walk_packages(baselines.__path__, "baselines."):
            if any(part.startswith("_") for part in m.name.split(".")):
                continue
            names.append(m.name)
    except ImportError:
        pass
    return sorted(names)


def load_builtin_plugins() -> None:
    global _LOADED
    if _LOADED:
        return
    for m in _discover():
        importlib.import_module(m)
    from lmkd.core.registry import import_entry_point_plugins

    import_entry_point_plugins()
    _LOADED = True


def _import_file(path: Path) -> None:
    path = path.resolve()
    key = str(path)
    if key in _USER_LOADED:
        return
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    mod_name = f"lmkd_user_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import plugin file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    _USER_LOADED.add(key)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _USER_LOADED.discard(key)
        sys.modules.pop(mod_name, None)
        raise


_DEV_SUFFIXES = (
    "_teacher_input",
    "_transform",
    "_producer",
    "_stage",
    "_teacher",
    "_algorithm",
    "_step",
    "_callback",
    "_eval",
    "_aligner",
)


def _strip_dev_suffix(name: str) -> str:
    for suffix in _DEV_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def discover_dev_plugin(name: str, root: Path | None = None) -> bool:
    """Auto-discover user plugins in ``dev/`` (packages or standalone ``.py`` files)."""
    base = (root or Path.cwd()) / "dev"
    if not base.is_dir():
        return False

    stem = _strip_dev_suffix(name)
    candidates: list[Path] = []
    for key in (name,) if stem == name else (name, stem):
        candidates.extend(
            [
                base / f"{key}.py",
                base / key / "__init__.py",
                *sorted(base.glob(f"*/{key}.py")),
                *sorted(base.glob(f"*/{key}/__init__.py")),
            ]
        )

    for cand in candidates:
        if cand.is_file() and not (cand.name.startswith("_") and cand.name != "__init__.py"):
            _import_file(cand)
            return True

    # Fallback: import standalone dev/*.py and dev/*/__init__.py (e.g., dev/my_fkl.py defining @lmkd.loss("custom_name"))
    for cand in [*sorted(base.glob("*.py")), *sorted(base.glob("*/__init__.py"))]:
        if cand.is_file() and (not cand.name.startswith("_") or cand.name == "__init__.py"):
            _import_file(cand)
    return False


def ensure_plugin(registry: str, name: str) -> None:
    """Import the specific module defining ``name`` in ``registry`` (lazy loading).

    If not in the static index, checks ``dev/`` workspace packages then falls back to built-in plugins.
    """
    from lmkd.core.plugin_index import lookup

    mod = lookup(registry, name)
    if mod is not None:
        try:
            importlib.import_module(mod)
            return
        except Exception:
            pass
    if discover_dev_plugin(name):
        return
    load_builtin_plugins()


def load_plugins(items: Iterable[str] | None, base_dir: str | Path | None = None) -> None:
    """Import user plugin files/directories/modules (idempotent). Relative paths are tried against
    ``base_dir`` (the recipe's folder) first, then the current directory."""
    for item in items or ():
        if item.endswith(".py") or "/" in item or "\\" in item or Path(item).is_dir():
            p = Path(item)
            candidates = [p] if p.is_absolute() else ([Path(base_dir) / p] if base_dir else []) + [p]
            found = None
            for c in candidates:
                if c.is_file():
                    found = c
                    break
                if c.is_dir() and (c / "__init__.py").is_file():
                    found = c / "__init__.py"
                    break
            if found is None:
                raise FileNotFoundError(f"plugin file {item!r} not found (tried {[str(c) for c in candidates]})")
            _import_file(found)
        else:
            if item not in _USER_LOADED:
                importlib.import_module(item)
                _USER_LOADED.add(item)
