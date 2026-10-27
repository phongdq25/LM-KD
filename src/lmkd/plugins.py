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
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    _USER_LOADED.add(key)


def ensure_plugin(registry: str, name: str) -> None:
    """Import the specific module defining ``name`` in ``registry`` (lazy loading).

    If not in the static index, falls back to loading all built-in plugins.
    """
    from lmkd.core.plugin_index import lookup

    mod = lookup(registry, name)
    if mod is not None:
        try:
            importlib.import_module(mod)
            return
        except Exception:
            pass
    load_builtin_plugins()


def load_plugins(items: Iterable[str] | None, base_dir: str | Path | None = None) -> None:
    """Import user plugin files/modules (idempotent). Relative ``.py`` paths are tried against
    ``base_dir`` (the recipe's folder) first, then the current directory."""
    for item in items or ():
        if item.endswith(".py") or "/" in item or "\\" in item:
            p = Path(item)
            candidates = [p] if p.is_absolute() else ([Path(base_dir) / p] if base_dir else []) + [p]
            found = next((c for c in candidates if c.is_file()), None)
            if found is None:
                raise FileNotFoundError(f"plugin file {item!r} not found (tried {[str(c) for c in candidates]})")
            _import_file(found)
        else:
            if item not in _USER_LOADED:
                importlib.import_module(item)
                _USER_LOADED.add(item)
