"""Static plugin index: ``(registry, name) -> module`` found by parsing source, without importing it.

Lazy plugin loading (``lmkd.plugins``) uses this index so that building ``criterions/fkl`` imports only the module
that defines it instead of every ``lmkd.*`` module (which pulls peft, trl, datasets, ... and costs seconds).

Only literal registrations are indexed: ``@REG.register("name", ...)``, ``REG.register("name")(obj)`` and
``stub(REG, "name", ...)`` where ``REG`` is a module-level :class:`~lmkd.core.registry.Registry` constant (also as
``R.REG``). Anything else (computed names, user files) is found by the eager fallback in
:func:`lmkd.plugins.load_builtin_plugins`, so a missing index entry costs time, never correctness.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

# Modules that are never auto-imported (same rule as lmkd.plugins discovery).
EXCLUDE = ("lmkd.cli", "lmkd.scaffold")


def _registry_vars() -> dict[str, str]:
    """``{"CRITERIONS": "criterions", ...}``: module-level Registry constants of ``lmkd.core.registry``."""
    from lmkd.core import registry as R

    return {k: v.name for k, v in vars(R).items() if isinstance(v, R.Registry)}


def module_names(root: Path | None = None) -> dict[str, Path]:
    """Every auto-discoverable ``lmkd.*`` module -> its file (skips ``_private`` parts and :data:`EXCLUDE`)."""
    import lmkd

    root = root or Path(lmkd.__path__[0])
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(["lmkd", *parts])
        if not parts or name.startswith(EXCLUDE) or any(p.startswith("_") for p in parts):
            continue
        out[name] = path
    return out


def _reg_name(node: ast.AST, regs: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return regs.get(node.id)
    if isinstance(node, ast.Attribute):
        return regs.get(node.attr)
    return None


def _literal(args: list[ast.expr], idx: int) -> str | None:
    if len(args) > idx:
        arg = args[idx]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def scan_source(source: str, regs: dict[str, str]) -> list[tuple[str, str]]:
    """``[(registry, name)]`` registered with a literal name in ``source``."""
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "register":
            reg, name = _reg_name(fn.value, regs), _literal(node.args, 0)
        elif isinstance(fn, ast.Name) and fn.id == "stub" and node.args:
            reg, name = _reg_name(node.args[0], regs), _literal(node.args, 1)
        else:
            continue
        if reg is not None and name is not None:
            found.append((reg, name))
    return found


@lru_cache(maxsize=1)
def build_index() -> dict[str, dict[str, str]]:
    """``{registry: {plugin name: module}}`` for every built-in module (parsed once per process)."""
    regs = _registry_vars()
    index: dict[str, dict[str, str]] = {}
    for mod, path in module_names().items():
        try:
            entries = scan_source(path.read_text(encoding="utf-8"), regs)
        except (SyntaxError, UnicodeDecodeError):   # broken file: left to the eager fallback to report
            continue
        for reg, name in entries:
            index.setdefault(reg, {}).setdefault(name, mod)
    return index


def lookup(registry: str, name: str) -> str | None:
    """Module that registers ``name`` in ``registry`` (None if not statically known)."""
    return build_index().get(registry, {}).get(name)

