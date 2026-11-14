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
    """Every auto-discoverable module -> its file (skips ``_private`` parts and :data:`EXCLUDE`)."""
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
    try:
        import baselines

        b_root = Path(baselines.__path__[0])
        for path in sorted(b_root.rglob("*.py")):
            rel = path.relative_to(b_root).with_suffix("")
            parts = list(rel.parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            name = ".".join(["baselines", *parts])
            if not parts or any(p.startswith("_") for p in parts):
                continue
            out[name] = path
    except ImportError:
        pass
    return out


def _reg_name(node: ast.AST, regs: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return regs.get(node.id)
    if isinstance(node, ast.Attribute):
        return regs.get(node.attr)
    return None


_LOSS_DECORATORS = frozenset({"loss", "register_loss", "compute_loss"})


def _literal(
    args: list[ast.expr], idx: int, keywords: list[ast.keyword] | None = None, kw_name: str = "name"
) -> str | None:
    if len(args) > idx:
        arg = args[idx]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    if keywords:
        for kw in keywords:
            if kw.arg == kw_name and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
    return None


def _call_reg_and_name(call: ast.Call, regs: dict[str, str]) -> tuple[str | None, str | None, bool]:
    """Return (registry_name, explicit_plugin_name, allows_default_name) for a Call AST node."""
    fn = call.func
    if isinstance(fn, ast.Attribute) and fn.attr == "register":
        return _reg_name(fn.value, regs), _literal(call.args, 0, call.keywords), len(call.args) == 0
    if isinstance(fn, ast.Name) and fn.id == "stub" and call.args:
        return _reg_name(call.args[0], regs), _literal(call.args, 1, call.keywords), False
    if isinstance(fn, ast.Name) and fn.id in _LOSS_DECORATORS:
        return "criterions", _literal(call.args, 0, call.keywords), len(call.args) == 0
    if isinstance(fn, ast.Attribute) and fn.attr in _LOSS_DECORATORS:
        return "criterions", _literal(call.args, 0, call.keywords), len(call.args) == 0
    return None, None, False


def scan_source(source: str, regs: dict[str, str]) -> list[tuple[str, str]]:
    """``[(registry, name)]`` registered in ``source``."""
    found: list[tuple[str, str]] = []
    tree = ast.parse(source)
    decorated_calls: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    decorated_calls.add(id(dec))
                    reg, name, allow_default = _call_reg_and_name(dec, regs)
                    if reg is not None:
                        resolved = name if name is not None else (node.name.lower() if allow_default else None)
                        if resolved:
                            found.append((reg, resolved))
                elif isinstance(dec, ast.Attribute) and dec.attr == "register":
                    reg = _reg_name(dec.value, regs)
                    if reg is not None:
                        found.append((reg, node.name.lower()))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and id(node) not in decorated_calls:
            reg, name, _ = _call_reg_and_name(node, regs)
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
        except (SyntaxError, UnicodeDecodeError):  # broken file: left to the eager fallback to report
            continue
        for reg, name in entries:
            index.setdefault(reg, {}).setdefault(name, mod)
    return index


def lookup(registry: str, name: str) -> str | None:
    """Module that registers ``name`` in ``registry`` (None if not statically known)."""
    return build_index().get(registry, {}).get(name)
