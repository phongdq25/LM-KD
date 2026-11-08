"""Maintainability guards: cheap static checks that keep the codebase readable as it grows.

They fail with the offending names, so the fix is obvious. Lint and types are separate:
``ruff check src tests examples`` and ``mypy`` (both configured in pyproject.toml).
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "lmkd"
ROOT = SRC.parents[1]

MAX_FUNCTION_LINES = 200
# Long on purpose (argument-parser table, linear preflight report); do not add to this list lightly.
LONG_FUNCTIONS_ALLOWED = {"cli.py:main", "core/validate.py:preflight", "scaffold/__init__.py:create_scaffold"}


def _modules():
    for p in sorted(SRC.rglob("*.py")):
        if any(part.startswith("_") for part in p.relative_to(SRC).parts[:-1]):
            continue
        yield p, ast.parse(p.read_text())


def test_every_public_class_has_a_docstring():
    """First docstring line is what ``lmkd list`` shows for plugins; nested ``Config`` classes are exempt."""
    missing = [
        f"{p.relative_to(SRC)}:{n.lineno} {n.name}"
        for p, tree in _modules()
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
        and not n.name.startswith("_")
        and n.name != "Config"
        and ast.get_docstring(n) is None
    ]
    assert not missing, "classes without a docstring:\n  " + "\n  ".join(missing)


def test_functions_stay_short():
    long = []
    for p, tree in _modules():
        rel = p.relative_to(SRC).as_posix()
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                length = n.end_lineno - n.lineno + 1
                if length > MAX_FUNCTION_LINES and f"{rel}:{n.name}" not in LONG_FUNCTIONS_ALLOWED:
                    long.append(f"{rel}:{n.lineno} {n.name} ({length} lines > {MAX_FUNCTION_LINES}): split it")
    assert not long, "\n".join(long)


# Private / fast-moving third-party APIs must be reached through src/lmkd/compat/ (see its docstring).
_COMPAT_ONLY = {
    r"\bds_id\b": "DeepSpeed ZeRO-3 marker -> compat.ds_api.is_partitioned_param",
    r"\bset_gradient_accumulation_boundary\b": "compat.ds_api.set_accumulation_boundary",
    r"\bwas_step_applied\b": "compat.ds_api.step_was_applied",
    r"deepspeed\.initialize\(": "compat.ds_api.initialize",
    r"\bGatheredParameters\b": "compat.ds_api.gathered_parameters",
    r"\bllm_engine\b|\bmodel_executor\b": "compat.vllm_api.in_process_model",
    r"from vllm\.|import vllm\b": "compat.vllm_api",
    r"\bapply_rotary_pos_emb\b": "compat.hf_api.rotary_apply_fn",
    r"transformers\.optimization import _": "private transformers function; write the formula (see train/schedules.py::wsd)",
    r"\btorch_dtype\b": "compat.hf_api.dtype_kwarg",
}


def test_third_party_internals_only_inside_compat():
    offenders = []
    for p in sorted(SRC.rglob("*.py")):
        rel = p.relative_to(SRC).as_posix()
        if rel.startswith("compat/"):
            continue
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            for pattern, hint in _COMPAT_ONLY.items():
                if re.search(pattern, code) and not code.lstrip().startswith(('"', "'")):
                    offenders.append(f"{rel}:{lineno}: {line.strip()[:80]}   ->  {hint}")
    assert not offenders, "use the compat layer:\n  " + "\n  ".join(offenders)


def test_plugin_modules_do_not_import_the_cli():
    bad = [
        p.relative_to(SRC).as_posix()
        for p, tree in _modules()
        if p.name != "cli.py" and any(isinstance(n, ast.ImportFrom) and n.module == "lmkd.cli" for n in ast.walk(tree))
    ]
    assert not bad, f"library code must not depend on the CLI: {bad}"


def test_lint_clean_when_ruff_is_available():
    pytest.importorskip("ruff", reason="ruff is a dev dependency (pip install -e '.[dev]')")
    r = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src", "tests", "examples", "--quiet"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout[-3000:]
