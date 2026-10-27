"""Declare not-yet-implemented plugins so that ``lmkd list`` shows the full roadmap and
``lmkd check`` fails with a pointer to the task doc.

HOW TO IMPLEMENT A STUB: delete the ``stub(...)`` call and write a real class registered under the
SAME name in the SAME module, keeping the documented Config fields (add more if needed).
"""
from __future__ import annotations

from typing import Any

from pydantic import create_model

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import Registry


def stub(
    registry: Registry,
    name: str,
    *,
    task: str,
    doc: str,
    source: str = "",
    paper: str = "",
    fields: dict[str, tuple[type, Any]] | None = None,
    base: type = Plugin,
) -> type:
    cfg = create_model(f"{name}_Config", __base__=BaseConfig, **(fields or {}))  # type: ignore[call-overload]
    cls = type(f"Stub_{name}", (base,), {"__doc__": doc, "Config": cfg, "__module__": __name__})
    registry.register(name, status="stub", task=task, source=source, paper=paper)(cls)
    return cls
