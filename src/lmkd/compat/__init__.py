"""Version-sensitive glue to third-party libraries, in ONE place.

Rule: code outside this package must not touch a private or fast-moving API of torch, transformers,
deepspeed, vllm or peft (underscore names, internal attributes such as ``ds_id``, module layouts such
as ``llm_engine.model_executor``). It calls the small, named functions here instead. When a pinned
library is upgraded, fix THIS package and re-run ``tests/unit/test_compat_canary.py``: the canary
tests assert every assumption (the symbol exists, has the shape we rely on, behaves like our own
reference), so a bad upgrade fails there with a precise message rather than deep inside a training run.

    hf_api      transformers: rotary helper used by last-row attention
    ds_api      deepspeed: ZeRO-3 parameter detection, engine step bookkeeping, gathered parameters
    vllm_api    vllm: engine construction, prompts, LoRA requests, in-process model access
    trl_api     trl: VLLMClient for the vllm_server rollout backend
"""
from __future__ import annotations

from importlib import metadata

# Libraries whose version is pinned in pyproject.toml (checked by the canary tests).
PINNED_LIBRARIES = ("torch", "transformers", "deepspeed", "vllm", "peft", "trl")


def installed_versions() -> dict[str, str | None]:
    """Installed version of every pinned library (None if missing)."""
    out: dict[str, str | None] = {}
    for name in PINNED_LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def has(package: str, min_version: str | None = None) -> bool:
    """Check if an optional package is installed, optionally with minimum version."""
    try:
        ver = metadata.version(package)
    except metadata.PackageNotFoundError:
        return False
    if min_version is None:
        return True
    try:
        from packaging.version import Version

        return Version(ver) >= Version(min_version)
    except Exception:
        # Fallback split comparison if packaging is not installed
        v_parts = [int(p) for p in ver.split(".")[:3] if p.isdigit()]
        m_parts = [int(p) for p in min_version.split(".")[:3] if p.isdigit()]
        return v_parts >= m_parts


def require(package: str, feature: str | None = None, min_version: str | None = None) -> str:
    """Ensure a dependency is present; raise informative ImportError otherwise."""
    if not has(package, min_version):
        msg = f"Feature '{feature}' requires package '{package}'" if feature else f"Package '{package}' is required"
        if min_version:
            msg += f" (version >= {min_version})"
        msg += f". Please install '{package}'."
        raise ImportError(msg)
    return metadata.version(package)
