"""Base classes shared by every plugin.

Contract (DESIGN §15.1):
  * Every plugin class has a nested ``Config`` (subclass of :class:`BaseConfig`).
  * The constructor signature is always ``__init__(self, cfg: Config)``.
  * Runtime dependencies (models, tokenizers, engine, ...) are injected later through
    ``setup(rt)`` or through the context objects passed to the plugin's main method,
    never through the constructor.
  * Plugins that read/write batch keys declare them in ``requires`` / ``provides``.
"""
from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Strict config: unknown keys raise a validation error (no silent filtering)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=False)


class Plugin:
    """Mixin for non-``nn.Module`` plugins."""

    class Config(BaseConfig):
        pass

    registry_name: ClassVar[str] = ""   # set by Registry.register
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        """Keys provided for a given config (override when it depends on cfg)."""
        return tuple(cls.provides)

    @classmethod
    def required_keys(cls, cfg: Any) -> tuple[str, ...]:
        return tuple(cls.requires)

    # Typed Any on purpose: every plugin declares its OWN nested ``class Config``, which a type checker
    # cannot connect to ``self.cfg`` without making every plugin generic. Pydantic validates it at build.
    cfg: Any

    def __init__(self, cfg: BaseConfig | None = None):
        self.cfg = cfg if cfg is not None else self.Config()

    def setup(self, rt: Any) -> None:  # noqa: D401 - optional hook
        """Called once by the builder after the runtime exists. Override if needed."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore plugin state from state_dict. Override if needed."""


    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.cfg!r})"
