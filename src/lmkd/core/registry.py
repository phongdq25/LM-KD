"""Strict, metadata-rich plugin registry.

Usage::

    from lmkd.core.registry import CRITERIONS

    @CRITERIONS.register("my_loss", paper="arXiv:xxxx", source="baseline/file.py:10")
    class MyLoss(Criterion):
        class Config(BaseConfig):
            temperature: float = 1.0

    crit = CRITERIONS.build({"type": "my_loss", "temperature": 2.0})

Rules:
  * ``build`` validates parameters with the plugin's ``Config`` (extra keys -> error).
  * Parameters may be given flat (``{"type": x, "a": 1}``) or nested (``{"type": x, "params": {...}}``).
  * ``status`` is ``"stable" | "experimental" | "stub"``. Building a ``stub`` raises
    :class:`NotImplementedError` with a pointer to the task document, so a recipe that uses an
    unimplemented plugin fails early in ``lmkd check``.
  * Built-in plugins load lazily: ``get``/``build``/``in`` import only the defining module (static index in
    :mod:`lmkd.core.plugin_index`); ``iter``/``len`` and unknown names import every built-in module.
"""
from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError


class RegistryError(KeyError):
    """Unknown or duplicate plugin name (a ``KeyError`` with close-match suggestions)."""



@dataclass(frozen=True)
class PluginMeta:
    """Registration record of one plugin: object, paper/source references, status (stable | experimental | stub),
    task doc, extras.
    """
    name: str
    registry: str
    obj: Any
    paper: str = ""
    source: str = ""
    status: str = "stable"
    task: str = ""
    extras: tuple[str, ...] = ()
    description: str = ""


@dataclass
class Registry:
    """Strict named collection of plugins: ``register`` / ``validate`` / ``build`` with pydantic-checked Config
    (unknown parameters are errors).
    """
    name: str
    _items: dict[str, PluginMeta] = field(default_factory=dict)

    # ------------------------------------------------------------------ registration
    def register(
        self,
        name: str | None = None,
        *,
        paper: str = "",
        source: str = "",
        status: str = "stable",
        task: str = "",
        extras: tuple[str, ...] = (),
        force: bool = False,
    ) -> Callable[[Any], Any]:
        if status not in {"stable", "experimental", "stub"}:
            raise ValueError(f"invalid status {status!r}")

        def deco(obj: Any) -> Any:
            key = name or obj.__name__
            cfg_cls = getattr(obj, "Config", None)
            fields = getattr(cfg_cls, "model_fields", {}) if cfg_cls is not None else {}
            reserved = {"type", "params"} & set(fields)
            if reserved:
                raise ValueError(f"[{self.name}] '{key}': Config field names {reserved} are reserved")
            if key in self._items and not force:
                raise RegistryError(f"'{key}' already registered in [{self.name}]")
            doc = (obj.__doc__ or "").strip().splitlines()
            self._items[key] = PluginMeta(
                name=key, registry=self.name, obj=obj, paper=paper, source=source,
                status=status, task=task, extras=extras, description=doc[0] if doc else "",
            )
            with contextlib.suppress(AttributeError, TypeError):   # builtins / slotted objects
                obj.registry_name = key
            return obj

        return deco

    # ------------------------------------------------------------------ lookup
    # Built-in plugins are imported lazily (lmkd.plugins): a single-name lookup imports only the module that
    # defines the name; enumerating a registry imports every built-in module first.
    def _resolve(self, key: str) -> bool:
        if key not in self._items:
            from lmkd.plugins import ensure_plugin

            ensure_plugin(self.name, key)
        return key in self._items

    def _load_all(self) -> None:
        from lmkd.plugins import load_builtin_plugins

        load_builtin_plugins()

    def __contains__(self, key: str) -> bool:
        return self._resolve(key)

    def __iter__(self) -> Iterator[str]:
        self._load_all()
        return iter(sorted(self._items))

    def __len__(self) -> int:
        self._load_all()
        return len(self._items)

    def loaded(self) -> list[str]:
        """Names registered so far, without importing anything."""
        return sorted(self._items)

    def meta(self, key: str) -> PluginMeta:
        if not self._resolve(key):
            self._load_all()   # the full list for the error message below
        if key not in self._items:
            import difflib

            close = difflib.get_close_matches(key, list(self._items), n=3)
            hint = f" Did you mean: {close}?" if close else ""
            raise RegistryError(
                f"'{key}' is not registered in [{self.name}].{hint} "
                f"Available: {sorted(self._items)}"
            )
        return self._items[key]

    def get(self, key: str) -> Any:
        return self.meta(key).obj

    # ------------------------------------------------------------------ build
    @staticmethod
    def split_spec(spec: dict | str) -> tuple[str, dict]:
        if isinstance(spec, str):
            return spec, {}
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(f"plugin spec must be a str or a dict with key 'type', got {spec!r}")
        spec = dict(spec)
        typ = spec.pop("type")
        params = spec.pop("params", {}) or {}
        overlap = set(params) & set(spec)
        if overlap:
            raise ValueError(f"keys {overlap} given both flat and under 'params' for '{typ}'")
        return typ, {**spec, **params}

    def validate(self, spec: dict | str) -> tuple[PluginMeta, Any]:
        """Validate a spec without instantiating. Returns (meta, validated Config)."""
        typ, params = self.split_spec(spec)
        meta = self.meta(typ)
        if meta.status == "stub":
            hint = f" Reference: {meta.task}." if meta.task else ""
            raise NotImplementedError(
                f"[{self.name}] '{typ}' is a stub.{hint}"
            )
        cfg_cls = getattr(meta.obj, "Config", None)
        if cfg_cls is None:
            if params:
                raise ValueError(f"'{typ}' takes no parameters, got {params}")
            return meta, None
        try:
            cfg = cfg_cls(**params)
        except ValidationError as e:
            raise ValueError(f"invalid config for [{self.name}] '{typ}':\n{e}") from e
        return meta, cfg

    def build(self, spec: dict | str) -> Any:
        meta, cfg = self.validate(spec)
        obj = meta.obj
        if cfg is None:
            return obj() if isinstance(obj, type) else obj
        return obj(cfg)


# ---------------------------------------------------------------------- global registries
SOURCES = Registry("sources")                 # raw data readers
TRANSFORMS = Registry("transforms")           # sample -> sample (offline or online)
SEGMENTERS = Registry("segmenters")           # text -> steps (Long-CoT)
ALIGNERS = Registry("aligners")               # cross-tokenizer alignment producers
STAGES = Registry("stages")                   # offline pipeline stages
PRODUCERS = Registry("producers")             # batch producers (Axis B)
ALGORITHMS = Registry("algorithms")           # augment + loss orchestration
ADAPTERS = Registry("adapters")               # model architecture adapters
POOLERS = Registry("poolers")                 # embedding poolers
TEACHERS = Registry("teachers")               # teacher providers (Axis C)
TEACHER_INPUTS = Registry("teacher_inputs")   # teacher input builders
WEIGHT_POLICIES = Registry("weight_policies") # self-teacher weight policies
CRITERIONS = Registry("criterions")           # losses (Axis D)
LOSS_FNS = Registry("loss_fns")               # tier-0 functions fn(batch, rt, **params) (algorithm compute_loss)
AGGREGATORS = Registry("aggregators")         # loss aggregators
PROJECTORS = Registry("projectors")           # learnable aux modules
LAYER_MAPPERS = Registry("layer_mappers")     # student/teacher layer pairing
REWARDS = Registry("rewards")                 # verifiers / reward functions
ADVANTAGES = Registry("advantages")           # advantage estimators
ADV_TRANSFORMS = Registry("adv_transforms")   # advantage transforms (RLSD...)
STEP_STRATEGIES = Registry("step_strategies") # step strategies (Axis E)
GRAD_HOOKS = Registry("grad_hooks")           # tensor (activation) gradient hooks
GRAD_TRANSFORMS = Registry("grad_transforms") # parameter gradient transforms
OPTIMIZERS = Registry("optimizers")           # optimizers (Axis F)
SCHEDULERS = Registry("schedulers")
CALLBACKS = Registry("callbacks")
ROLLOUT = Registry("rollout")                 # rollout engines (HF / vLLM)
EVALUATORS = Registry("evaluators")
GRADERS = Registry("graders")
ENGINES = Registry("engines")                 # local | accelerate (DeepSpeed; T05 stub)

ALL_REGISTRIES: dict[str, Registry] = {
    r.name: r
    for r in [
        ENGINES, SOURCES, TRANSFORMS, SEGMENTERS, ALIGNERS, STAGES, PRODUCERS, ALGORITHMS, ADAPTERS,
        POOLERS, TEACHERS, TEACHER_INPUTS, WEIGHT_POLICIES, CRITERIONS, LOSS_FNS, AGGREGATORS, PROJECTORS,
        LAYER_MAPPERS, REWARDS, ADVANTAGES, ADV_TRANSFORMS, STEP_STRATEGIES, GRAD_HOOKS,
        GRAD_TRANSFORMS, OPTIMIZERS, SCHEDULERS, CALLBACKS, ROLLOUT, EVALUATORS, GRADERS,
    ]
}


def import_entry_point_plugins() -> None:
    """Import third-party plugin modules declared under the ``lmkd.plugins`` entry-point group."""
    from importlib.metadata import entry_points

    for ep in entry_points(group="lmkd.plugins"):
        importlib.import_module(ep.value)
