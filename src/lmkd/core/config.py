"""Recipe loading: YAML + ``_base_`` inheritance + dotted overrides + ``${...}`` interpolation,
then validation into :class:`RecipeConfig` (strict top-level schema).

Plugin specs inside sections stay plain dicts ``{"type": ..., ...}``; they are validated by the
plugin's own ``Config`` when built (see :mod:`lmkd.core.registry`).
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from lmkd.core.plugin import BaseConfig

# ====================================================================== raw loading


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads '1e-4' (no dot) as float, like YAML 1.2."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def deep_merge(base: Any, override: Any) -> Any:
    """Dicts merge recursively; any other type (incl. lists) is replaced. ``_delete_: true``
    inside an override dict replaces the base dict entirely."""
    if isinstance(base, dict) and isinstance(override, dict):
        if override.get("_delete_"):
            return {k: copy.deepcopy(v) for k, v in override.items() if k != "_delete_"}
        out = dict(base)
        for k, v in override.items():
            out[k] = deep_merge(base[k], v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(override)


def load_yaml_with_bases(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"cyclic _base_ chain: {[str(p) for p in _seen + (path,)]}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_Loader) or {}
    bases = cfg.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    if "imports" in cfg:
        cfg["imports"] = [_resolve_import(i, path.parent) for i in (cfg["imports"] or [])]
    merged: dict = {}
    for b in bases:
        merged = deep_merge(merged, load_yaml_with_bases(path.parent / b, _seen + (path,)))
    imports = list(dict.fromkeys([*merged.get("imports", []), *cfg.get("imports", [])]))  # union, not replace
    out = deep_merge(merged, cfg)
    if imports:
        out["imports"] = imports
    return out


def _resolve_import(item: str, folder: Path) -> str:
    """Plugin file paths in ``imports:`` are relative to the YAML file that lists them."""
    if item.endswith(".py") and not Path(item).is_absolute() and (folder / item).is_file():
        return str((folder / item).resolve())
    return item


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur: Any = cfg
    for i, k in enumerate(keys[:-1]):
        nxt_is_index = keys[i + 1].lstrip("-").isdigit()
        if isinstance(cur, list):
            cur = cur[int(k)]
            continue
        if k not in cur or cur[k] is None:
            cur[k] = [] if nxt_is_index else {}
        cur = cur[k]
    last = keys[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def _del_dotted(cfg: dict, dotted: str) -> None:
    keys = dotted.split(".")
    cur: Any = cfg
    for k in keys[:-1]:
        if isinstance(cur, list):
            cur = cur[int(k)]
        elif isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return
    last = keys[-1]
    if isinstance(cur, list):
        idx = int(last)
        if 0 <= idx < len(cur):
            del cur[idx]
    elif isinstance(cur, dict):
        cur.pop(last, None)


def _append_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur: Any = cfg
    for i, k in enumerate(keys[:-1]):
        nxt_is_index = keys[i + 1].lstrip("-").isdigit()
        if isinstance(cur, list):
            cur = cur[int(k)]
            continue
        if k not in cur or cur[k] is None:
            cur[k] = [] if nxt_is_index else {}
        cur = cur[k]
    last = keys[-1]
    if isinstance(cur, list):
        target = cur[int(last)]
    elif isinstance(cur, dict):
        if last not in cur or cur[last] is None:
            cur[last] = []
        target = cur[last]
    else:
        raise ValueError(f"cannot append to '{dotted}': parent is not a container")

    if not isinstance(target, list):
        raise ValueError(f"cannot append to '{dotted}': target is not a list (got {type(target).__name__})")

    if isinstance(value, list):
        target.extend(value)
    else:
        target.append(value)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.0.c=value``, ``a.b+=[...]`` or ``a.b~`` overrides. Values are parsed as YAML (``1e-4`` -> float)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        item = item.strip()
        if item.endswith("~"):
            _del_dotted(cfg, item[:-1].strip())
        elif "=" in item:
            key, raw = item.split("=", 1)
            key = key.strip()
            if key.endswith("+"):
                _append_dotted(cfg, key[:-1].strip(), parse_scalar(raw))
            else:
                _set_dotted(cfg, key, parse_scalar(raw))
        else:
            raise ValueError(f"override must look like key=value or key~, got {item!r}")
    return cfg


_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)$")


def parse_scalar(raw: str) -> Any:
    """YAML parse, plus '1e-4' -> float (PyYAML's YAML 1.1 reads it as a string)."""
    s = raw.strip()
    if s.lower() in ("no", "yes"):
        return s.lower()
    v = yaml.load(raw, Loader=_Loader)
    if isinstance(v, str) and _FLOAT_RE.match(v.strip()):
        return float(v)
    return v


_INTERP = re.compile(r"\$\{([^}]+)\}")


def _lookup(cfg: dict, dotted: str, at_path: str = "") -> Any:
    if dotted.startswith("env:"):
        rest = dotted[4:]
        if "," in rest:
            var_name, default_raw = rest.split(",", 1)
            if var_name in os.environ:
                return os.environ[var_name]
            return parse_scalar(default_raw)
        var_name = rest
        if var_name in os.environ:
            return os.environ[var_name]
        raise KeyError(f"interpolation ${{env:{var_name}}}: environment variable not set")

    at_str = f" at '{at_path}'" if at_path else ""
    cur: Any = cfg
    for k in dotted.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(k)]
            except (IndexError, ValueError):
                raise KeyError(f"interpolation ${{{dotted}}}{at_str}: key '{k}' not found") from None
        elif isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            raise KeyError(f"interpolation ${{{dotted}}}{at_str}: key '{k}' not found")
    return cur


def resolve_interpolations(cfg: dict, max_depth: int = 10) -> dict:
    """Resolve ``${a.b}`` and ``${env:NAME}``. A string that is exactly ``${x}`` takes the referenced object;
    otherwise the reference is substituted as text."""

    def walk(node: Any, root: dict, path: str = "") -> Any:
        if isinstance(node, dict):
            return {k: walk(v, root, f"{path}.{k}" if path else str(k)) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, root, f"{path}.{i}" if path else str(i)) for i, v in enumerate(node)]
        if isinstance(node, str):
            m = _INTERP.fullmatch(node.strip())
            if m:
                return copy.deepcopy(_lookup(root, m.group(1), path))
            return _INTERP.sub(lambda mm: str(_lookup(root, mm.group(1), path)), node)
        return node

    out = cfg
    for _ in range(max_depth):
        new = walk(out, out, "")
        if new == out:
            return new
        out = new
    raise ValueError("interpolation did not converge (cycle?)")


# ====================================================================== schema


class ModelSpec(BaseConfig):
    """One model of the recipe (``models.student`` / ``models.teacher``): path, dtype, task, LoRA, placement,
    attention implementation.
    """
    path: str
    dtype: str = "bf16"                     # bf16 | fp16 | fp32
    task: str = "causal_lm"                 # causal_lm | encoder
    tokenizer: str | None = None            # default = path
    adapter: str | None = None              # ModelAdapter name, None = auto
    gradient_checkpointing: bool = False
    trainable: bool = True                  # student only: False freezes all model weights (train aux modules only)
    lora: dict | None = None                # PEFT LoraConfig kwargs
    placement: str = "same_device"          # same_device | cpu | device:N | shard
    trust_remote_code: bool = False
    attn_implementation: str | None = None  # sdpa | flash_attention_2 | eager


class ModelsSection(BaseConfig):
    """``models:`` section: the student and an optional teacher."""
    student: ModelSpec
    teacher: ModelSpec | None = None


class DataSection(BaseConfig):
    """``data:`` section: training source (or none when a pipeline / injected records provide it), transforms,
    length and padding.
    """
    train: dict | None = None                # source spec {"type": "jsonl", "path": ...}; None = pipeline / injected records
    eval: dict | None = None
    transforms: list[dict | str] = Field(default_factory=list)  # in-memory chain if no pipeline
    max_length: int = 1024
    padding: str = "dynamic"                 # dynamic | max_length
    pad_to_multiple_of: int | None = 8
    packing: bool = False
    num_workers: int = 0
    pin_memory: bool = True


class TeacherSection(BaseConfig):
    """``teacher:`` section: provider (who the teacher is), input builder (what it sees), optional signal override."""
    provider: dict | str = "none"
    input: dict | str = "same"
    signal: dict | None = None               # override TeacherSignalSpec


class LossTermSpec(BaseConfig):
    """One entry of ``loss.terms``: criterion name, weight, optional alias and its parameters."""
    criterion: str
    weight: float = 1.0
    alias: str | None = None
    params: dict = Field(default_factory=dict)


class LossSection(BaseConfig):
    """``loss:`` section: aggregator, terms and parameter schedules."""
    aggregator: dict | str = "weighted_sum"
    terms: list[LossTermSpec] = Field(default_factory=list)   # may be empty with a loss function (algorithm compute_loss)
    schedules: dict[str, dict] = Field(default_factory=dict)


class OptimSection(BaseConfig):
    """``optim:`` section: parameter groups (match rules), optimizers by name, default lr / weight decay, LR
    schedulers.
    """
    groups: list[dict] = Field(default_factory=lambda: [{"name": "default", "match": {}}])
    optimizers: dict[str, dict] = Field(default_factory=lambda: {"default": {"type": "adamw"}})
    lr: float = 2e-5
    weight_decay: float = 0.0
    schedulers: list[dict] = Field(default_factory=lambda: [{"type": "constant"}])


class GradSection(BaseConfig):
    """``grad:`` section: global clip norm, activation-gradient hooks and parameter-gradient transforms."""
    clip_norm: float | None = 1.0
    hooks: list[dict] = Field(default_factory=list)
    transforms: list[dict] = Field(default_factory=list)


class TrainSection(BaseConfig):
    """``train:`` section: steps/epochs, global and micro batch size, seed, logging/eval/save cadence, resume."""
    output_dir: str = "outputs/run"
    max_steps: int | None = None
    epochs: float | None = 1.0
    batch_size: int = Field(default=8, gt=0)
    micro_batch_size: int = Field(default=8, gt=0)
    seed: int = 42
    log_every: int = 10
    eval_every: int | str | None = None      # int steps | "epoch" | None
    save_every: int | str | None = None
    save: dict = Field(default_factory=dict)
    resume_from: str | None = None

    @model_validator(mode="after")
    def _validate_batch_sizes(self) -> TrainSection:
        if self.micro_batch_size > self.batch_size:
            raise ValueError(f"micro_batch_size ({self.micro_batch_size}) cannot exceed batch_size ({self.batch_size})")
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be divisible by micro_batch_size ({self.micro_batch_size})")
        return self


class RecipeConfig(BaseConfig):
    """Top-level recipe schema (strict: unknown keys are errors). Plugin specs inside stay dicts and are
    validated by each plugin's own Config.
    """
    name: str | None = None
    imports: list[str] = Field(default_factory=list)   # user plugin files (.py, relative to the YAML) or modules
    models: ModelsSection
    data: DataSection = Field(default_factory=DataSection)
    pipeline: list[dict] = Field(default_factory=list)
    artifacts_dir: str | None = None
    teacher: TeacherSection = Field(default_factory=TeacherSection)
    loss: LossSection = Field(default_factory=LossSection)
    producer: dict | str = "dataset"
    algorithm: dict | str = "standard"
    step: dict | str = "standard"
    optim: OptimSection = Field(default_factory=OptimSection)
    grad: GradSection = Field(default_factory=GradSection)
    rollout: dict | None = None
    engine: dict | str = "local"
    callbacks: list[dict | str] = Field(default_factory=list)
    eval: list[dict] = Field(default_factory=list)
    train: TrainSection = Field(default_factory=TrainSection)
    compat: dict = Field(default_factory=dict)


def load_recipe(path: str | Path, overrides: list[str] | None = None) -> RecipeConfig:
    raw = load_yaml_with_bases(path)
    raw = apply_overrides(raw, overrides)
    raw = resolve_interpolations(raw)
    return RecipeConfig(**raw)


def recipe_from_dict(raw: dict, overrides: list[str] | None = None) -> RecipeConfig:
    raw = resolve_interpolations(apply_overrides(raw, overrides))
    return RecipeConfig(**raw)
