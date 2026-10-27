"""Param-group rules, optimizer and scheduler factories (DESIGN §8.1).

Group rule ``match`` keys (all must hold; first matching rule wins):
  owner:       "student" | "modules"
  group:       ModuleRegistry group name (only for owner=modules)
  regex:       re.search on the parameter name
  ndim / ndim_le: exact / max tensor ndim
  layer_range: [lo, hi] inclusive on the integer captured by the FIRST group of ``layer_regex``
               (default r"\\.(?:layers|h|layer)\\.(\\d+)\\.") ; negative hi counts from the end.
Each rule may set: lr, weight_decay, optimizer (name in optim.optimizers), and any extra key.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import OPTIMIZERS, SCHEDULERS

DEFAULT_LAYER_RE = r"\.(?:layers|h|layer)\.(\d+)\."


@dataclass
class NamedParam:
    """A parameter with its name, owner (student | modules) and module-registry group, for group matching rules."""
    name: str
    param: nn.Parameter
    owner: str           # student | modules
    group: str = ""      # module registry group


@dataclass
class GroupSpec:
    """One ``optim.groups`` entry: name, match rule, options (lr, weight decay) and optimizer name."""
    name: str
    match: dict
    opts: dict = field(default_factory=dict)
    optimizer: str = "default"


def _layer_index(name: str, regex: str) -> int | None:
    m = re.search(regex, name)
    return int(m.group(1)) if m else None


def rule_matches(np_: NamedParam, match: dict, num_layers: int | None) -> bool:
    if "owner" in match and match["owner"] != np_.owner:
        return False
    if "group" in match and match["group"] != np_.group:
        return False
    if "regex" in match and not re.search(match["regex"], np_.name):
        return False
    if "ndim" in match and np_.param.ndim != match["ndim"]:
        return False
    if "ndim_le" in match and np_.param.ndim > match["ndim_le"]:
        return False
    if "layer_range" in match:
        idx = _layer_index(np_.name, match.get("layer_regex", DEFAULT_LAYER_RE))
        if idx is None:
            return False
        lo, hi = match["layer_range"]
        if hi < 0:
            if num_layers is None:
                raise ValueError("negative layer_range needs num_layers")
            hi = num_layers + hi
        if not (lo <= idx <= hi):
            return False
    known = {"owner", "group", "regex", "ndim", "ndim_le", "layer_range", "layer_regex"}
    unknown = set(match) - known
    if unknown:
        raise ValueError(f"unknown match keys {unknown}")
    return True


def build_param_groups(
    params: Iterable[NamedParam], rules: list[dict], *, default_lr: float, default_wd: float,
    num_layers: int | None = None,
) -> dict[str, list[dict]]:
    """Returns {optimizer_name: [torch param-group dicts]} ; every trainable param in exactly one group."""
    specs = []
    for r in rules:
        r = dict(r)
        name = r.pop("name")
        match = r.pop("match", {})
        opt = r.pop("optimizer", "default")
        specs.append(GroupSpec(name, match, r, opt))
    buckets: dict[str, list[NamedParam]] = {s.name: [] for s in specs}
    for np_ in params:
        if not np_.param.requires_grad:
            continue
        for s in specs:
            if rule_matches(np_, s.match, num_layers):
                buckets[s.name].append(np_)
                break
        else:
            raise ValueError(f"parameter {np_.name} matched no optim.groups rule (add a catch-all {{match: {{}}}})")
    out: dict[str, list[dict]] = {}
    for s in specs:
        if not buckets[s.name]:
            continue
        g = {"params": [p.param for p in buckets[s.name]], "names": [p.name for p in buckets[s.name]],
             "group_name": s.name, "lr": s.opts.get("lr", default_lr),
             "weight_decay": s.opts.get("weight_decay", default_wd)}
        g.update({k: v for k, v in s.opts.items() if k not in ("lr", "weight_decay")})
        g["initial_lr"] = g["lr"]
        out.setdefault(s.optimizer, []).append(g)
    return out


def talas_groups(compat: bool = False) -> list[dict]:
    """Param group rules for TALAS-M (Jordan et al. 2024 / embedding-distill).

    If compat=True, uses the unanchored substring regex 'layer\\.(0|1|2|3)' matching TALAS-M's
    original bug which inadvertently classified layer.10, layer.11, etc. as early layers.
    If compat=False (default), uses exact layer_range [0, 3].
    """
    early_match = {"regex": r"layer\.(0|1|2|3)"} if compat else {"layer_range": [0, 3]}
    return [
        {"name": "projector", "match": {"owner": "modules"}, "lr": 1.0e-4, "weight_decay": 0.01},
        {"name": "edge", "match": {"regex": r"(embeddings|pooler|classifier|head)"}},
        {"name": "norm_bias", "match": {"regex": r"(LayerNorm|bias)"}},
        {"name": "one_d", "match": {"ndim_le": 1}},
        {"name": "early", "match": early_match},
        {"name": "muon_body", "match": {"ndim": 2}, "optimizer": "muon", "lr": 1.0e-3},
        {"name": "rest", "match": {}},
    ]


# ---------------------------------------------------------------------- optimizers
class OptimizerBuilder(Plugin):
    """Builds a torch optimizer from resolved parameter groups (``build(groups)``)."""
    def build(self, groups: list[dict]) -> torch.optim.Optimizer:
        raise NotImplementedError


def _strip(groups: list[dict]) -> list[dict]:
    return [{k: v for k, v in g.items() if k != "names"} for g in groups]


@OPTIMIZERS.register("adamw")
class AdamWBuilder(OptimizerBuilder):
    """torch AdamW."""
    class Config(BaseConfig):
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        fused: bool | None = None

    def build(self, groups):
        kw: dict[str, Any] = {"betas": self.cfg.betas, "eps": self.cfg.eps}
        if self.cfg.fused is not None:
            kw["fused"] = self.cfg.fused
        return torch.optim.AdamW(_strip(groups), **kw)


@OPTIMIZERS.register("sgd")
class SGDBuilder(OptimizerBuilder):
    """torch SGD with optional momentum."""
    class Config(BaseConfig):
        momentum: float = 0.0

    def build(self, groups):
        return torch.optim.SGD(_strip(groups), momentum=self.cfg.momentum)


@OPTIMIZERS.register(
    "muon",
    paper="Muon (Jordan et al. 2024)",
    source="embedding-distill/TALAS-M-main/train_talas_m.py:38-71",
)
class MuonBuilder(OptimizerBuilder):
    """Muon optimizer for 2-D weight matrices (Jordan et al. 2024)."""

    class Config(BaseConfig):
        momentum: float = 0.95
        nesterov: bool = True
        ns_steps: int = 5
        adjust_lr_fn: str | None = None  # None | "match_rms_adamw" | "original"

    def build(self, groups):
        if not hasattr(torch.optim, "Muon"):
            raise ImportError("torch>=2.9 required for muon")
        bad = [n for g in groups for n, p in zip(g["names"], g["params"]) if p.ndim != 2]
        if bad:
            raise ValueError(f"muon only supports 2-D params; got {bad[:5]} ... route them to adamw via optim.groups")
        kw = {"momentum": self.cfg.momentum, "nesterov": self.cfg.nesterov, "ns_steps": self.cfg.ns_steps}
        if self.cfg.adjust_lr_fn:
            kw["adjust_lr_fn"] = self.cfg.adjust_lr_fn
        return torch.optim.Muon(_strip(groups), **kw)


@OPTIMIZERS.register("lion", extras=("optim",))
class LionBuilder(OptimizerBuilder):
    """Lion optimizer from ``pytorch_optimizer``."""
    class Config(BaseConfig):
        betas: tuple[float, float] = (0.9, 0.99)
        weight_decouple: bool = True

    def build(self, groups):
        from pytorch_optimizer import Lion

        return Lion(_strip(groups), betas=self.cfg.betas, weight_decouple=self.cfg.weight_decouple)


@OPTIMIZERS.register("adamw_8bit", extras=("bnb",))
class AdamW8bitBuilder(OptimizerBuilder):
    """bitsandbytes 8-bit AdamW."""
    class Config(BaseConfig):
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8

    def build(self, groups):
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(_strip(groups), betas=self.cfg.betas, eps=self.cfg.eps)


# ---------------------------------------------------------------------- schedulers
class SchedulerBuilder(Plugin):
    """Builds an LR scheduler for an optimizer given the total number of steps."""
    def build(self, optimizer: torch.optim.Optimizer, total_steps: int):
        raise NotImplementedError


@SCHEDULERS.register("constant")
class ConstantSched(SchedulerBuilder):
    """Constant LR with optional linear warmup."""
    class Config(BaseConfig):
        warmup_ratio: float = 0.0
        warmup_steps: int | None = None

    def build(self, optimizer, total_steps):
        w = self.cfg.warmup_steps if self.cfg.warmup_steps is not None else int(self.cfg.warmup_ratio * total_steps)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, w)) if w else 1.0)


@SCHEDULERS.register("cosine")
class CosineSched(SchedulerBuilder):
    """Linear warmup then cosine to ``min_lr_ratio`` x each group's OWN lr (fixes TALAS min_lr bug).

    Note on min_lr: HF cosine_with_min_lr takes absolute min_lr, causing LR to increase when min_lr > group_lr.
    Framework only takes min_lr_ratio; convert via min_lr_ratio = min_lr / lr_group.
    """

    class Config(BaseConfig):
        warmup_ratio: float = 0.0
        warmup_steps: int | None = None
        min_lr_ratio: float = 0.0

    def build(self, optimizer, total_steps):
        w = self.cfg.warmup_steps if self.cfg.warmup_steps is not None else int(self.cfg.warmup_ratio * total_steps)
        m = self.cfg.min_lr_ratio

        def f(s: int) -> float:
            if w and s < w:
                return (s + 1) / w
            p = (s - w) / max(1, total_steps - w)
            return m + (1 - m) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@SCHEDULERS.register("polynomial")
class PolynomialSched(SchedulerBuilder):
    """Polynomial decay with linear warmup (= noam DistiLLM)."""

    class Config(BaseConfig):
        warmup_ratio: float = 0.0
        warmup_steps: int | None = None
        power: float = 0.5
        min_lr_ratio: float = 0.0

    def build(self, optimizer, total_steps):
        from lmkd.train.schedules import polynomial

        w = self.cfg.warmup_steps if self.cfg.warmup_steps is not None else int(self.cfg.warmup_ratio * total_steps)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: polynomial(s, total_steps, w, self.cfg.power, self.cfg.min_lr_ratio),
        )


@SCHEDULERS.register("wsd")
class WSDSched(SchedulerBuilder):
    """Warmup-Stable-Decay schedule using transformers get_wsd_schedule."""

    class Config(BaseConfig):
        warmup_ratio: float = 0.01
        warmup_steps: int | None = None
        decay_ratio: float = 0.1
        decay_steps: int | None = None
        min_lr_ratio: float = 0.0
        decay: str = "linear"  # linear | cosine

    def build(self, optimizer, total_steps):
        from transformers import get_wsd_schedule

        w = self.cfg.warmup_steps if self.cfg.warmup_steps is not None else int(self.cfg.warmup_ratio * total_steps)
        d = self.cfg.decay_steps if self.cfg.decay_steps is not None else int(self.cfg.decay_ratio * total_steps)
        return get_wsd_schedule(
            optimizer,
            num_warmup_steps=w,
            num_decay_steps=d,
            num_training_steps=total_steps,
            decay_type=self.cfg.decay,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )


@SCHEDULERS.register("cosine_annealing")
class CosineAnnealingSched(SchedulerBuilder):
    """Cosine Annealing without warmup (= DistiLLM CosineAnnealingLR relative to group lr)."""

    class Config(BaseConfig):
        min_lr_ratio: float = 0.0

    def build(self, optimizer, total_steps):
        from lmkd.train.schedules import cosine_annealing

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: cosine_annealing(s, total_steps, self.cfg.min_lr_ratio),
        )


@SCHEDULERS.register("linear")
class LinearSched(SchedulerBuilder):
    """Linear decay with optional warmup."""
    class Config(BaseConfig):
        warmup_ratio: float = 0.0
        min_lr_ratio: float = 0.0

    def build(self, optimizer, total_steps):
        w = int(self.cfg.warmup_ratio * total_steps)
        m = self.cfg.min_lr_ratio

        def f(s: int) -> float:
            if w and s < w:
                return (s + 1) / w
            p = (s - w) / max(1, total_steps - w)
            return max(m, 1 - (1 - m) * p)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
