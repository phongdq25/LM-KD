"""ModuleRegistry: every learnable auxiliary module (projector, predictor head, dual potential...)
lives here so that it is optimized, checkpointed and grouped (DESIGN §6.4).

Rule: criteria never create ``nn.Parameter``/``nn.Module`` outside ``modules.add(...)``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import PROJECTORS


@dataclass
class ModuleInfo:
    """Bookkeeping of one registered aux module (optimizer group, owner, frozen, saved)."""
    name: str
    group: str
    frozen: bool
    save: bool
    owner: str


class ModuleRegistry(nn.Module):
    """Learnable aux modules (projectors, predictors) created by criteria; they get optimizer groups and are
    checkpointed by name.
    """
    def __init__(self) -> None:
        super().__init__()
        self.mods = nn.ModuleDict()
        self.info: dict[str, ModuleInfo] = {}

    @staticmethod
    def key(name: str) -> str:
        """nn.ModuleDict forbids '.', so 'alias.proj' is stored as 'alias__proj'."""
        return name.replace(".", "__")

    def add(self, name: str, module: nn.Module, *, group: str = "projector",
            init: str | Callable[[nn.Module], None] | None = None, frozen: bool = False,
            save: bool = True, owner: str = "") -> nn.Module:
        name = self.key(name)
        if name in self.mods:
            raise KeyError(f"module '{name}' already registered (owner={self.info[name].owner})")
        if isinstance(init, str):
            INITS[init](module)
        elif callable(init):
            init(module)
        if frozen:
            for p in module.parameters():
                p.requires_grad_(False)
        self.mods[name] = module
        self.info[name] = ModuleInfo(name, group, frozen, save, owner)
        return module

    def get(self, name: str) -> nn.Module:
        return self.mods[self.key(name)]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.key(name) in self.mods

    def named_params_with_group(self):
        for name, mod in self.mods.items():
            for pn, p in mod.named_parameters():
                yield f"modules.{name}.{pn}", p, self.info[name].group

    def saveable_state_dict(self) -> dict:
        # state_dict keys look like "mods.<name>.<param path>"
        return {k: v for k, v in self.state_dict().items() if self.info[k.split(".")[1]].save}


def _zeros(m: nn.Module) -> None:
    for p in m.parameters():
        nn.init.zeros_(p)


def _normal(std: float):
    def f(m: nn.Module) -> None:
        for n, p in m.named_parameters():
            if n.endswith("weight"):
                nn.init.normal_(p, std=std)
            else:
                nn.init.zeros_(p)
    return f


INITS: dict[str, Callable[[nn.Module], None]] = {
    "zeros": _zeros,
    "normal_1e-3": _normal(1e-3),
    "normal_0.02": _normal(0.02),
    # "pinv_logit_equiv" (DSKDv2) and "ridge_buffer" (IMPACT) need model weights -> task T12/T15
}


# ---------------------------------------------------------------------- example projectors
@PROJECTORS.register("linear")
class LinearProjector(nn.Module):
    """Linear projection, optionally with bias."""
    class Config(BaseConfig):
        in_dim: int
        out_dim: int
        bias: bool = False

    def __init__(self, cfg: LinearProjector.Config):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.in_dim, cfg.out_dim, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@PROJECTORS.register("mlp")
class MLPProjector(nn.Module):
    """Two-layer MLP projection with a configurable hidden size and activation."""
    class Config(BaseConfig):
        in_dim: int
        out_dim: int
        hidden_dim: int | None = None
        act: str = "gelu"
        layernorm: bool = False

    def __init__(self, cfg: MLPProjector.Config):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim or cfg.out_dim
        act = {"gelu": nn.GELU(), "relu": nn.ReLU()}[cfg.act]
        layers: list[nn.Module] = [nn.LayerNorm(cfg.in_dim)] if cfg.layernorm else []
        layers += [nn.Linear(cfg.in_dim, h), act, nn.Linear(h, cfg.out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@PROJECTORS.register("dsl", source="white-box-distillation/cross-tokenizer/IMPACT/code/distiller.py:98-139")
class DSLProjector(nn.Module):
    """IMPACT projector DSL (e.g. '2s-2t', 's-p-relu-t'). Parses structured layer specs."""

    class Config(BaseConfig):
        structure: str
        s: int
        t: int
        p: int | None = None
        bias: bool = True

    def __init__(self, cfg: DSLProjector.Config):
        super().__init__()
        self.cfg = cfg
        name_dict: dict[str, int | None] = {"s": cfg.s, "t": cfg.t, "p": cfg.p}
        raw_tokens = cfg.structure.split("-")
        parsed: list[int | str] = []
        for tok in raw_tokens:
            if tok == "relu":
                parsed.append("relu")
            else:
                base_char = tok[-1]
                if base_char not in name_dict:
                    raise ValueError(f"unknown symbol {base_char!r} in structure {cfg.structure!r}")
                base_val = name_dict[base_char]
                if base_val is None:
                    raise ValueError(f"symbol {base_char!r} used in {cfg.structure!r} but value is None")
                coef_str = tok[:-1]
                coef = int(coef_str) if coef_str else 1
                parsed.append(coef * base_val)

        layers: list[nn.Module] = []
        last_size: int | None = None
        for i in range(len(parsed) - 1):
            curr, nxt = parsed[i], parsed[i + 1]
            if isinstance(curr, int) and isinstance(nxt, int):
                layers.append(nn.Linear(curr, nxt, bias=cfg.bias))
            elif isinstance(curr, int) and nxt == "relu":
                layers.append(nn.ReLU())
                last_size = curr
            elif curr == "relu" and isinstance(nxt, int):
                if last_size is None:
                    raise ValueError(f"invalid structure starting with relu: {cfg.structure!r}")
                layers.append(nn.Linear(last_size, nxt, bias=cfg.bias))
            else:
                raise NotImplementedError(f"invalid transition in structure {cfg.structure!r}: {curr} -> {nxt}")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
