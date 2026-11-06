"""ModuleRegistry: every learnable auxiliary module (projector, predictor head, dual potential...)
lives here so that it is optimized, checkpointed and grouped.

Rule: criteria never create ``nn.Parameter``/``nn.Module`` outside ``modules.add(...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar

import torch
import torch.nn as nn

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import PROJECTORS

ModuleT = TypeVar("ModuleT", bound=nn.Module)
InitName: TypeAlias = Literal["zeros", "normal_1e-3", "normal_0.02"]
InitSpec: TypeAlias = InitName | str | Callable[[nn.Module], None] | None


@dataclass
class ModuleInfo:
    """Bookkeeping of one registered aux module (optimizer group, owner, frozen, saved)."""

    name: str
    group: str
    frozen: bool
    save: bool
    owner: str


class ScopedModuleView:
    """Owner-scoped view over a :class:`ModuleRegistry` supporting direct attribute and item access.

    Allows criteria to access their auxiliary modules directly as attributes (e.g. ``self.aux(ctx).projector``)
    without hardcoding ``f"{self.alias}__projector"`` or calling ``.get(...)``.
    """

    _registry: ModuleRegistry | dict[str, Any] | None
    _owner: str
    _fallback: Mapping[str, nn.Module]

    def __init__(
        self,
        registry: ModuleRegistry | dict[str, Any] | None,
        owner: str,
        fallback: Mapping[str, nn.Module] | None = None,
    ) -> None:
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_fallback", fallback if fallback is not None else {})

    def add(
        self,
        name: str,
        module: ModuleT,
        *,
        group: str = "projector",
        init: InitSpec = None,
        frozen: bool = False,
        save: bool = True,
    ) -> ModuleT:
        if not isinstance(self._registry, ModuleRegistry):
            raise RuntimeError(f"Owner '{self._owner}' cannot add module '{name}': ModuleRegistry is None")
        return self._registry.add(
            name,
            module,
            group=group,
            init=init,
            frozen=frozen,
            save=save,
            owner=self._owner,
        )

    def get(self, name: str, default: Any = None) -> Any:
        if isinstance(self._registry, ModuleRegistry):
            mod = self._registry.get(name, owner=self._owner)
            if mod is not None:
                return mod
        elif isinstance(self._registry, dict):
            key = ModuleRegistry.format_key(name, owner=self._owner)
            if key in self._registry:
                return self._registry[key]
            if name in self._registry:
                return self._registry[name]
        if name in self._fallback:
            return self._fallback[name]
        return default

    def require(self, name: str) -> Any:
        mod = self.get(name)
        if mod is not None:
            return mod
        if isinstance(self._registry, ModuleRegistry):
            avail = self._registry.keys()
        elif isinstance(self._registry, dict):
            avail = list(self._registry.keys())
        else:
            avail = list(self._fallback.keys())
        expected = ModuleRegistry.format_key(name, owner=self._owner)
        raise KeyError(
            f"Criterion '{self._owner}' could not find required module '{name}' (expected key: '{expected}'). "
            f"Available modules in registry: {avail}. "
            f"Ensure 'build_modules()' registered it, or check the module name."
        )

    def __getitem__(self, name: str) -> Any:
        return self.require(name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.require(name)
        except KeyError as exc:
            raise AttributeError(str(exc)) from exc

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.get(name) is not None


class ModuleRegistry(nn.Module):
    """Learnable aux modules (projectors, predictors) created by criteria; they get optimizer groups and are
    checkpointed by name.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mods = nn.ModuleDict()
        self.info: dict[str, ModuleInfo] = {}

    @classmethod
    def key(cls, name: str) -> str:
        """nn.ModuleDict forbids '.', so 'alias.proj' is stored as 'alias__proj'."""
        return name.replace(".", "__")

    @classmethod
    def format_key(cls, name: str, owner: str = "") -> str:
        """Format a canonical module key given a name and optional owner."""
        clean_name = name.replace(".", "__")
        if owner and "__" not in clean_name:
            clean_owner = owner.replace(".", "__")
            return f"{clean_owner}__{clean_name}"
        return clean_name

    def for_owner(self, owner: str, fallback: Mapping[str, nn.Module] | None = None) -> ScopedModuleView:
        """Return an owner-scoped view supporting direct attribute access (e.g. ``modules.for_owner('emo').projector``)."""
        return ScopedModuleView(self, owner, fallback=fallback)

    def resolve_key(self, name: str, owner: str = "") -> str | None:
        """Resolve a module name (short name, dotted name, or fully-qualified key) to a registered key.

        Returns None if not found or ambiguous.
        """
        clean_name = name.replace(".", "__")
        if owner and "__" not in clean_name:
            k = f"{owner.replace('.', '__')}__{clean_name}"
            if k in self.mods:
                return k

        k = self.key(name)
        if k in self.mods:
            return k

        matches = [m for m in self.mods if m == clean_name or m.endswith(f"__{clean_name}")]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1 and owner:
            clean_owner = owner.replace(".", "__")
            owner_matches = [m for m in matches if m.startswith(f"{clean_owner}__")]
            if len(owner_matches) == 1:
                return owner_matches[0]

        return None

    def add(
        self,
        name: str,
        module: ModuleT,
        *,
        group: str = "projector",
        init: InitSpec = None,
        frozen: bool = False,
        save: bool = True,
        owner: str = "",
    ) -> ModuleT:
        key = self.format_key(name, owner=owner)
        if key in self.mods:
            raise KeyError(
                f"module '{key}' already registered (owner={self.info[key].owner}). "
                f"Existing modules: {list(self.mods.keys())}"
            )
        if isinstance(init, str):
            INITS[init](module)
        elif callable(init):
            init(module)
        if frozen:
            for p in module.parameters():
                p.requires_grad_(False)
        self.mods[key] = module
        self.info[key] = ModuleInfo(key, group, frozen, save, owner)
        return module

    def get(self, name: str, default: Any = None, *, owner: str = "") -> nn.Module | Any:
        res = self.resolve_key(name, owner=owner)
        return self.mods[res] if res is not None else default

    def __getitem__(self, name: str) -> Any:
        res = self.resolve_key(name)
        if res is not None:
            return self.mods[res]
        avail = list(self.mods.keys())
        raise KeyError(
            f"Module '{name}' not found in ModuleRegistry. "
            f"Available modules: {avail}. "
            f"Note: modules are namespaced as '<owner>__<name>' (e.g. 'emo__projector') or short names (e.g. 'projector')."
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if not name.startswith("_") and "mods" in self._modules:
                res = self.resolve_key(name)
                if res is not None:
                    return self.mods[res]
            raise

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.resolve_key(name) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self.mods)

    def __len__(self) -> int:
        return len(self.mods)

    def keys(self) -> list[str]:
        return list(self.mods.keys())

    def named_params_with_group(self) -> Iterator[tuple[str, nn.Parameter, str]]:
        for name, mod in self.mods.items():
            for pn, p in mod.named_parameters():
                yield f"modules.{name}.{pn}", p, self.info[name].group

    def saveable_state_dict(self) -> dict[str, Any]:
        # state_dict keys look like "mods.<name>.<param path>"
        return {k: v for k, v in self.state_dict().items() if self.info[k.split(".")[1]].save}

    @staticmethod
    def resolve_ckpt_path(ckpt_path: str | Path) -> Path:
        """Resolve a checkpoint file or output directory to a ``modules.pt`` file path."""
        path = Path(ckpt_path)
        if path.is_dir():
            cand = path / "checkpoint-final" / "modules.pt"
            path = cand if cand.is_file() else path / "modules.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Module checkpoint not found: {path}")
        return path

    @classmethod
    def _read_state_dict(cls, ckpt_path: str | Path) -> tuple[Path, dict[str, Any]]:
        path = cls.resolve_ckpt_path(ckpt_path)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            raw = raw["state_dict"]
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a state_dict mapping in {path}, got {type(raw).__name__}")
        return path, raw

    def load_module_checkpoint(
        self,
        name: str,
        ckpt_path: str | Path,
        *,
        owner: str = "",
        source_name: str | None = None,
        source_owner: str | None = None,
        strict: bool = True,
        allow_raw: bool = True,
        missing_ok: bool = False,
    ) -> nn.Module | None:
        """Load weights for a registered auxiliary module from a checkpoint file or directory.

        Automatically handles ``mods.<owner>__<name>.``, ``<owner>__<name>.``, cross-alias
        ``__<short_name>.``, and flat module ``state_dict`` formats.
        """
        target_key = self.resolve_key(name, owner=owner)
        if target_key is None:
            if missing_ok:
                return None
            expected = self.format_key(name, owner=owner)
            raise KeyError(
                f"Cannot load checkpoint for unknown module '{name}' (expected key: '{expected}'). "
                f"Available modules: {list(self.mods.keys())}"
            )
        target_mod = self.mods[target_key]
        try:
            path, raw = self._read_state_dict(ckpt_path)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise

        src_owner = source_owner if source_owner is not None else owner
        src_key = self.format_key(source_name or name, owner=src_owner)
        short_name = (source_name or name).split("__")[-1].split(".")[-1]

        weights: dict[str, Any] = {}
        for prefix in (f"mods.{src_key}.", f"{src_key}."):
            weights = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
            if weights:
                break

        if not weights and source_owner is None:
            marker = f"__{short_name}."
            weights = {k.split(marker, 1)[1]: v for k, v in raw.items() if marker in k}

        if not weights and allow_raw:
            target_param_keys = set(target_mod.state_dict().keys())
            if target_param_keys and (
                target_param_keys.issubset(raw.keys()) or any(k in target_param_keys for k in raw)
            ):
                weights = {k: v for k, v in raw.items() if k in target_param_keys} if strict else dict(raw)

        if not weights:
            raise ValueError(
                f"Checkpoint {path} contains no weights for module '{src_key}' "
                f"(available keys: {list(raw.keys())})"
            )

        target_mod.load_state_dict(weights, strict=strict)
        return target_mod

    def load_owner_checkpoint(
        self,
        owner: str,
        ckpt_path: str | Path,
        *,
        source_owner: str | None = None,
        strict: bool = True,
    ) -> list[str]:
        """Load weights for all registered modules belonging to ``owner`` from ``ckpt_path``."""
        clean_owner = owner.replace(".", "__")
        src_owner = (source_owner or owner).replace(".", "__")
        path, raw = self._read_state_dict(ckpt_path)

        target_keys = [
            k for k, info in self.info.items() if info.owner == owner or k.startswith(f"{clean_owner}__")
        ]
        loaded: list[str] = []
        for tk in target_keys:
            short_name = tk.split("__", 1)[1] if "__" in tk else tk
            src_key = f"{src_owner}__{short_name}" if "__" in tk else tk
            sub_sd: dict[str, Any] = {}
            for prefix in (f"mods.{src_key}.", f"{src_key}.", f"mods.{tk}.", f"{tk}."):
                sub_sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
                if sub_sd:
                    break
            if not sub_sd and source_owner is None:
                marker = f"__{short_name}."
                sub_sd = {k.split(marker, 1)[1]: v for k, v in raw.items() if marker in k}
            if sub_sd:
                self.mods[tk].load_state_dict(sub_sd, strict=strict)
                loaded.append(tk)
        if not loaded:
            raise RuntimeError(
                f"No module weights found for owner '{owner}' in {path} (available keys: {list(raw.keys())})"
            )
        return loaded


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
    # "pinv_logit_equiv" (DSKDv2) and "ridge_buffer" (IMPACT) need model weights
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


@PROJECTORS.register("dsl")
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
