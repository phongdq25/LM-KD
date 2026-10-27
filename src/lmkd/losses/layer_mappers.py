"""Layer selection and matching for IMPACT and other feature distillation losses."""
from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import LAYER_MAPPERS
from lmkd.core.state import TrainState
from lmkd.models.logits_view import LogitsView

Pair = tuple[int, int, float]
_SOURCE = "white-box-distillation/cross-tokenizer/IMPACT/code/impact_utils.py:157-633"


@dataclass
class MapperInputs:
    """Layer counts, current state, lazy hidden access and optional PPL/ALP inputs."""

    L_S: int
    L_T: int
    state: TrainState
    teacher_hidden: Callable[[int], torch.Tensor] | None = None
    student_hidden: Callable[[int], torch.Tensor] | None = None
    teacher_labels: torch.Tensor | None = None
    teacher_view_at_layer: Callable[[int], LogitsView] | None = None
    W: torch.Tensor | None = None
    rt: Any = None


def compute_bi_scores(inp: MapperInputs) -> torch.Tensor:
    """BI(l) = 1 - mean cos(H[l-1], H[l]), for teacher layers 1..L_T."""
    if inp.teacher_hidden is None:
        raise ValueError("BI layer selection needs teacher hidden states")
    return torch.stack([1 - F.cosine_similarity(inp.teacher_hidden(l - 1).float(),
                                               inp.teacher_hidden(l).float(), dim=-1, eps=1e-6).mean()
                        for l in range(1, inp.L_T + 1)])


def compute_ppl_scores(inp: MapperInputs) -> torch.Tensor:
    """Per-layer target NLL via a chunked LogitsView; original selects the largest NLL."""
    if inp.teacher_labels is None or inp.teacher_view_at_layer is None:
        raise ValueError("PPL layer selection needs teacher labels and teacher head")
    labels = inp.teacher_labels
    valid = labels.ne(-100)
    scores = []
    for layer in range(1, inp.L_T + 1):
        logp = inp.teacher_view_at_layer(layer).target_logp(labels)
        scores.append(-(logp * valid).sum() / valid.sum().clamp_min(1))
    return torch.stack(scores)


class LayerMapper(Plugin):
    """Map one-based teacher and student hidden-state layers to weighted pairs."""

    needs_all_teacher_layers: ClassVar[bool] = False

    class Config(BaseConfig):
        k: int = 4
        tau: float = 1.0

    def prepare(self, rt: Any) -> None:
        """Load offline calibration, if this mapper uses it."""

    def observe(self, inp: MapperInputs) -> None:
        """Update online or accumulating calibration, if this mapper uses it."""

    def pairs(self, inp: MapperInputs) -> list[Pair]:
        raise NotImplementedError

    def fuse_weights(self, inp: MapperInputs) -> torch.Tensor | None:
        """ALP-KD may return [B,L_S,L_T] attention weights; other mappers return None."""
        return None


def _depth_pair(layer: int, L_T: int, L_S: int, weight: float) -> Pair:
    return layer, math.ceil(layer * L_S / L_T), weight


class _ScoreMapper(LayerMapper):
    """Shared BI/PPL calibration and softmax selection state."""

    class Config(LayerMapper.Config):
        calibration: str = "accumulate"
        calib_samples: int = 256

    def __init__(self, cfg: BaseConfig | None = None):
        super().__init__(cfg)
        self._score_sum: torch.Tensor | None = None
        self._scores: torch.Tensor | None = None
        self._frozen: torch.Tensor | None = None
        self._count = 0

    def _batch_scores(self, inp: MapperInputs) -> torch.Tensor:
        raise NotImplementedError

    def observe(self, inp: MapperInputs) -> None:
        if self.cfg.calibration not in ("offline", "accumulate", "online"):
            raise ValueError(f"unknown layer calibration {self.cfg.calibration!r}")
        if self._frozen is not None and self.cfg.calibration != "online":
            return
        scores = self._batch_scores(inp).detach().float()
        if self.cfg.calibration == "online":
            self._scores = scores
            return
        if inp.teacher_hidden is None:
            raise ValueError("calibration needs teacher hidden states")
        batch_size = inp.teacher_hidden(1).shape[0]
        self._score_sum = scores * batch_size if self._score_sum is None else self._score_sum.to(scores) + scores * batch_size
        self._count += batch_size
        self._scores = self._score_sum / self._count
        if self._count >= self.cfg.calib_samples:
            self._frozen = self._scores.clone()

    def pairs(self, inp: MapperInputs) -> list[Pair]:
        scores = self._frozen if self._frozen is not None else self._scores
        if scores is None:
            self.observe(inp)
            scores = self._frozen if self._frozen is not None else self._scores
        assert scores is not None
        if self.cfg.tau <= 0 or self.cfg.k <= 0:
            raise ValueError("layer mapper k and tau must be positive")
        selected = scores.topk(min(self.cfg.k, inp.L_T)).indices.sort().values
        weights = F.softmax(scores[selected] / self.cfg.tau, dim=0)
        return [_depth_pair(int(l) + 1, inp.L_T, inp.L_S, float(w)) for l, w in zip(selected, weights)]

    def state_dict(self) -> dict:
        return {"score_sum": self._score_sum, "scores": self._scores, "frozen": self._frozen, "count": self._count}

    def load_state_dict(self, state: dict) -> None:
        self._score_sum = state.get("score_sum")
        self._scores = state.get("scores")
        self._frozen = state.get("frozen")
        self._count = int(state.get("count", 0))


@LAYER_MAPPERS.register("topk_bi", source=_SOURCE)
class TopKBI(_ScoreMapper):
    """Top-K block influence, softmax(BI/tau), and relative-depth pairs; IMPACT impact_utils.py:157-176, 619-633."""

    needs_all_teacher_layers = True

    class Config(_ScoreMapper.Config):
        calibration: str = "offline"

    def prepare(self, rt: Any) -> None:
        if self.cfg.calibration != "offline":
            return
        scores = getattr(rt, "extra", {}).get("impact_layer_scores")
        if scores is None:
            logging.getLogger(__name__).warning("IMPACT offline BI unavailable; accumulating train batches")
            self.cfg.calibration = "accumulate"
        else:
            self._frozen = torch.as_tensor(scores).detach().float()

    def _batch_scores(self, inp: MapperInputs) -> torch.Tensor:
        return compute_bi_scores(inp)


@LAYER_MAPPERS.register("topk_ppl", source=_SOURCE)
class TopKPPL(_ScoreMapper):
    """Top-K largest teacher layer NLL, as in original PPL ablation (impact_utils.py:297-316); high NLL is intentional."""

    needs_all_teacher_layers = True

    class Config(_ScoreMapper.Config):
        calibration: str = "accumulate"

    def observe(self, inp: MapperInputs) -> None:
        if self.cfg.calibration == "offline":
            raise ValueError("topk_ppl calibration must be 'accumulate' or 'online'")
        super().observe(inp)

    def _batch_scores(self, inp: MapperInputs) -> torch.Tensor:
        return compute_ppl_scores(inp)


class _FixedMapper(LayerMapper):
    """Uniform-weight mapper for deterministic layer indices."""

    def _layers(self, inp: MapperInputs) -> list[int]:
        raise NotImplementedError

    def pairs(self, inp: MapperInputs) -> list[Pair]:
        layers = self._layers(inp)
        if not layers:
            raise ValueError("layer mapper selected no layers")
        return [_depth_pair(l, inp.L_T, inp.L_S, 1 / len(layers)) for l in layers]


@LAYER_MAPPERS.register("first", source=_SOURCE)
class First(_FixedMapper):
    """First K transformer layers with uniform weights; IMPACT impact_utils.py:270-317."""

    def _layers(self, inp: MapperInputs) -> list[int]:
        return list(range(1, min(self.cfg.k, inp.L_T) + 1))


@LAYER_MAPPERS.register("last", source=_SOURCE)
class Last(_FixedMapper):
    """Last K transformer layers with uniform weights; IMPACT impact_utils.py:270-317."""

    def _layers(self, inp: MapperInputs) -> list[int]:
        return list(range(max(1, inp.L_T - self.cfg.k + 1), inp.L_T + 1))


@LAYER_MAPPERS.register("random", source=_SOURCE)
class Random(_FixedMapper):
    """Fixed seeded random K-layer subset, as IMPACT impact_utils.py:270-317."""

    class Config(LayerMapper.Config):
        seed: int | None = None

    def _layers(self, inp: MapperInputs) -> list[int]:
        seed = self.cfg.seed
        if seed is None:
            seed = inp.rt.config.train.seed
        return sorted(l + 1 for l in random.Random(seed).sample(range(inp.L_T), min(self.cfg.k, inp.L_T)))


@LAYER_MAPPERS.register("ratio_1to1", source=_SOURCE)
class RatioOneToOne(LayerMapper):
    """Each student layer m receives teacher ceil(m*L_T/L_S), weight 1/L_S; IMPACT impact_utils.py:603-608."""

    def pairs(self, inp: MapperInputs) -> list[Pair]:
        return [(math.ceil(m * inp.L_T / inp.L_S), m, 1 / inp.L_S) for m in range(1, inp.L_S + 1)]


@LAYER_MAPPERS.register("bi_ratio", source="T15 extension for ablation")
class BIRatio(_FixedMapper):
    """Extension for ablation: user-selected one-based teacher layers mapped by relative depth."""

    class Config(LayerMapper.Config):
        layers: list[int] = []

    def _layers(self, inp: MapperInputs) -> list[int]:
        if any(not 1 <= l <= inp.L_T for l in self.cfg.layers):
            raise ValueError("bi_ratio layers must be one-based teacher layer indices")
        return self.cfg.layers


def sample_rail_layers(L_T: int, L_S: int, seed: int | None, *, no_random: bool,
                       has_embed: bool, has_final: bool) -> list[int]:
    """GLMKD RAIL-KD hook sample; IMPACT impact_utils.py:429-484."""
    if L_T < 1 or L_S < 1:
        return [1]
    n_sample = max(L_S - 1, 1)
    if no_random:
        interval = max(int(L_T / max(L_S, 1)), 1)
        layers = [int(l) for l in list(range(0, L_T + 1, interval))[1:-1] if 0 < l < L_T]
        if len(layers) < n_sample:
            layers += [layers[-1] if layers else 1] * (n_sample - len(layers))
        layers = layers[:n_sample]
    else:
        pool = list(range(1, L_T)) or [1]
        layers = sorted(random.Random(seed).sample(pool, min(n_sample, len(pool))))
        if len(layers) < n_sample:
            layers += [layers[-1]] * (n_sample - len(layers))
    return ([0] if has_embed else []) + layers + ([L_T] if has_final else [])


@LAYER_MAPPERS.register("railkd", source=_SOURCE)
class RailKD(LayerMapper):
    """RAIL-KD pairs and epoch/iteration resampling; IMPACT impact_utils.py:429-514, 1098-1153.

    Unlike the original, this alignment is available to every criterion using the mapper.
    """

    class Config(LayerMapper.Config):
        resample_epochs: int = 1
        resample_iters: int = 0
        no_random: bool = False
        has_embed: bool = False
        has_final: bool = False
        seed: int | None = None

    def __init__(self, cfg: BaseConfig | None = None):
        super().__init__(cfg)
        self._layers: list[int] | None = None
        self._last_epoch = -1
        self._last_step = -1

    def pairs(self, inp: MapperInputs) -> list[Pair]:
        epoch, step = int(inp.state.epoch), inp.state.step
        due = self._layers is None
        if self.cfg.resample_iters > 0:
            due |= step % self.cfg.resample_iters == 0 and step != self._last_step
        elif self.cfg.resample_epochs > 0:
            due |= epoch % self.cfg.resample_epochs == 0 and epoch != self._last_epoch
        if due:
            seed = self.cfg.seed
            if seed is None and inp.rt is not None:
                seed = inp.rt.config.train.seed
            seed_eff = None if seed is None else int(seed) + epoch * 10007 + step
            self._layers = sample_rail_layers(inp.L_T, inp.L_S, seed_eff, no_random=self.cfg.no_random,
                                              has_embed=self.cfg.has_embed, has_final=self.cfg.has_final)
            self._last_epoch, self._last_step = epoch, step
        assert self._layers is not None
        n = inp.L_S if self.cfg.has_final else max(inp.L_S - 1, 1)
        layers = self._layers[1:] if self.cfg.has_embed else self._layers
        return [(layers[i], i + 1, 1 / n) for i in range(n)]

    def state_dict(self) -> dict:
        return {"layers": self._layers, "last_epoch": self._last_epoch, "last_step": self._last_step}

    def load_state_dict(self, state: dict) -> None:
        self._layers = state.get("layers")
        self._last_epoch = int(state.get("last_epoch", -1))
        self._last_step = int(state.get("last_step", -1))


@LAYER_MAPPERS.register("alpkd", source=_SOURCE)
class ALPKD(LayerMapper):
    """ALP-KD softmax over projected teacher CLS dot student CLS; IMPACT impact_utils.py:517-553."""

    needs_all_teacher_layers = True

    def pairs(self, inp: MapperInputs) -> list[Pair]:
        return [(-1, m, 1 / inp.L_S) for m in range(1, inp.L_S + 1)]

    def fuse_weights(self, inp: MapperInputs) -> torch.Tensor:
        if inp.student_hidden is None or inp.teacher_hidden is None or inp.W is None:
            raise ValueError("ALP-KD needs both hidden stacks and projection W")
        s_cls = torch.stack([inp.student_hidden(m)[:, 0].float() for m in range(1, inp.L_S + 1)], dim=1)
        W = inp.W.to(device=s_cls.device, dtype=torch.float32)
        t_cls = torch.stack([inp.teacher_hidden(l)[:, 0].float() @ W for l in range(1, inp.L_T + 1)], dim=1)
        return F.softmax(torch.bmm(s_cls, t_cls.transpose(1, 2)), dim=-1)
