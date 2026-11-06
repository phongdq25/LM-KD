"""Embedding distillation criteria (SimCSE Unsupervised, InfoNCE, TAMD, LASD).

Sources:
- SimCSE Unsup: embedding-distill/TALAS-M-main/trainer.py:113-130
- Paper: SimCSE: Simple Contrastive Sentence Embedding Framework (Gao et al., EMNLP 2021)
- InfoNCE: Representation Learning with Contrastive Predictive Coding (van den Oord et al., 2018)
- TAMD & LASD: TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo
from lmkd.losses.functional.contrastive import info_nce_loss
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.modules import LinearProjector, ModuleRegistry
from lmkd.teachers.signals import TeacherSignalSpec

__all__ = [
    "CosineToTeacher",
    "DimWeightedMSE",
    "InfoNCE",
    "SimCSEUnsup",
    "info_nce_loss",
]


@CRITERIONS.register(
    "simcse_unsupervised",
    source="embedding-distill/TALAS-M-main/trainer.py:113-130",
    paper="SimCSE: Simple Contrastive Sentence Embedding Framework (Gao et al., EMNLP 2021)",
)
class SimCSEUnsup(Criterion):
    """Unsupervised SimCSE contrastive loss over n_views=2 stochastic dropout views.

    Formula:
      q = normalize(view[0].pooled(layer, pooler), dim=-1)
      p = normalize(view[1].pooled(layer, pooler), dim=-1)
      scores = (q @ p.T) / tau
      loss = cross_entropy(scores, arange(B))

    Source: embedding-distill/TALAS-M-main/trainer.py:113-130
    Paper: SimCSE: Simple Contrastive Sentence Embedding Framework (Gao et al., EMNLP 2021)
    """

    supports_packing = False
    needs_teacher = False

    class Config(BaseConfig):
        temperature: float = 0.05
        layer: int = -1
        pooler: str = "cls_attention"
        bidirectional: bool = False

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(pooled=((self.cfg.layer, self.cfg.pooler),), n_views=2))

    def forward(self, ctx: LossContext) -> LossOutput:
        assert len(ctx.student_views) >= 2, (
            f"SimCSEUnsup requires at least 2 student views (got {len(ctx.student_views)}). "
            "Ensure n_views=2 is satisfied by ForwardRunner."
        )
        q = ctx.student_views[0].pooled(self.cfg.layer, self.cfg.pooler)
        p = ctx.student_views[1].pooled(self.cfg.layer, self.cfg.pooler)
        loss = info_nce_loss(
            q=q,
            k=p,
            temperature=self.cfg.temperature,
            bidirectional=self.cfg.bidirectional,
            reduction="mean",
        )
        return LossOutput(terms={self.alias: LossTerm(loss, "mean")})


@CRITERIONS.register("infonce")
class InfoNCE(Criterion):
    """Generic InfoNCE contrastive distillation loss between query representations and target representations.

    Formula:
      q = proj(student.pooled(layer, pooler))   # normalized
      k = target representation (e.g. teacher embedding)  # normalized
      scores = (q @ k.T) / tau
      loss = cross_entropy(scores, arange(B))

    Unlike SimCSE unsupervised, InfoNCE operates across distinct query and target representations
    (e.g., student vs. teacher embedding signal or positive paired keys), requiring only n_views=1.
    """

    supports_packing = False
    needs_teacher = True

    class Config(BaseConfig):
        temperature: float = 0.07
        layer: int = -1
        pooler: str = "cls_attention"
        target: str = "teacher"  # "teacher" | "batch"
        teacher_dim: int | None = None
        projector: bool = True
        proj_init: str = "normal_1e-3"
        bidirectional: bool = False

    cfg: Config

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        if self.cfg.target == "teacher" and self.cfg.projector:
            d_s = info.student_hidden
            d_t = self.cfg.teacher_dim or info.teacher_hidden
            if d_s is not None and d_t is not None and d_s != d_t:
                self.register_aux_module(
                    modules,
                    "proj",
                    LinearProjector(LinearProjector.Config(in_dim=d_s, out_dim=d_t, bias=False)),
                    init=self.cfg.proj_init,
                    group="projector",
                )

    def features(self) -> FeatureRequest:
        student_req = SideRequest(pooled=((self.cfg.layer, self.cfg.pooler),), n_views=1)
        if self.cfg.target == "teacher":
            return FeatureRequest(
                student=student_req,
                teacher_signal=TeacherSignalSpec(kind="embedding"),
            )
        return FeatureRequest(student=student_req)

    def forward(self, ctx: LossContext) -> LossOutput:
        q = ctx.student.pooled(self.cfg.layer, self.cfg.pooler)
        proj = self.aux(ctx).get("proj")
        if proj is not None:
            q = proj(q)

        if self.cfg.target == "teacher":
            assert ctx.teacher is not None and ctx.teacher.embedding is not None, (
                "InfoNCE with target='teacher' requires teacher embedding signal"
            )
            k = ctx.teacher.embedding
        elif self.cfg.target == "batch":
            assert "pos.embedding" in ctx.batch, "InfoNCE with target='batch' requires 'pos.embedding' in batch"
            k = ctx.batch["pos.embedding"]
        else:
            raise ValueError(f"Unsupported InfoNCE target: {self.cfg.target}")

        loss = info_nce_loss(
            q=q,
            k=k,
            temperature=self.cfg.temperature,
            bidirectional=self.cfg.bidirectional,
            reduction="mean",
        )
        return LossOutput(terms={self.alias: LossTerm(loss, "mean")})


@CRITERIONS.register(
    "cosine_to_teacher",
    source="embedding-distill/TALAS-M-main/trainer.py:87-97",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
)
class CosineToTeacher(Criterion):
    """TAMD: mean_l mean_b (1 - cos(proj_l(pool_l), t)) over target layers.

    Formula:
      loss = mean_{l in layers} mean_b (1 - cos(proj_l(pool_l(view[0])), t))
    Source: embedding-distill/TALAS-M-main/trainer.py:87-97
    """

    supports_packing = False
    needs_teacher = True

    class Config(BaseConfig):
        layers: list[int] = [-2, -1]
        pooler: str = "cls_attention"
        teacher_dim: int | None = None
        proj_init: str = "normal_1e-3"

    cfg: Config

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        d_s = info.student_hidden
        d_t = self.cfg.teacher_dim or info.teacher_hidden
        if d_t is None:
            raise ValueError("set teacher_dim (= manifest dim of embed_teacher)")
        self._d_t = d_t
        self._L = info.student_layers
        for l in self.cfg.layers:
            norm_l = l + self._L + 1 if l < 0 else l
            self.register_aux_module(
                modules,
                f"proj{norm_l}",
                LinearProjector(LinearProjector.Config(in_dim=d_s, out_dim=d_t, bias=False)),
                init=self.cfg.proj_init,
                group="projector",
            )

    def features(self) -> FeatureRequest:
        pooled = tuple((l, self.cfg.pooler) for l in self.cfg.layers)
        return FeatureRequest(
            student=SideRequest(pooled=pooled, n_views=1),
            teacher_signal=TeacherSignalSpec(kind="embedding"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        assert ctx.teacher is not None and ctx.teacher.embedding is not None, "teacher embedding signal required"
        t = ctx.teacher.embedding
        if hasattr(self, "_d_t"):
            assert t.shape[-1] == self._d_t, f"teacher embedding dim mismatch: expected {self._d_t}, got {t.shape[-1]}"
        L = ctx.student.num_layers
        aux = self.aux(ctx)
        losses = []
        for l in self.cfg.layers:
            norm_l = l + L + 1 if l < 0 else l
            pool = ctx.student_views[0].pooled(l, self.cfg.pooler)
            proj = aux[f"proj{norm_l}"]
            h_proj = proj(pool)
            cos = F.cosine_similarity(h_proj, t, dim=-1)
            losses.append((1.0 - cos).mean())
        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=t.device)
        return LossOutput(terms={self.alias: LossTerm(loss, "mean")})


@CRITERIONS.register(
    "dim_weighted_mse",
    source="embedding-distill/TALAS-M-main/trainer.py:56-66",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
)
class DimWeightedMSE(Criterion):
    """Port of TALAS mse_dim_weight_loss:
    w_d = (std_d(t) / mean(std)) ** power
    loss = mean((s - t)^2 * w)
    Source: embedding-distill/TALAS-M-main/trainer.py:56-66
    """

    supports_packing = False
    needs_teacher = True

    class Config(BaseConfig):
        layer: int = -1
        pooler: str = "cls_attention"
        teacher_dim: int | None = None
        proj_init: str = "normal_1e-3"
        power: float = 1.0

    cfg: Config

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        d_s = info.student_hidden
        d_t = self.cfg.teacher_dim or info.teacher_hidden
        if d_t is None:
            raise ValueError("set teacher_dim")
        self._d_t = d_t
        self.register_aux_module(
            modules,
            "proj",
            LinearProjector(LinearProjector.Config(in_dim=d_s, out_dim=d_t, bias=False)),
            init=self.cfg.proj_init,
            group="projector",
        )

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(pooled=((self.cfg.layer, self.cfg.pooler),), n_views=1),
            teacher_signal=TeacherSignalSpec(kind="embedding"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        assert ctx.teacher is not None and ctx.teacher.embedding is not None, "teacher embedding signal required"
        pool = ctx.student_views[0].pooled(self.cfg.layer, self.cfg.pooler)
        proj = self.aux(ctx).proj
        s = proj(pool)
        t = ctx.teacher.embedding
        std_embeddings = t.std(dim=0)
        mean_std = std_embeddings.mean()
        std_scaled = std_embeddings / mean_std.clamp(min=1e-9)
        weights = std_scaled**self.cfg.power
        squared_diff = (s - t) ** 2
        weighted_squared_diff = squared_diff * weights
        loss = weighted_squared_diff.mean()
        return LossOutput(terms={self.alias: LossTerm(loss, "mean")})
