"""Embedding distillation criteria (TALAS: InfoNCE / SimCSE, TAMD, LASD).

Source: embedding-distill/TALAS-M-main
Paper: TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.modules import LinearProjector
from lmkd.teachers.signals import TeacherSignalSpec


@CRITERIONS.register(
    "infonce",
    source="embedding-distill/TALAS-M-main/trainer.py:113-130",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
)
class InfoNCE(Criterion):
    """Unsupervised SimCSE InfoNCE over n_views=2 dropout views.

    Formula:
      q = normalize(view[0].pooled(layer, pooler), dim=-1)
      p = normalize(view[1].pooled(layer, pooler), dim=-1)
      scores = (q @ p.T) / tau
      loss = cross_entropy(scores, arange(B))
    Source: embedding-distill/TALAS-M-main/trainer.py:113-130
    """

    supports_packing = False
    needs_teacher = False

    class Config(BaseConfig):
        temperature: float = 0.05
        layer: int = -1
        pooler: str = "cls_attention"

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(pooled=((self.cfg.layer, self.cfg.pooler),), n_views=2)
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        q = F.normalize(ctx.student_views[0].pooled(self.cfg.layer, self.cfg.pooler), p=2, dim=-1)
        p = F.normalize(ctx.student_views[1].pooled(self.cfg.layer, self.cfg.pooler), p=2, dim=-1)
        B = q.shape[0]
        scores = torch.matmul(q, p.t()) / self.cfg.temperature
        targets = torch.arange(B, device=scores.device, dtype=torch.long)
        loss = F.cross_entropy(scores, targets, reduction="mean")
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

    def build_modules(self, info: ModelPairInfo, modules: Any) -> None:
        d_s = info.student_hidden
        d_t = self.cfg.teacher_dim or info.teacher_hidden
        if d_t is None:
            raise ValueError("set teacher_dim (= manifest dim of embed_teacher)")
        self._d_t = d_t
        self._L = info.student_layers
        for l in self.cfg.layers:
            norm_l = l + self._L + 1 if l < 0 else l
            modules.add(
                f"{self.alias}__proj{norm_l}",
                LinearProjector(LinearProjector.Config(in_dim=d_s, out_dim=d_t, bias=False)),
                init=self.cfg.proj_init,
                group="projector",
                owner=self.alias,
            )

    def features(self) -> FeatureRequest:
        pooled = tuple((l, self.cfg.pooler) for l in self.cfg.layers)
        return FeatureRequest(
            student=SideRequest(pooled=pooled, n_views=2),
            teacher_signal=TeacherSignalSpec(kind="embedding"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        assert ctx.teacher is not None and ctx.teacher.embedding is not None, "teacher embedding signal required"
        t = ctx.teacher.embedding
        if hasattr(self, "_d_t"):
            assert t.shape[-1] == self._d_t, f"teacher embedding dim mismatch: expected {self._d_t}, got {t.shape[-1]}"
        L = ctx.student.num_layers
        losses = []
        for l in self.cfg.layers:
            norm_l = l + L + 1 if l < 0 else l
            pool = ctx.student_views[0].pooled(l, self.cfg.pooler)
            proj = ctx.modules.get(f"{self.alias}__proj{norm_l}")
            h_proj = proj(pool)
            cos = F.cosine_similarity(h_proj, t, dim=-1)
            losses.append((1.0 - cos).mean())
        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=t.device)
        return LossOutput(terms={self.alias: LossTerm(loss, "mean")})


@CRITERIONS.register(
    "sim_matrix_mse",
    source="embedding-distill/TALAS-M-main/trainer.py:68-77,99-103",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
)
class SimMatrixMSE(Criterion):
    """LASD: MSE between BxB normalized similarity matrices of adjacent layers.

    Formula:
      e_i = normalize(pool_i(view[0]), dim=-1)
      S_i = e_i @ e_i.T
      loss = mean_{i=start}^{L-1} MSE(S_i, S_{i+1})
    Source: embedding-distill/TALAS-M-main/trainer.py:68-77,99-103
    """

    supports_packing = False
    needs_teacher = False

    class Config(BaseConfig):
        pooler: str = "cls_attention"
        start: int = 1
        detach_target: bool = False

    def build_modules(self, info: ModelPairInfo, modules: Any) -> None:
        self._L = info.student_layers

    def features(self) -> FeatureRequest:
        L = getattr(self, "_L", None)
        if L is not None:
            pooled = tuple((i, self.cfg.pooler) for i in range(self.cfg.start, L + 1))
        else:
            pooled = ()
        return FeatureRequest(student=SideRequest(hidden="all", pooled=pooled, n_views=2))

    def forward(self, ctx: LossContext) -> LossOutput:
        L = getattr(self, "_L", None) or ctx.student.num_layers
        losses = []
        for i in range(self.cfg.start, L):
            e_i = F.normalize(ctx.student_views[0].pooled(i, self.cfg.pooler), p=2, dim=-1)
            e_next = F.normalize(ctx.student_views[0].pooled(i + 1, self.cfg.pooler), p=2, dim=-1)
            if self.cfg.detach_target:
                e_next = e_next.detach()
            S_i = torch.matmul(e_i, e_i.t())
            S_next = torch.matmul(e_next, e_next.t())
            losses.append(F.mse_loss(S_i, S_next, reduction="mean"))
        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=ctx.student.last_hidden.device)
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

    def build_modules(self, info: ModelPairInfo, modules: Any) -> None:
        d_s = info.student_hidden
        d_t = self.cfg.teacher_dim or info.teacher_hidden
        if d_t is None:
            raise ValueError("set teacher_dim")
        self._d_t = d_t
        modules.add(
            f"{self.alias}__proj",
            LinearProjector(LinearProjector.Config(in_dim=d_s, out_dim=d_t, bias=False)),
            init=self.cfg.proj_init,
            group="projector",
            owner=self.alias,
        )

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(pooled=((self.cfg.layer, self.cfg.pooler),), n_views=2),
            teacher_signal=TeacherSignalSpec(kind="embedding"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        assert ctx.teacher is not None and ctx.teacher.embedding is not None, "teacher embedding signal required"
        pool = ctx.student_views[0].pooled(self.cfg.layer, self.cfg.pooler)
        proj = ctx.modules.get(f"{self.alias}__proj")
        s = proj(pool)
        t = ctx.teacher.embedding
        std_embeddings = t.std(dim=0)
        mean_std = std_embeddings.mean()
        std_scaled = std_embeddings / mean_std.clamp(min=1e-9)
        weights = std_scaled ** self.cfg.power
        squared_diff = (s - t) ** 2
        weighted_squared_diff = squared_diff * weights
        loss = weighted_squared_diff.mean()
        return LossOutput(terms={self.alias: LossTerm(loss, "mean")})

