"""Embedding distillation criteria for TALAS baseline.

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


@CRITERIONS.register(
    "lasd",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
)
class LASD(Criterion):
    """LASD: MSE between BxB normalized similarity matrices of adjacent layers.

    Formula:
      e_i = normalize(pool_i(view[0]), dim=-1)
      S_i = e_i @ e_i.T
      loss = mean_{i=start}^{L-1} MSE(S_i, S_{i+1})
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
        return FeatureRequest(student=SideRequest(hidden="all", pooled=pooled, n_views=1))

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
