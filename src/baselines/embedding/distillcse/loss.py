"""DistillCSE (Distilled Contrastive Learning for Sentence Embeddings) baseline.

Paper: DistillCSE: Distilled Contrastive Learning for Sentence Embeddings (EMNLP 2023)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def distillcse_kd_loss(
    s1: torch.Tensor,
    s2: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    tau: float = 0.05,
    detach_teacher: bool = True,
) -> torch.Tensor:
    """DistillCSE cross-view KD loss between student views (s1, s2) and teacher representations (t1, t2).

    Formula (from embedding-distill/distillcse (2).ipynb:cell 18):
      s1 = normalize(s1), s2 = normalize(s2)
      t1 = normalize(t1), t2 = normalize(t2)
      s_logits = (s1 @ s2.T) / tau
      t_logits = (t1 @ t2.T) / tau
      mask diagonal (self-pairs) with -inf
      t_probs = softmax(t_logits, dim=1)
      s_log_probs = log_softmax(s_logits, dim=1)
      kd = KL(s_log_probs || t_probs, reduction="batchmean")
    """
    if s1.shape[0] < 2:
        return torch.tensor(0.0, device=s1.device, dtype=s1.dtype, requires_grad=s1.requires_grad)

    s1_norm = F.normalize(s1, p=2, dim=-1)
    s2_norm = F.normalize(s2, p=2, dim=-1)
    t1_norm = F.normalize(t1, p=2, dim=-1)
    t2_norm = F.normalize(t2, p=2, dim=-1)

    s_logits = torch.matmul(s1_norm, s2_norm.transpose(0, 1)) / tau
    t_logits = torch.matmul(t1_norm, t2_norm.transpose(0, 1)) / tau

    mask = torch.eye(s_logits.size(0), device=s_logits.device, dtype=torch.bool)
    neg_inf_s = torch.finfo(s_logits.dtype).min
    neg_inf_t = torch.finfo(t_logits.dtype).min

    s_logits = s_logits.masked_fill(mask, neg_inf_s)
    t_logits = t_logits.masked_fill(mask, neg_inf_t)

    t_probs = F.softmax(t_logits, dim=1)
    if detach_teacher:
        t_probs = t_probs.detach()

    s_log_probs = F.log_softmax(s_logits, dim=1)
    kd = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
    return kd


@CRITERIONS.register(
    "distillcse",
    paper="DistillCSE: Distilled Contrastive Learning for Sentence Embeddings",
)
class DistillCSE(Criterion):
    """DistillCSE: Cross-view contrastive distillation with masked diagonal KL divergence.

    Paper: DistillCSE: Distilled Contrastive Learning for Sentence Embeddings (EMNLP 2023)
    """

    supports_packing = False
    needs_teacher = True

    class Config(BaseConfig):
        tau: float = 0.05
        layer: int = -1
        pooler: str = "cls"
        detach_teacher: bool = True
        bidirectional: bool = False

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(pooled=((self.cfg.layer, self.cfg.pooler),), n_views=2),
            teacher_signal=TeacherSignalSpec(kind="embedding"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        assert len(ctx.student_views) >= 2, (
            f"{self.alias} requires at least 2 student views (got {len(ctx.student_views)}). "
            "Ensure n_views=2 is satisfied."
        )
        s1 = ctx.student_views[0].pooled(self.cfg.layer, self.cfg.pooler)
        s2 = ctx.student_views[1].pooled(self.cfg.layer, self.cfg.pooler)

        assert ctx.teacher is not None and ctx.teacher.embedding is not None, (
            f"{self.alias} requires teacher embedding signal (e.g. from embedding_store or teacher model)"
        )
        t = ctx.teacher.embedding
        t1 = t
        t2 = t

        loss = distillcse_kd_loss(
            s1=s1,
            s2=s2,
            t1=t1,
            t2=t2,
            tau=self.cfg.tau,
            detach_teacher=self.cfg.detach_teacher,
        )

        if self.cfg.bidirectional:
            loss_rev = distillcse_kd_loss(
                s1=s2,
                s2=s1,
                t1=t2,
                t2=t1,
                tau=self.cfg.tau,
                detach_teacher=self.cfg.detach_teacher,
            )
            loss = (loss + loss_rev) * 0.5

        return LossOutput(terms={self.alias: LossTerm(loss, "mean")})


@CRITERIONS.register(
    "distillcse_kd",
    paper="DistillCSE: Distilled Contrastive Learning for Sentence Embeddings",
)
class DistillCSEKD(DistillCSE):
    """Alias for DistillCSE criterion."""


