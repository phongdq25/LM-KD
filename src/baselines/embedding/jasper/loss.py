"""Jasper (distillation of SOTA embedding models) baseline.

Paper: Jasper and Stella: distillation of SOTA embedding models
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo, PrepareContext
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.modules import LinearProjector, ModuleRegistry
from lmkd.teachers.signals import TeacherSignalSpec


def get_score_diff(embedding: torch.Tensor) -> torch.Tensor:
    """Pairwise similarity difference between unique pairs (embedding-distill/jasper (1).ipynb:cell 15).

    Formula:
      scores = (E @ E.T)[triu(1)]
      score_diff = (scores - scores.T)[triu(1)]
    """
    scores = torch.matmul(embedding, embedding.T)
    mask1 = torch.triu(torch.ones_like(scores), diagonal=1).bool()
    scores_upper = scores[mask1]
    if scores_upper.numel() < 2:
        return torch.empty(0, device=embedding.device, dtype=embedding.dtype)
    score_diff_mat = scores_upper.reshape((1, -1)) - scores_upper.reshape((-1, 1))
    mask2 = torch.triu(torch.ones_like(score_diff_mat), diagonal=1).bool()
    return score_diff_mat[mask2]


def cosine_embedding_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Cosine embedding loss (embedding-distill/jasper (1).ipynb:cell 15).

    Formula:
      loss = 1 - cos(normalize(s), normalize(t))
    """
    s_norm = F.normalize(student_embeddings, p=2, dim=-1)
    t_norm = F.normalize(teacher_embeddings, p=2, dim=-1)
    target = torch.ones(s_norm.size(0), device=s_norm.device)
    return F.cosine_embedding_loss(s_norm, t_norm, target)


def pair_inbatch_similarity_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    """In-batch similarity matrix MSE loss (embedding-distill/jasper (1).ipynb:cell 15).

    Formula:
      S = normalize(s) @ normalize(s).T
      T = normalize(t) @ normalize(t).T
      loss = MSE(S, T)
    """
    s_norm = F.normalize(student_embeddings, p=2, dim=-1)
    t_norm = F.normalize(teacher_embeddings, p=2, dim=-1)
    s_sim = torch.matmul(s_norm, s_norm.transpose(-1, -2))
    t_sim = torch.matmul(t_norm, t_norm.transpose(-1, -2))
    return F.mse_loss(s_sim, t_sim)


def pair_inbatch_triplet_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    triplet_margin: float = 0.015,
) -> torch.Tensor:
    """Relative ranking triplet loss based on score difference signs (embedding-distill/jasper (1).ipynb:cell 15).

    Formula:
      diff_t = get_score_diff(normalize(t))
      diff_s = get_score_diff(normalize(s))
      triplet_label = where(diff_t < 0, 1, -1)
      loss = ReLU(diff_s * triplet_label + margin).mean()
    """
    s_norm = F.normalize(student_embeddings, p=2, dim=-1)
    t_norm = F.normalize(teacher_embeddings, p=2, dim=-1)
    diff_t = get_score_diff(t_norm)
    diff_s = get_score_diff(s_norm)
    if diff_t.numel() == 0 or diff_s.numel() == 0:
        return (student_embeddings * 0.0).sum()
    triplet_label = torch.where(diff_t < 0, 1, -1)
    return F.relu(diff_s * triplet_label + triplet_margin).mean()


@CRITERIONS.register(
    "jasper",
    paper="Jasper and Stella: distillation of SOTA embedding models",
)
class Jasper(Criterion):
    """Jasper embedding distillation combining Cosine Loss, In-batch Similarity MSE and Triplet Ranking Loss.

    Paper: Jasper and Stella: distillation of SOTA embedding models
    """

    supports_packing = False
    needs_teacher = True

    class Config(BaseConfig):
        layer: int = -1
        pooler: str = "cls"
        teacher_dim: int | None = None
        projector: bool = True
        proj_init: str = "normal_1e-3"
        projector_ckpt: str = ""
        w_cos: float = 0.5
        w_sim: float = 2.0
        w_tri: float = 0.5
        triplet_margin: float = 0.015
        mrl_dims: tuple[int, ...] = ()
        w_mrl_sim: float = 2.0
        w_mrl_tri: float = 0.5

    cfg: Config

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        d_s = info.student_hidden
        d_t = self.cfg.teacher_dim or info.teacher_hidden
        if self.cfg.projector and d_s is not None and d_t is not None and d_s != d_t:
            self.register_aux_module(
                modules,
                "proj",
                LinearProjector(LinearProjector.Config(in_dim=d_s, out_dim=d_t, bias=False)),
                init=self.cfg.proj_init,
                group="projector",
            )
        for dim in self.cfg.mrl_dims:
            self.register_aux_module(
                modules,
                f"mrl_{dim}",
                LinearProjector(LinearProjector.Config(in_dim=d_s, out_dim=dim, bias=False)),
                init=self.cfg.proj_init,
                group="projector",
            )

    def prepare(self, ctx: PrepareContext | None) -> None:
        if self.cfg.projector_ckpt:
            self.load_aux_checkpoint(ctx, "proj", allow_raw=False)

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(pooled=((self.cfg.layer, self.cfg.pooler),), n_views=1),
            teacher_signal=TeacherSignalSpec(kind="embedding"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        s = ctx.student.pooled(self.cfg.layer, self.cfg.pooler)
        aux = self.aux(ctx)
        proj = aux.get("proj")
        s_emb = proj(s) if proj is not None else s

        assert ctx.teacher is not None and ctx.teacher.embedding is not None, (
            f"{self.alias} requires teacher embedding signal"
        )
        t_emb = ctx.teacher.embedding

        loss_cos = cosine_embedding_loss(s_emb, t_emb)
        loss_sim = pair_inbatch_similarity_loss(s_emb, t_emb)
        loss_tri = pair_inbatch_triplet_loss(s_emb, t_emb, triplet_margin=self.cfg.triplet_margin)

        loss = self.cfg.w_cos * loss_cos + self.cfg.w_sim * loss_sim + self.cfg.w_tri * loss_tri

        if self.cfg.mrl_dims and ctx.modules:
            mrl_sim_terms = []
            mrl_tri_terms = []
            for dim in self.cfg.mrl_dims:
                mrl_proj = aux[f"mrl_{dim}"]
                s_mrl = mrl_proj(s)
                mrl_sim_terms.append(pair_inbatch_similarity_loss(s_emb, s_mrl))
                mrl_tri_terms.append(pair_inbatch_triplet_loss(s_emb, s_mrl, triplet_margin=self.cfg.triplet_margin))
            if mrl_sim_terms:
                loss = loss + self.cfg.w_mrl_sim * sum(mrl_sim_terms)
            if mrl_tri_terms:
                loss = loss + self.cfg.w_mrl_tri * sum(mrl_tri_terms)

        metrics = {
            f"{self.alias}/cos": loss_cos.detach(),
            f"{self.alias}/sim": loss_sim.detach(),
            f"{self.alias}/tri": loss_tri.detach(),
        }
        return LossOutput(terms={self.alias: LossTerm(loss, "mean")}, metrics=metrics)
