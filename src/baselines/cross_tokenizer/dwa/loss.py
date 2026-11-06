"""DWA-KD: Dynamic Window Alignment Knowledge Distillation criterion.

Formula:
    1. Cross-Model Attention (CMA) Dual-Space Distillation:
       - Uncertainty-guided student entropy weighting w_S and teacher certainty weighting w_T.
       - Query projection over concatenated input/target embeddings: align = (q @ k^T) / sqrt(2*D_t).
       - Confidence-gated student-space divergence:
           L_t2s_kd = sum( div(p_S, p_t2s) * conf_t2s * warmup_kd * w_S )
       - Teacher-space divergence:
           L_s2t_kd = sum( FKL(p_s2t, p_T) * w_T )
    2. Dynamic Window Alignment (DWA) Soft-DTW Sequence Representation Distillation:
       - Pairwise cosine distance matrix C with adaptive soft band penalty based on CMA cross-attention entropy.
       - Sequence-level divergence:
           D_sDTW(S, T) = sDTW(C_s_t) - 0.5 * (sDTW(C_s_s) + sDTW(C_t_t))
       - Applied across hidden states (h_S vs P_t2s(h_T)) and token embeddings (e_S vs P_embed(e_T)).
"""

from __future__ import annotations

import functools
import math
from typing import Literal

import torch
import torch.nn as nn

from baselines.cross_tokenizer.dwa.soft_dtw import (
    SoftDTW,
    soft_dtw_divergence,
)
from baselines.dskd.common import (
    _DualSpaceBase,
    _get_embeddings,
    _get_heads,
    _load_projector_checkpoint,
    _view_argmax,
)
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import LossContext, LossOutput, LossTerm, ModelPairInfo, PrepareContext
from lmkd.losses.cross_tokenizer._common import (
    DetachedLinear,
    target_hidden,
    target_logits_full,
    vd_divergence,
    view_from_target_hidden,
)
from lmkd.models.modules import DSLProjector, ModuleRegistry


def _cma_embed_indices(
    ids_s: torch.Tensor,
    ids_t: torch.Tensor,
    pm: torch.Tensor,
    tpm: torch.Tensor,
    stu_emb: nn.Module,
    tea_emb: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build normalized cross-model embedding indices for CMA."""
    in_s = torch.zeros_like(ids_s)
    in_s[:, 1:] = ids_s[:, :-1]
    formal_in_s = torch.where(pm.bool(), in_s, torch.zeros_like(in_s))
    formal_tgt_s = torch.where(pm.bool(), ids_s, torch.zeros_like(ids_s))
    stu_index = torch.cat([stu_emb(formal_in_s).detach(), stu_emb(formal_tgt_s).detach()], dim=-1)

    in_t = torch.zeros_like(ids_t)
    in_t[:, 1:] = ids_t[:, :-1]
    formal_in_t = torch.where(tpm.bool(), in_t, torch.zeros_like(in_t))
    formal_tgt_t = torch.where(tpm.bool(), ids_t, torch.zeros_like(ids_t))
    tea_in_emb = tea_emb(formal_in_t).detach()
    tea_tgt_emb = tea_emb(formal_tgt_t).detach()
    tea_index = torch.cat([tea_in_emb, tea_tgt_emb], dim=-1)
    return stu_index, tea_index / (tea_index.std() + 1e-9), tea_tgt_emb / (tea_tgt_emb.std() + 1e-9)


@CRITERIONS.register("dwa_kd")
@CRITERIONS.register("dwa")
class DWAKD(_DualSpaceBase):
    """Dynamic Window Alignment Knowledge Distillation (DWA-KD).

    Integrates Dual-Space CMA with differentiable Soft-DTW representation alignment
    governed by an entropy-guided adaptive window (soft band).
    """

    class Config(BaseConfig):
        # CMA Distillation
        temperature: float = 2.0
        teacher_temperature: float = 1.0
        kd_objective: str = "skewed_reverse_kl"
        skew_lambda: float = 0.1
        adaptive_kl_alpha: float = 0.5
        chunk_size: int = 1024
        projector_bias: bool = True
        train_s2t: bool = True
        projector_ckpt: str = ""
        kd_warmup_steps: int = 300
        only_t2s_ce: bool = False

        # Soft-DTW sequence representation alignment
        dtw_rate: float = 0.2
        dtw_gamma: float = 2.0
        dtw_gamma_start: float = 2.0
        dtw_gamma_end: float = 0.8
        dtw_gamma_steps: int = 0
        dtw_warmup_steps: int = 0

        # DWA Dynamic Window / Soft Banding
        dtw_band_width: float = 5.0
        dtw_band_penalty: float = 1.0
        dtw_band_center_blend: float = 0.7
        dtw_band_entropy_coef: float = 2.0
        dtw_band_warmup_steps: int = 0
        dtw_band_source: Literal["cma", "sdtw"] = "cma"

        # Hidden & embedding representation weights
        dtw_hidden_weight: float = 1.0
        dtw_embed_weight: float = 1.0
        dtw_separate_hidden_projector: bool = False
        dtw_reduction: Literal["token_sum", "seq_sum", "mean"] = "token_sum"
        slice_mode: Literal["prefix", "masked"] = "prefix"
        dtw_backend: Literal["antidiag", "numba", "cuda"] = "antidiag"

    cfg: Config

    def __init__(self, cfg: BaseConfig | None = None) -> None:
        super().__init__(cfg)
        self.dtw = SoftDTW(
            gamma=self.cfg.dtw_gamma,
            backend=self.cfg.dtw_backend,
        )
        self.last_align: torch.Tensor | None = None
        self._step: int = 0

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        if info.teacher_hidden is None:
            raise ValueError("dwa_kd requires teacher_hidden")
        d_s, d_t = info.student_hidden, info.teacher_hidden

        self.register_aux_module(
            modules,
            "query",
            DSLProjector(DSLProjector.Config(structure="2s-2t", s=d_s, t=d_t, bias=self.cfg.projector_bias)),
            group="projector",
        )
        self.register_aux_module(
            modules,
            "s2t",
            DSLProjector(DSLProjector.Config(structure="s-t", s=d_s, t=d_t, bias=self.cfg.projector_bias)),
            group="projector",
            frozen=not self.cfg.train_s2t,
        )
        self.register_aux_module(
            modules,
            "t2s",
            DSLProjector(DSLProjector.Config(structure="t-s", s=d_s, t=d_t, bias=self.cfg.projector_bias)),
            group="projector",
        )
        self.register_aux_module(
            modules,
            "dtw_embed_t2s",
            DSLProjector(DSLProjector.Config(structure="t-s", s=d_s, t=d_t, bias=self.cfg.projector_bias)),
            group="projector",
        )
        if self.cfg.dtw_separate_hidden_projector:
            self.register_aux_module(
                modules,
                "dtw_hidden_t2s",
                DSLProjector(DSLProjector.Config(structure="t-s", s=d_s, t=d_t, bias=self.cfg.projector_bias)),
                group="projector",
            )

    def prepare(self, ctx: PrepareContext) -> None:
        _load_projector_checkpoint(self, getattr(ctx, "rt", None))

    def _calculate_alignment_loss(
        self,
        student_embs: torch.Tensor,
        teacher_embs: torch.Tensor,
        student_mask: torch.Tensor,
        teacher_mask: torch.Tensor,
        align_matrix: torch.Tensor | None = None,
        band_warmup_scale: float = 1.0,
    ) -> torch.Tensor:
        """Calculate pairwise representation Soft-DTW loss across batch sequences."""
        batch_size = student_embs.size(0)
        total_loss = torch.tensor(0.0, device=student_embs.device, requires_grad=True)
        non_empty_pairs = 0

        for i in range(batch_size):
            s_len = int(student_mask[i].sum().item())
            t_len = int(teacher_mask[i].sum().item())

            if s_len <= 0 or t_len <= 0:
                continue

            if self.cfg.slice_mode == "masked":
                s_seq = student_embs[i][student_mask[i].bool()]
                t_seq = teacher_embs[i][teacher_mask[i].bool()]
            else:
                s_seq = student_embs[i, :s_len, :]
                t_seq = teacher_embs[i, :t_len, :]

            if (
                torch.isnan(s_seq).any()
                or torch.isinf(s_seq).any()
                or torch.isnan(t_seq).any()
                or torch.isinf(t_seq).any()
            ):
                continue

            pair_loss, _ = soft_dtw_divergence(
                self.dtw,
                s_seq=s_seq,
                t_seq=t_seq,
                align_matrix=align_matrix[i] if align_matrix is not None else None,
                band_width=self.cfg.dtw_band_width,
                band_penalty=self.cfg.dtw_band_penalty,
                center_blend=self.cfg.dtw_band_center_blend,
                entropy_coef=self.cfg.dtw_band_entropy_coef,
                band_warmup_scale=band_warmup_scale,
                band_source=self.cfg.dtw_band_source,
            )

            if torch.isnan(pair_loss) or torch.isinf(pair_loss):
                continue

            total_loss = total_loss + pair_loss
            non_empty_pairs += 1

        if non_empty_pairs == 0:
            return torch.tensor(0.0, device=student_embs.device, requires_grad=True)

        return total_loss

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.student.logits is None or ctx.teacher is None or ctx.teacher.logits is None:
            raise ValueError("dwa_kd requires student and teacher logits views")
        s_logits, t_logits = ctx.student.logits, ctx.teacher.logits
        stu_emb, tea_emb = _get_embeddings(ctx)
        stu_head, tea_head = _get_heads(ctx)

        pm = ctx.supervised_mask()
        labels = ctx.batch["student.labels"]
        t_labels = ctx.batch.get("teacher.labels", labels)
        tpm = (t_labels != -100).float() if "teacher.labels" in ctx.batch else pm

        ids_s = ctx.batch["student.input_ids"]
        ids_t = ctx.batch.get("teacher.input_ids", ids_s)
        stu_index, norm_tea_index, norm_tea_tgt = _cma_embed_indices(ids_s, ids_t, pm, tpm, stu_emb, tea_emb)

        h_s_raw = ctx.student.hidden(-1)
        h_t_raw = ctx.teacher.side.hidden(-1) if (ctx.teacher and ctx.teacher.side) else None
        if h_t_raw is None:
            raise ValueError("dwa_kd requires teacher hidden states")

        h_s = target_hidden(h_s_raw)
        h_t = target_hidden(h_t_raw)
        norm_h_t = h_t / (h_t.std() + 1e-9)

        aux = self.aux(ctx)
        q = aux.query(stu_index).float()
        k = norm_tea_index.float()
        v_s = aux.s2t(h_s).float()
        v_t = aux.t2s(norm_h_t + norm_tea_tgt).float()

        # Step tracking and warmup schedules
        step = getattr(ctx.state, "step", self._step)
        self._step += 1

        if self.cfg.kd_warmup_steps > 0:
            kd_warmup_scale = min(1.0, float(step + 1) / float(self.cfg.kd_warmup_steps))
        else:
            kd_warmup_scale = 1.0

        if self.cfg.dtw_warmup_steps > 0:
            dtw_warmup_scale = min(1.0, float(step + 1) / float(self.cfg.dtw_warmup_steps))
        else:
            dtw_warmup_scale = 1.0

        if self.cfg.dtw_band_warmup_steps > 0:
            band_warmup_scale = min(1.0, float(step + 1) / float(self.cfg.dtw_band_warmup_steps))
        else:
            band_warmup_scale = 1.0

        # Dynamic DTW gamma annealing
        if self.cfg.dtw_gamma_steps > 0:
            progress = min(1.0, float(step + 1) / float(self.cfg.dtw_gamma_steps))
            current_gamma = self.cfg.dtw_gamma_start + (self.cfg.dtw_gamma_end - self.cfg.dtw_gamma_start) * progress
            self.dtw.gamma = current_gamma
        else:
            self.dtw.gamma = self.cfg.dtw_gamma

        # Entropy-based weights w_s (student uncertainty) and w_t (teacher certainty)
        s_full = target_logits_full(s_logits)
        stu_probs = torch.softmax(s_full, -1, dtype=torch.float32)
        stu_entropy = -(stu_probs * torch.log(stu_probs + 1e-9)).sum(-1)
        w_s = stu_entropy.detach() * pm
        valid_s = pm.sum(dim=1, keepdim=True).clamp_min(1.0)
        w_s = w_s * (valid_s / (w_s.sum(dim=1, keepdim=True) + 1e-9))

        t_full = target_logits_full(t_logits)
        tea_probs = torch.softmax(t_full, -1, dtype=torch.float32)
        tea_entropy = -(tea_probs * torch.log(tea_probs + 1e-9)).sum(-1)
        max_entropy = math.log(t_full.size(-1))
        tea_certainty = (1.0 - (tea_entropy / (max_entropy + 1e-9))).clamp(min=0.0, max=1.0)
        w_t = tea_certainty.detach() * tpm
        valid_t = tpm.sum(dim=1, keepdim=True).clamp_min(1.0)
        w_t = w_t * (valid_t / (w_t.sum(dim=1, keepdim=True) + 1e-9))

        # CMA cross-attention alignment
        align = (q @ k.transpose(-1, -2)) / math.sqrt(2 * h_t.shape[-1])
        align_mask = pm.unsqueeze(-1) * tpm.unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000.0)
        t2s_weight = torch.softmax(align, dim=-1)
        self.last_align = t2s_weight.detach()

        # Teacher-to-student mapping and logits
        t2s_h = (t2s_weight @ v_t).to(h_s.dtype)
        t2s_view = view_from_target_hidden(t2s_h, DetachedLinear(stu_head))
        t2s_ce = -(t2s_view.target_logp(labels, chunk_size=self.cfg.chunk_size) * pm).sum()
        acc = (_view_argmax(t2s_view, self.cfg.chunk_size) == labels) * pm

        terms: dict[str, LossTerm] = {"t2s_ce": LossTerm(t2s_ce, "token_sum")}

        # Gated student-space divergence and teacher-space forward KL
        if not self.cfg.only_t2s_ce:
            t2s_probs = torch.softmax(t2s_view.full(), dim=-1, dtype=torch.float32)
            t2s_conf = t2s_probs.max(dim=-1)[0]
            kd_gate = t2s_conf * kd_warmup_scale * pm

            div_tea = self._make_divergence(use_tea_temp=True, tea_temp=self.cfg.teacher_temperature)
            t2s_kd_vec = div_tea(s_full, t2s_view.full().detach())
            t2s_kd_weighted = (t2s_kd_vec * kd_gate * w_s).sum()

            s2t_weight = torch.softmax(align.transpose(-1, -2), dim=-1)
            s2t_h = (s2t_weight @ v_s).to(h_s.dtype)
            s2t_view = view_from_target_hidden(s2t_h, DetachedLinear(tea_head))
            fkl = functools.partial(
                vd_divergence, "forward_kl", kd_temp=self.cfg.temperature, tea_temp=1.0, use_tea_temp=False
            )
            s2t_kd_vec = fkl(s2t_view.full(), t_full)
            s2t_kd_weighted = (s2t_kd_vec * tpm * w_t).sum()

            terms["t2s_kd"] = LossTerm(t2s_kd_weighted, "token_sum")
            terms["s2t_kd"] = LossTerm(s2t_kd_weighted, "token_sum")

        # Soft-DTW sequence representation alignment
        loss_hidden = torch.tensor(0.0, device=pm.device)
        loss_embed = torch.tensor(0.0, device=pm.device)
        total_dtw_loss = torch.tensor(0.0, device=pm.device)
        weighted_dtw_loss = torch.tensor(0.0, device=pm.device)

        if self.cfg.dtw_rate > 0:
            formal_tgt_s = torch.where(pm.bool(), labels, torch.zeros_like(labels))
            stu_target_embeds = stu_emb(formal_tgt_s)

            formal_tgt_t = torch.where(tpm.bool(), t_labels, torch.zeros_like(t_labels))
            tea_target_embeds = tea_emb(formal_tgt_t).detach()

            proj_tea_embeds = aux.dtw_embed_t2s(tea_target_embeds)
            loss_embed = self._calculate_alignment_loss(
                stu_target_embeds,
                proj_tea_embeds,
                pm,
                tpm,
                align_matrix=self.last_align,
                band_warmup_scale=band_warmup_scale,
            )

            if self.cfg.dtw_separate_hidden_projector:
                proj_tea_hiddens = aux.dtw_hidden_t2s(h_t_raw)
            else:
                proj_tea_hiddens = aux.t2s(h_t_raw)

            loss_hidden = self._calculate_alignment_loss(
                h_s_raw,
                proj_tea_hiddens,
                pm,
                tpm,
                align_matrix=self.last_align,
                band_warmup_scale=band_warmup_scale,
            )

            total_dtw_loss = (
                self.cfg.dtw_hidden_weight * loss_hidden + self.cfg.dtw_embed_weight * loss_embed
            )
            weighted_dtw_loss = total_dtw_loss * dtw_warmup_scale
            terms["dtw"] = LossTerm(self.cfg.dtw_rate * weighted_dtw_loss, self.cfg.dtw_reduction)

        top1_prob = t2s_view.topk(1, chunk_size=self.cfg.chunk_size)[1].squeeze(-1).exp()
        metrics = {
            "t2s_acc": acc.sum() / pm.sum().clamp(min=1),
            "max_t2s_prob": (top1_prob * pm).sum() / pm.sum().clamp(min=1),
            "student_entropy_mean": (stu_entropy * pm).sum() / pm.sum().clamp(min=1),
            "student_entropy_std": stu_entropy.std(),
            "dtw_loss": total_dtw_loss.detach(),
            "dtw_hidden_loss": loss_hidden.detach(),
            "dtw_embed_loss": loss_embed.detach(),
            "weighted_dtw_loss": weighted_dtw_loss.detach(),
        }
        return LossOutput(terms=terms, metrics=metrics)
