"""Dual-Space Knowledge Distillation (DSKD) criteria: DSKD (same-tokenizer) and DSKD-CMA (cross-tokenizer)."""

from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn

from baselines.dskd.common import (
    _compute_s2t_hidden,
    _DualSpaceBase,
    _get_embeddings,
    _get_heads,
    _init_pinv_and_buffers,
    _load_projector_checkpoint,
    _view_argmax,
)
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import LossContext, LossOutput, LossTerm, ModelPairInfo, PrepareContext
from lmkd.losses.cross_tokenizer._common import (
    DetachedLinear,
    target_hidden,
    vd_divergence,
    view_from_target_hidden,
)
from lmkd.models.logits_view import LogitsView, chunked_token_divergence
from lmkd.models.modules import DSLProjector, ModuleRegistry


@CRITERIONS.register(
    "dskd", source="white-box-distillation/cross-tokenizer/DSKDv2-main/code/criterions/dual_space_kd.py"
)
class DSKD(_DualSpaceBase):
    """Dual-space KD: t2s/s2t projectors, KD in both student and teacher vocab spaces (same tokenizer).

    Formula:
        t2s_logits = (h_t @ W_t2s) @ W_S_head^T, s2t_logits = (h_s @ W_s2t) @ W_T_head^T
        L = div(p_s, p_t2s) + div(p_s2t, p_t)

    Differences from original:
        - Target-aligned LogitsView representation avoiding [B,T,V] materialization.
        - Fixed s2t overlap mismatch when vocabularies differ.
    """

    class Config(BaseConfig):
        temperature: float = 3.0
        teacher_temperature: float = 1.0
        kd_objective: str = "forward_kl"
        skew_lambda: float = 0.1
        adaptive_kl_alpha: float = 0.5
        chunk_size: int = 1024
        projector_bias: bool = True
        train_s2t: bool = True
        projector_ckpt: str = ""
        only_stu_kd: bool = False
        only_tea_kd: bool = False
        topk_vocab: int = -1
        init_t2s_pinv: bool = False
        analytic_s2t: bool = False

    cfg: Config

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        if info.teacher_hidden is None:
            raise ValueError("dskd requires teacher_hidden")
        d_s, d_t = info.student_hidden, info.teacher_hidden
        self.register_aux_module(
            modules, "t2s", nn.Linear(d_t, d_s, bias=self.cfg.projector_bias), group="projector"
        )
        if not self.cfg.analytic_s2t:
            self.register_aux_module(
                modules,
                "s2t",
                nn.Linear(d_s, d_t, bias=self.cfg.projector_bias),
                group="projector",
                frozen=not self.cfg.train_s2t,
            )

    def prepare(self, ctx: PrepareContext) -> None:
        _init_pinv_and_buffers(self, ctx)
        _load_projector_checkpoint(self, getattr(ctx, "rt", None))

    def forward(self, ctx: LossContext) -> LossOutput:
        s_ids = ctx.batch["student.input_ids"]
        t_ids = ctx.batch.get("teacher.input_ids", s_ids)
        if s_ids.shape != t_ids.shape:
            raise ValueError(
                "dskd requires same tokenizer / identical input shapes; use dskd_cma or dskdv2_eta for cross-tokenizer setups"
            )
        if ctx.student.logits is None or ctx.teacher is None or ctx.teacher.logits is None:
            raise ValueError("dskd requires student and teacher logits views")
        s_logits, t_logits = ctx.student.logits, ctx.teacher.logits
        stu_head, tea_head = _get_heads(ctx)
        h_s = ctx.student.hidden(-1)
        h_t = ctx.teacher.side.hidden(-1) if (ctx.teacher and ctx.teacher.side) else None
        if h_t is None:
            raise ValueError("dskd requires teacher hidden states")

        t2s_view = LogitsView(self.aux(ctx).t2s(h_t), DetachedLinear(stu_head))
        s2t_h_pos = _compute_s2t_hidden(self, ctx, h_s, stu_head, tea_head)
        s2t_view = LogitsView(s2t_h_pos, tea_head)

        pad_mask = ctx.supervised_mask()
        labels = ctx.batch["student.labels"]
        t_labels = ctx.batch.get("teacher.labels", labels)
        t_pad_mask = (t_labels != -100).float() if "teacher.labels" in ctx.batch else pad_mask

        div = self._make_divergence()
        t2s_ce = -(t2s_view.target_logp(labels, chunk_size=self.cfg.chunk_size) * pad_mask).sum()
        t2s_kd = (
            chunked_token_divergence(s_logits, t2s_view, div, mask=pad_mask, chunk_size=self.cfg.chunk_size)
            * pad_mask
        ).sum()
        s2t_kd = (
            chunked_token_divergence(s2t_view, t_logits, div, mask=t_pad_mask, chunk_size=self.cfg.chunk_size)
            * t_pad_mask
        ).sum()

        terms: dict[str, LossTerm] = {}
        if not self.cfg.only_tea_kd:
            terms["t2s_ce"] = LossTerm(t2s_ce, "token_sum")
            terms["t2s_kd"] = LossTerm(t2s_kd, "token_sum")
        if not self.cfg.only_stu_kd:
            terms["s2t_kd"] = LossTerm(s2t_kd, "token_sum")

        t_preds = _view_argmax(t_logits, self.cfg.chunk_size)
        t2s_preds = _view_argmax(t2s_view, self.cfg.chunk_size)
        s_preds = _view_argmax(s_logits, self.cfg.chunk_size)
        s2t_preds = _view_argmax(s2t_view, self.cfg.chunk_size)
        labels = ctx.batch["student.labels"]
        n_sup = pad_mask.sum().clamp(min=1)
        metrics = {
            "t2s_agreement": ((t2s_preds == t_preds) * pad_mask).sum() / n_sup,
            "s2t_agreement": ((s2t_preds == s_preds) * pad_mask).sum() / n_sup,
            "t_acc": ((t_preds == labels) * pad_mask).sum() / n_sup,
            "t2s_acc": ((t2s_preds == labels) * pad_mask).sum() / n_sup,
        }
        return LossOutput(terms=terms, metrics=metrics)


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
    return stu_index, tea_index / tea_index.std(), tea_tgt_emb / tea_tgt_emb.std()


@CRITERIONS.register("dskd_cma", source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/DSKD.py:61-160")
class DSKDCMA(_DualSpaceBase):
    """DSKD with cross-model attention (CMA) between student and teacher embedding spaces.

    Formula:
        stu_index = [E_s(in_s), E_s(tgt_s)], tea_index = [E_t(in_t), E_t(tgt_t)]
        align = q(stu_index) @ k(tea_index)^T / sqrt(2*D_t) + (1 - mask) * (-1e5)
        t2s_h = softmax(align) @ t2s(norm(h_t) + norm(tea_tgt))
        s2t_h = softmax(align^T) @ s2t(h_s)

    Differences from original:
        - Target-aligned LogitsView representation avoiding [B,T,V] materialization.
        - Scaled by global token norms instead of local student count.
    """

    class Config(BaseConfig):
        temperature: float = 3.0
        teacher_temperature: float = 1.0
        kd_objective: str = "forward_kl"
        skew_lambda: float = 0.1
        adaptive_kl_alpha: float = 0.5
        chunk_size: int = 1024
        projector_bias: bool = True
        train_s2t: bool = True
        projector_ckpt: str = ""
        only_stu_kd: bool = False
        only_tea_kd: bool = False
        only_t2s_ce: bool = False
        topk_vocab: int = -1

    cfg: Config

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        if info.teacher_hidden is None:
            raise ValueError("dskd_cma requires teacher_hidden")
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
        )
        self.register_aux_module(
            modules,
            "t2s",
            DSLProjector(DSLProjector.Config(structure="t-s", s=d_s, t=d_t, bias=self.cfg.projector_bias)),
            group="projector",
        )

    def prepare(self, ctx: PrepareContext) -> None:
        _load_projector_checkpoint(self, getattr(ctx, "rt", None))

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.student.logits is None or ctx.teacher is None or ctx.teacher.logits is None:
            raise ValueError("dskd_cma requires student and teacher logits views")
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

        h_s = target_hidden(ctx.student.hidden(-1))
        h_t_raw = ctx.teacher.side.hidden(-1) if (ctx.teacher and ctx.teacher.side) else None
        if h_t_raw is None:
            raise ValueError("dskd_cma requires teacher hidden states")
        h_t = target_hidden(h_t_raw)
        norm_h_t = h_t / h_t.std()

        aux = self.aux(ctx)
        q = aux.query(stu_index).float()
        k = norm_tea_index.float()
        v_s = aux.s2t(h_s).float()
        v_t = aux.t2s(norm_h_t + norm_tea_tgt).float()

        align = (q @ k.transpose(-1, -2)) / math.sqrt(2 * h_t.shape[-1])
        align_mask = pm.unsqueeze(-1) * tpm.unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000.0)

        t2s_h = (torch.softmax(align, dim=-1) @ v_t).to(h_s.dtype)
        t2s_view = view_from_target_hidden(t2s_h, DetachedLinear(stu_head))
        t2s_ce = -(t2s_view.target_logp(labels, chunk_size=self.cfg.chunk_size) * pm).sum()
        acc = (_view_argmax(t2s_view, self.cfg.chunk_size) == labels) * pm

        terms: dict[str, LossTerm] = {"t2s_ce": LossTerm(t2s_ce, "token_sum")}
        if not self.cfg.only_t2s_ce:
            div_tea = self._make_divergence(use_tea_temp=True, tea_temp=self.cfg.teacher_temperature)
            t2s_kd = (
                chunked_token_divergence(s_logits, t2s_view, div_tea, mask=pm * acc, chunk_size=self.cfg.chunk_size)
                * pm
                * acc
            ).sum()
            s2t_h = (torch.softmax(align.transpose(-1, -2), dim=-1) @ v_s).to(h_s.dtype)
            s2t_view = view_from_target_hidden(s2t_h, DetachedLinear(tea_head))
            fkl = functools.partial(
                vd_divergence, "forward_kl", kd_temp=self.cfg.temperature, tea_temp=1.0, use_tea_temp=False
            )
            s2t_kd = (
                chunked_token_divergence(s2t_view, t_logits, fkl, mask=tpm, chunk_size=self.cfg.chunk_size) * tpm
            ).sum()
            terms["t2s_kd"] = LossTerm(t2s_kd, "token_sum")
            terms["s2t_kd"] = LossTerm(s2t_kd, "token_sum")

        top1_prob = t2s_view.topk(1, chunk_size=self.cfg.chunk_size)[1].squeeze(-1).exp()
        metrics = {
            "t2s_acc": acc.sum() / pm.sum().clamp(min=1),
            "max_t2s_prob": (top1_prob * pm).sum() / pm.sum().clamp(min=1),
        }
        return LossOutput(terms=terms, metrics=metrics)
