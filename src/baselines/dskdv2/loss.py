"""DSKDv2 criteria: DSKDv2 (same-tokenizer) and DSKDv2-ETA (cross-tokenizer)."""

from __future__ import annotations

from typing import ClassVar, Literal

import torch
import torch.nn as nn

from baselines.dskd.common import (
    _compute_s2t_hidden,
    _DualSpaceBase,
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
    get_vocab_map,
    target_hidden,
    view_from_target_hidden,
)
from lmkd.models.logits_view import LogitsView, chunked_token_divergence
from lmkd.models.modules import ModuleRegistry


@CRITERIONS.register(
    "dskdv2", source="white-box-distillation/cross-tokenizer/DSKDv2-main/code/criterions/dual_space_kd_v2.py:57-130"
)
class DSKDv2(_DualSpaceBase):
    """DSKDv2: teacher argmax as label for t2s CE, agreement filtering, analytic s2t = W_S @ pinv(W_T).

    Formula:
        t2s_ce = -sum(log p_t2s(t_preds) * pad_mask)
        agree = (t2s_preds == t_preds) * pad_mask
        t2s_kd = (kd_tok * agree).sum() / agree.sum() if agreement <= thresh else (kd_tok * pad_mask).sum()
        s2t_hidden = h_s @ (W_S @ pinv(part_T))
        s2t_kd = sum(div(p_s2t, p_t) * t_pad_mask)

    Differences from original:
        - Fixed s2t overlap mismatch when vocabularies differ.
        - Dropped on-policy buggy indexing (line 179).
        - Agreement denominator uses local supervised count.
    """

    class Config(BaseConfig):
        temperature: float = 2.0
        teacher_temperature: float = 1.0
        kd_objective: str = "forward_kl"
        skew_lambda: float = 0.1
        adaptive_kl_alpha: float = 0.5
        chunk_size: int = 1024
        projector_bias: bool = False
        train_s2t: bool = True
        projector_ckpt: str = ""
        only_stu_kd: bool = False
        only_tea_kd: bool = False
        t2s_agreement: float = 0.0
        topk_vocab: int = -1
        init_t2s_pinv: bool = True
        analytic_s2t: bool = True

    cfg: Config

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        if info.teacher_hidden is None:
            raise ValueError("dskdv2 requires teacher_hidden")
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
            raise ValueError("dskdv2 requires same tokenizer; use dskd_cma or dskdv2_eta for cross-tokenizer setups")
        if ctx.student.logits is None or ctx.teacher is None or ctx.teacher.logits is None:
            raise ValueError("dskdv2 requires student and teacher logits views")
        s_logits, t_logits = ctx.student.logits, ctx.teacher.logits
        stu_head, tea_head = _get_heads(ctx)
        h_s = ctx.student.hidden(-1)
        h_t = ctx.teacher.side.hidden(-1) if (ctx.teacher and ctx.teacher.side) else None
        if h_t is None:
            raise ValueError("dskdv2 requires teacher hidden states")

        t2s_view = LogitsView(self.aux(ctx).t2s(h_t), DetachedLinear(stu_head))
        s2t_h_pos = _compute_s2t_hidden(self, ctx, h_s, stu_head, tea_head)
        s2t_view = LogitsView(s2t_h_pos, tea_head)

        pad_mask = ctx.supervised_mask()
        labels = ctx.batch["student.labels"]
        t_labels = ctx.batch.get("teacher.labels", labels)
        t_pad_mask = (t_labels != -100).float() if "teacher.labels" in ctx.batch else pad_mask

        t_preds = _view_argmax(t_logits, self.cfg.chunk_size)
        t_preds = torch.where(t_pad_mask.bool(), t_preds, t_labels)
        t_preds_masked = torch.where(t_pad_mask.bool(), t_preds, torch.tensor(-100, device=t_preds.device))
        t2s_ce = -(t2s_view.target_logp(t_preds_masked, chunk_size=self.cfg.chunk_size) * pad_mask).sum()

        t2s_preds = _view_argmax(t2s_view, self.cfg.chunk_size)
        agree = (t2s_preds == t_preds) * pad_mask
        n_sup = pad_mask.sum().clamp(min=1)
        agreement = agree.sum() / n_sup

        div = self._make_divergence()
        kd_tok = chunked_token_divergence(s_logits, t2s_view, div, mask=pad_mask, chunk_size=self.cfg.chunk_size)
        if float(agreement) <= self.cfg.t2s_agreement:
            t2s_kd = (kd_tok * agree).sum() / max(1e-3, float(agree.sum()))
            t2s_kd_term = LossTerm(t2s_kd, "mean")
            t2s_kd_branch = 0.0
        else:
            t2s_kd = (kd_tok * pad_mask).sum()
            t2s_kd_term = LossTerm(t2s_kd, "token_sum")
            t2s_kd_branch = 1.0

        s2t_kd = (
            chunked_token_divergence(s2t_view, t_logits, div, mask=t_pad_mask, chunk_size=self.cfg.chunk_size)
            * t_pad_mask
        ).sum()

        terms: dict[str, LossTerm] = {}
        if not self.cfg.only_tea_kd:
            terms["t2s_ce"] = LossTerm(t2s_ce, "token_sum")
            terms["t2s_kd"] = t2s_kd_term
        if not self.cfg.only_stu_kd:
            terms["s2t_kd"] = LossTerm(s2t_kd, "token_sum")

        s_preds = _view_argmax(s_logits, self.cfg.chunk_size)
        s2t_preds = _view_argmax(s2t_view, self.cfg.chunk_size)
        metrics = {
            "t2s_agreement": agreement,
            "s2t_agreement": ((s2t_preds == s_preds) * pad_mask).sum() / n_sup,
            "t_acc": ((t_preds == labels) * pad_mask).sum() / n_sup,
            "t2s_acc": ((t2s_preds == labels) * pad_mask).sum() / n_sup,
            "t2s_kd_branch": torch.tensor(t2s_kd_branch, device=pad_mask.device),
        }
        return LossOutput(terms=terms, metrics=metrics)


def _eta_align_hiddens(
    criterion: DSKDv2ETA,
    ctx: LossContext,
    tea_v: torch.Tensor,
    stu_v: torch.Tensor,
    t_preds: torch.Tensor,
    m_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    tp = t_preds.gather(1, m_s.clamp(min=0))
    vm = get_vocab_map(ctx)
    tea2stu_exact = vm.tea2stu_exact.to(tp.device)
    mapped = tea2stu_exact[tp.clamp(min=0)]
    if criterion.tea_eos_id is not None and criterion.stu_eos_id is not None:
        mapped = torch.where(tp == criterion.tea_eos_id, criterion.stu_eos_id, mapped)
    ok = (m_s >= 0) & (mapped >= 0) & (tp >= 0)
    t_preds_as_label = torch.where(ok, mapped, torch.tensor(-100, device=tp.device))
    n_align = max(1e-3, float(ok.sum()))

    b, t_s, d_s = tea_v.shape[0], m_s.shape[1], tea_v.shape[2]
    t_t, d_t = t_preds.shape[1], stu_v.shape[2]
    tea_v_gathered = tea_v.gather(1, m_s.clamp(min=0).unsqueeze(-1).expand(-1, -1, d_s))
    t2s_h_align = torch.zeros(b, t_s, d_s, device=tea_v.device, dtype=tea_v.dtype)
    t2s_h_align[ok] = tea_v_gathered[ok]

    b_idx, s_idx = torch.where(ok)
    t_idx = m_s[b_idx, s_idx]
    s2t_h_align = torch.zeros(b, t_t, d_t, device=stu_v.device, dtype=stu_v.dtype)
    s2t_h_align[b_idx, t_idx] = stu_v[b_idx, s_idx]
    return t_preds_as_label, t2s_h_align, s2t_h_align, ok, n_align


@CRITERIONS.register(
    "dskdv2_eta",
    source="white-box-distillation/cross-tokenizer/DSKDv2-main/code/criterions/dual_space_kd_v2_with_exact_token_alignment.py:96-238",
)
class DSKDv2ETA(_DualSpaceBase):
    """DSKDv2 with exact token alignment (ETA) across different tokenizers.

    Formula:
        Uses align.eta_t_for_s to pair student and teacher tokens.
        t_preds_as_label = mapped teacher preds where ETA pair is valid and vocabulary overlap exists.
        t2s_h_align = gathered tea_v at paired indices.
        s2t_h_align = scattered stu_v at paired indices.

    Differences from original:
        - Vectorized scatter (index_put) without Python loop.
        - Configurable normalization: 'local' (matching original) or 'global' (standard token_sum).
        - Fixed s2t overlap mismatch when vocabularies differ.
    """

    requires: ClassVar[tuple[str, ...]] = ("align.eta_t_for_s", "align.eta_s_for_t")

    class Config(BaseConfig):
        temperature: float = 2.0
        teacher_temperature: float = 1.0
        kd_objective: str = "forward_kl"
        skew_lambda: float = 0.1
        adaptive_kl_alpha: float = 0.5
        chunk_size: int = 1024
        projector_bias: bool = False
        train_s2t: bool = True
        projector_ckpt: str = ""
        only_stu_kd: bool = False
        only_tea_kd: bool = False
        t2s_agreement: float = 0.0
        topk_vocab: int = -1
        init_t2s_pinv: bool = True
        analytic_s2t: bool = True
        normalize: str = "local"

    cfg: Config

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        if info.teacher_hidden is None:
            raise ValueError("dskdv2_eta requires teacher_hidden")
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
        if ctx.student.logits is None or ctx.teacher is None or ctx.teacher.logits is None:
            raise ValueError("dskdv2_eta requires student and teacher logits views")
        s_logits, t_logits = ctx.student.logits, ctx.teacher.logits
        stu_head, tea_head = _get_heads(ctx)
        h_s = target_hidden(ctx.student.hidden(-1))
        h_t_raw = ctx.teacher.side.hidden(-1) if (ctx.teacher and ctx.teacher.side) else None
        if h_t_raw is None:
            raise ValueError("dskdv2_eta requires teacher hidden states")
        h_t = target_hidden(h_t_raw)

        tea_v = self.aux(ctx).t2s(h_t)
        stu_v = _compute_s2t_hidden(self, ctx, h_s, stu_head, tea_head)

        pad_mask = ctx.supervised_mask()
        labels = ctx.batch["student.labels"]
        t_labels = ctx.batch.get("teacher.labels", labels)
        t_pad_mask = (t_labels != -100).float() if "teacher.labels" in ctx.batch else pad_mask
        m_s = ctx.batch["align.eta_t_for_s"]

        t_preds = _view_argmax(t_logits, self.cfg.chunk_size)
        t_preds = torch.where(t_pad_mask.bool(), t_preds, t_labels)
        t_preds_as_label, t2s_h_align, s2t_h_align, ok, n_align = _eta_align_hiddens(
            self, ctx, tea_v, stu_v, t_preds, m_s
        )

        t2s_view = view_from_target_hidden(t2s_h_align, DetachedLinear(stu_head))
        s2t_view = view_from_target_hidden(s2t_h_align, tea_head)

        agree = _view_argmax(t2s_view, self.cfg.chunk_size) == t_preds_as_label
        agreement = (agree & ok).sum() / n_align

        div = self._make_divergence()
        kd_tok = chunked_token_divergence(s_logits, t2s_view, div, mask=ok.float(), chunk_size=self.cfg.chunk_size)
        rows = s2t_h_align.ne(0).any(-1)
        s2t_kd_raw = (
            chunked_token_divergence(s2t_view, t_logits, div, mask=rows, chunk_size=self.cfg.chunk_size) * rows
        ).sum()

        red: Literal["mean", "token_sum"] = "mean" if self.cfg.normalize == "local" else "token_sum"
        if self.cfg.normalize == "local":
            t2s_ce = -(t2s_view.target_logp(t_preds_as_label, chunk_size=self.cfg.chunk_size)).sum() / n_align
            if float(agreement) <= self.cfg.t2s_agreement:
                t2s_kd = (kd_tok * agree).sum() / max(1e-3, float(agree.sum()))
            else:
                t2s_kd = (kd_tok * ok).sum() / n_align
            s2t_kd = s2t_kd_raw / max(float(rows.sum()), 1e-8)
        else:
            t2s_ce = -(t2s_view.target_logp(t_preds_as_label, chunk_size=self.cfg.chunk_size)).sum()
            t2s_kd = (kd_tok * agree).sum() if float(agreement) <= self.cfg.t2s_agreement else (kd_tok * ok).sum()
            s2t_kd = s2t_kd_raw

        terms: dict[str, LossTerm] = {}
        if not self.cfg.only_tea_kd:
            terms["t2s_ce"] = LossTerm(t2s_ce, red)
            terms["t2s_kd"] = LossTerm(t2s_kd, red)
        if not self.cfg.only_stu_kd:
            terms["s2t_kd"] = LossTerm(s2t_kd, red)

        metrics = {
            "align_ratio": ok.sum() / pad_mask.sum().clamp(min=1),
            "t2s_agreement": agreement,
            "t2s_acc": (_view_argmax(t2s_view, self.cfg.chunk_size).eq(labels) * pad_mask).sum()
            / pad_mask.sum().clamp(min=1),
        }
        return LossOutput(terms=terms, metrics=metrics)
