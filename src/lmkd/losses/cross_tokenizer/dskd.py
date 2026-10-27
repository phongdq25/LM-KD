"""Dual-Space Knowledge Distillation (DSKD) criteria family (T12).
Sources:
  - white-box-distillation/cross-tokenizer/DSKDv2-main/code/criterions/dual_space_kd.py
  - white-box-distillation/cross-tokenizer/DSKDv2-main/code/criterions/dual_space_kd_v2.py
  - white-box-distillation/cross-tokenizer/DSKDv2-main/code/criterions/dual_space_kd_v2_with_exact_token_alignment.py
  - white-box-distillation/cross-tokenizer/IMPACT/code/criterions/DSKD.py
"""
from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any, ClassVar, Literal, cast

import torch
import torch.nn as nn

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo, PrepareContext
from lmkd.losses.cross_tokenizer._common import (
    DetachedLinear,
    get_vocab_map,
    target_hidden,
    vd_divergence,
    view_from_target_hidden,
)
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.logits_view import LogitsView, chunked_token_divergence
from lmkd.models.modules import DSLProjector
from lmkd.teachers.signals import TeacherSignalSpec


def _view_argmax(view: LogitsView, chunk_size: int = 1024) -> torch.Tensor:
    """Argmax index [B, T] from a LogitsView without materializing full logits."""
    return view.topk(1, chunk_size=chunk_size)[0].squeeze(-1)


def _get_heads(ctx: LossContext) -> tuple[nn.Module, nn.Module]:
    """Retrieve student and teacher lm_head modules from runtime or logits views."""
    rt = getattr(ctx, "rt", None)
    stu_head: nn.Module | None = None
    if rt is not None and getattr(rt, "student_adapter", None) is not None and getattr(rt, "student_model", None) is not None:
        stu_head = rt.student_adapter.lm_head(rt.student_model)
    if stu_head is None and ctx.student.logits is not None:
        stu_head = getattr(ctx.student.logits, "_head", None)

    tea_head: nn.Module | None = None
    if rt is not None and getattr(rt, "teacher_adapter", None) is not None and getattr(rt, "teacher_model", None) is not None:
        tea_head = rt.teacher_adapter.lm_head(rt.teacher_model)
    if tea_head is None and ctx.teacher is not None and ctx.teacher.logits is not None:
        tea_head = getattr(ctx.teacher.logits, "_head", None)

    if stu_head is None or tea_head is None:
        raise ValueError("Could not resolve student and teacher lm_head modules")
    return stu_head, tea_head


def _get_embeddings(ctx: LossContext) -> tuple[nn.Module, nn.Module]:
    """Retrieve student and teacher embed_tokens modules for CMA."""
    rt = getattr(ctx, "rt", None)
    stu_emb = None
    if rt is not None and getattr(rt, "student_adapter", None) is not None and getattr(rt, "student_model", None) is not None:
        stu_emb = rt.student_adapter.embed_tokens(rt.student_model)
    tea_emb = None
    if rt is not None and getattr(rt, "teacher_adapter", None) is not None and getattr(rt, "teacher_model", None) is not None:
        tea_emb = rt.teacher_adapter.embed_tokens(rt.teacher_model)

    if stu_emb is None:
        stu_emb = getattr(ctx, "student_embed_tokens", None)
    if tea_emb is None:
        tea_emb = getattr(ctx, "teacher_embed_tokens", None)
    if stu_emb is None or tea_emb is None:
        raise ValueError("dskd_cma requires student and teacher embed_tokens modules")
    return stu_emb, tea_emb


class _DualSpaceBase(Criterion):
    """Base class providing shared feature requests, divergence setup, and state persistence."""
    needs_teacher: ClassVar[bool] = True

    def __init__(self, cfg: BaseConfig | None = None) -> None:
        super().__init__(cfg)
        self.overlap_s_part: torch.Tensor | None = None
        self.tea_eos_id: int | None = None
        self.stu_eos_id: int | None = None

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view", hidden=(-1,)),
            teacher=SideRequest(logits="view", hidden=(-1,)),
            teacher_signal=TeacherSignalSpec(kind="full"),
        )

    def _make_divergence(
        self,
        name: str | None = None,
        use_tea_temp: bool = False,
        tea_temp: float = 1.0,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        obj = name if name is not None else self.cfg.kd_objective
        return functools.partial(
            vd_divergence,
            obj,
            kd_temp=self.cfg.temperature,
            tea_temp=tea_temp if use_tea_temp else 1.0,
            use_tea_temp=use_tea_temp,
            skew_lambda=self.cfg.skew_lambda,
            adaptive_alpha=self.cfg.adaptive_kl_alpha,
        )

    def extra_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        pinv_buf: torch.Tensor | None = getattr(self, "part_T_pinv", None)
        if pinv_buf is not None:
            state["part_T_pinv"] = pinv_buf.detach().cpu()
        if self.overlap_s_part is not None:
            state["overlap_s_part"] = self.overlap_s_part.detach().cpu()
        return state

    def load_extra_state(self, state: dict[str, Any]) -> None:
        if "part_T_pinv" in state:
            self.register_buffer("part_T_pinv", state["part_T_pinv"], persistent=False)
        if "overlap_s_part" in state:
            self.overlap_s_part = state["overlap_s_part"]


def _init_pinv_and_buffers(criterion: _DualSpaceBase, ctx: PrepareContext) -> None:
    """Compute pseudo-inverse weights and buffers for t2s/s2t projectors."""
    rt = getattr(ctx, "rt", None)
    if rt is None or getattr(rt, "student_model", None) is None or getattr(rt, "teacher_model", None) is None:
        return
    stu_adapter = getattr(rt, "student_adapter", None)
    tea_adapter = getattr(rt, "teacher_adapter", None)
    if stu_adapter is None or tea_adapter is None:
        return
    stu_head = stu_adapter.lm_head(rt.student_model)
    tea_head = tea_adapter.lm_head(rt.teacher_model)
    if stu_head is None or tea_head is None:
        return

    stu_w: torch.Tensor = cast(Any, stu_head).weight
    tea_w: torch.Tensor = cast(Any, tea_head).weight
    W_S = stu_w.detach().transpose(0, 1)  # [D_s, V_s]
    W_T = tea_w.detach().transpose(0, 1)  # [D_t, V_t]
    topk_vocab = getattr(criterion.cfg, "topk_vocab", -1)
    vm = getattr(getattr(rt, "data_ctx", None), "vocab_map", None)

    same_vocab = vm.same_vocab if vm is not None else (W_S.shape[1] == W_T.shape[1])
    if same_vocab:
        part_S = W_S[:, :topk_vocab] if topk_vocab != -1 else W_S
        part_T = W_T[:, :topk_vocab] if topk_vocab != -1 else W_T
        criterion.overlap_s_part = None
    else:
        if vm is None and getattr(rt, "student_tokenizer", None) is not None and getattr(rt, "teacher_tokenizer", None) is not None:
            from lmkd.data.vocab_map import VocabMap
            vm = VocabMap.build(rt.student_tokenizer, rt.teacher_tokenizer)
        if vm is not None:
            overlap_s = vm.overlap_s.to(W_S.device)
            overlap_t = vm.overlap_t.to(W_T.device)
            part_S = W_S[:, overlap_s]
            part_T = W_T[:, overlap_t]
            criterion.overlap_s_part = overlap_s[:topk_vocab] if topk_vocab != -1 else overlap_s
        else:
            part_S = W_S[:, :topk_vocab] if topk_vocab != -1 else W_S
            part_T = W_T[:, :topk_vocab] if topk_vocab != -1 else W_T
            criterion.overlap_s_part = None

    if getattr(criterion.cfg, "init_t2s_pinv", False):
        init_t2s = (part_T.float() @ torch.linalg.pinv(part_S.float())).transpose(0, 1)
        if getattr(rt, "modules", None) is not None and f"{criterion.alias}__t2s" in rt.modules:
            t2s_mod = rt.modules.get(f"{criterion.alias}__t2s")
            for m in t2s_mod.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.copy_(init_t2s.to(m.weight.dtype))
                    if m.bias is not None:
                        m.bias.data.zero_()
                    break

    if getattr(criterion.cfg, "analytic_s2t", False):
        part_T_pinv = torch.linalg.pinv(part_T.float()).to(part_T.dtype)
        criterion.register_buffer("part_T_pinv", part_T_pinv, persistent=False)

    if hasattr(rt, "teacher_tokenizer") and rt.teacher_tokenizer is not None:
        criterion.tea_eos_id = getattr(rt.teacher_tokenizer, "eos_token_id", None)
    if hasattr(rt, "student_tokenizer") and rt.student_tokenizer is not None:
        criterion.stu_eos_id = getattr(rt.student_tokenizer, "eos_token_id", None)


def _load_projector_checkpoint(criterion: _DualSpaceBase, rt: Any) -> None:
    """Load pretrained aux module weights from checkpoint file."""
    ckpt_path = getattr(criterion.cfg, "projector_ckpt", "")
    if not ckpt_path:
        return
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    prefix = f"mods.{criterion.alias}__"
    matched = {k[len("mods."):]: v for k, v in state.items() if k.startswith(prefix)}
    if not matched:
        raise RuntimeError(f"no keys starting with {prefix!r} in {ckpt_path!r}")
    for mod_key, tensor in matched.items():
        parts = mod_key.split(".")
        mod_name = parts[0]
        param_name = ".".join(parts[1:])
        target_mod = rt.modules.get(mod_name)
        p = target_mod.get_parameter(param_name)
        p.data.copy_(tensor.to(p.dtype))


def _compute_s2t_hidden(
    criterion: _DualSpaceBase,
    ctx: LossContext,
    h_s: torch.Tensor,
    stu_head: nn.Module,
    tea_head: nn.Module,
) -> torch.Tensor:
    """Calculate s2t projected hidden representation (analytic or learned module)."""
    if getattr(criterion.cfg, "analytic_s2t", False):
        if not hasattr(criterion, "part_T_pinv") or getattr(criterion, "part_T_pinv", None) is None:
            _init_pinv_and_buffers(criterion, PrepareContext(rt=getattr(ctx, "rt", None)))
        stu_w: torch.Tensor = cast(Any, stu_head).weight
        W_S_curr = stu_w.detach().transpose(0, 1)  # [D_s, V_s]
        if criterion.overlap_s_part is not None:
            W_S_curr = W_S_curr[:, criterion.overlap_s_part.to(W_S_curr.device)]
        elif getattr(criterion.cfg, "topk_vocab", -1) != -1:
            W_S_curr = W_S_curr[:, :criterion.cfg.topk_vocab]
        pinv_buf: torch.Tensor = cast(Any, criterion).part_T_pinv
        pinv_mat = pinv_buf.to(W_S_curr.device).float()
        s2t_mat = (W_S_curr.float() @ pinv_mat).to(h_s.dtype)
        return h_s @ s2t_mat
    s2t_mod = ctx.modules.get(f"{criterion.alias}__s2t")
    res: torch.Tensor = s2t_mod(h_s)
    return res


@CRITERIONS.register("dskd", source="white-box-distillation/cross-tokenizer/DSKDv2-main/code/criterions/dual_space_kd.py")
class DSKD(_DualSpaceBase):
    """Dual-space KD: t2s/s2t projectors, KD in both student and teacher vocab spaces (same tokenizer).

    Formula:
        t2s_logits = t2s(h_t) @ W_S.detach().T
        t2s_ce = -sum(log p_t2s(gold_labels) * pad_mask)
        t2s_kd = sum(div(p_s, p_t2s) * pad_mask)
        s2t_logits = tea_head(s2t(h_s))
        s2t_kd = sum(div(p_s2t, p_t) * t_pad_mask)

    Differences from original:
        - Scaled by global token norms instead of local loss_denom.
        - train_s2t flag enables optimizing s2t (original omitted s2t from optimizer).
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

    def build_modules(self, info: ModelPairInfo, modules: Any) -> None:
        if info.teacher_hidden is None:
            raise ValueError("dskd requires teacher_hidden")
        d_s, d_t = info.student_hidden, info.teacher_hidden
        modules.add(f"{self.alias}__t2s", nn.Linear(d_t, d_s, bias=self.cfg.projector_bias), group="projector", owner=self.alias)
        modules.add(f"{self.alias}__s2t", nn.Linear(d_s, d_t, bias=self.cfg.projector_bias), group="projector",
                    frozen=not self.cfg.train_s2t, owner=self.alias)

    def prepare(self, ctx: PrepareContext) -> None:
        _init_pinv_and_buffers(self, ctx)
        _load_projector_checkpoint(self, getattr(ctx, "rt", None))

    def forward(self, ctx: LossContext) -> LossOutput:
        s_ids = ctx.batch["student.input_ids"]
        t_ids = ctx.batch.get("teacher.input_ids", s_ids)
        if s_ids.shape != t_ids.shape:
            raise ValueError("dskd requires same tokenizer / identical input shapes; use dskd_cma or dskdv2_eta for cross-tokenizer setups")
        if ctx.student.logits is None or ctx.teacher is None or ctx.teacher.logits is None:
            raise ValueError("dskd requires student and teacher logits views")
        s_logits, t_logits = ctx.student.logits, ctx.teacher.logits
        stu_head, tea_head = _get_heads(ctx)
        h_s = ctx.student.hidden(-1)
        h_t = ctx.teacher.side.hidden(-1) if (ctx.teacher and ctx.teacher.side) else None
        if h_t is None:
            raise ValueError("dskd requires teacher hidden states")

        t2s_view = LogitsView(ctx.modules.get(f"{self.alias}__t2s")(h_t), DetachedLinear(stu_head))
        s2t_view = LogitsView(ctx.modules.get(f"{self.alias}__s2t")(h_s), tea_head)

        pad_mask = ctx.supervised_mask()
        labels = ctx.batch["student.labels"]
        t_labels = ctx.batch.get("teacher.labels", labels)
        t_pad_mask = (t_labels != -100).float() if "teacher.labels" in ctx.batch else pad_mask

        div = self._make_divergence()
        t2s_ce = -(t2s_view.target_logp(labels, chunk_size=self.cfg.chunk_size) * pad_mask).sum()
        t2s_kd = (chunked_token_divergence(s_logits, t2s_view, div, mask=pad_mask, chunk_size=self.cfg.chunk_size) * pad_mask).sum()
        s2t_kd = (chunked_token_divergence(s2t_view, t_logits, div, mask=t_pad_mask, chunk_size=self.cfg.chunk_size) * t_pad_mask).sum()

        terms: dict[str, LossTerm] = {}
        if not self.cfg.only_tea_kd:
            terms["t2s_ce"] = LossTerm(t2s_ce, "token_sum")
            terms["t2s_kd"] = LossTerm(t2s_kd, "token_sum")
        if not self.cfg.only_stu_kd:
            terms["s2t_kd"] = LossTerm(s2t_kd, "token_sum")

        t_preds = _view_argmax(t_logits, self.cfg.chunk_size)
        t_preds = torch.where(t_pad_mask.bool(), t_preds, t_labels)
        t2s_preds = _view_argmax(t2s_view, self.cfg.chunk_size)
        s_preds = _view_argmax(s_logits, self.cfg.chunk_size)
        s2t_preds = _view_argmax(s2t_view, self.cfg.chunk_size)
        n_sup = pad_mask.sum().clamp(min=1)

        metrics = {
            "t2s_agreement": ((t2s_preds == t_preds) * pad_mask).sum() / n_sup,
            "s2t_agreement": ((s2t_preds == s_preds) * pad_mask).sum() / n_sup,
            "t_acc": ((t_preds == labels) * pad_mask).sum() / n_sup,
            "t2s_acc": ((t2s_preds == labels) * pad_mask).sum() / n_sup,
        }
        return LossOutput(terms=terms, metrics=metrics)


@CRITERIONS.register("dskdv2", source="white-box-distillation/cross-tokenizer/DSKDv2-main/code/criterions/dual_space_kd_v2.py:57-130")
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

    def build_modules(self, info: ModelPairInfo, modules: Any) -> None:
        if info.teacher_hidden is None:
            raise ValueError("dskdv2 requires teacher_hidden")
        d_s, d_t = info.student_hidden, info.teacher_hidden
        modules.add(f"{self.alias}__t2s", nn.Linear(d_t, d_s, bias=self.cfg.projector_bias), group="projector", owner=self.alias)
        if not self.cfg.analytic_s2t:
            modules.add(f"{self.alias}__s2t", nn.Linear(d_s, d_t, bias=self.cfg.projector_bias), group="projector",
                        frozen=not self.cfg.train_s2t, owner=self.alias)

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

        t2s_view = LogitsView(ctx.modules.get(f"{self.alias}__t2s")(h_t), DetachedLinear(stu_head))
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

        s2t_kd = (chunked_token_divergence(s2t_view, t_logits, div, mask=t_pad_mask, chunk_size=self.cfg.chunk_size) * t_pad_mask).sum()

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

    def build_modules(self, info: ModelPairInfo, modules: Any) -> None:
        if info.teacher_hidden is None:
            raise ValueError("dskd_cma requires teacher_hidden")
        d_s, d_t = info.student_hidden, info.teacher_hidden
        modules.add(f"{self.alias}__query", DSLProjector(DSLProjector.Config(structure="2s-2t", s=d_s, t=d_t, bias=self.cfg.projector_bias)),
                    group="projector", owner=self.alias)
        modules.add(f"{self.alias}__s2t", DSLProjector(DSLProjector.Config(structure="s-t", s=d_s, t=d_t, bias=self.cfg.projector_bias)),
                    group="projector", owner=self.alias)
        modules.add(f"{self.alias}__t2s", DSLProjector(DSLProjector.Config(structure="t-s", s=d_s, t=d_t, bias=self.cfg.projector_bias)),
                    group="projector", owner=self.alias)

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

        q = ctx.modules.get(f"{self.alias}__query")(stu_index).float()
        k = norm_tea_index.float()
        v_s = ctx.modules.get(f"{self.alias}__s2t")(h_s).float()
        v_t = ctx.modules.get(f"{self.alias}__t2s")(norm_h_t + norm_tea_tgt).float()

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
            t2s_kd = (chunked_token_divergence(s_logits, t2s_view, div_tea, mask=pm * acc, chunk_size=self.cfg.chunk_size) * pm * acc).sum()
            s2t_h = (torch.softmax(align.transpose(-1, -2), dim=-1) @ v_s).to(h_s.dtype)
            s2t_view = view_from_target_hidden(s2t_h, DetachedLinear(tea_head))
            fkl = functools.partial(vd_divergence, "forward_kl", kd_temp=self.cfg.temperature, tea_temp=1.0, use_tea_temp=False)
            s2t_kd = (chunked_token_divergence(s2t_view, t_logits, fkl, mask=tpm, chunk_size=self.cfg.chunk_size) * tpm).sum()
            terms["t2s_kd"] = LossTerm(t2s_kd, "token_sum")
            terms["s2t_kd"] = LossTerm(s2t_kd, "token_sum")

        top1_prob = t2s_view.topk(1, chunk_size=self.cfg.chunk_size)[1].squeeze(-1).exp()
        metrics = {
            "t2s_acc": acc.sum() / pm.sum().clamp(min=1),
            "max_t2s_prob": (top1_prob * pm).sum() / pm.sum().clamp(min=1),
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


@CRITERIONS.register("dskdv2_eta", source="white-box-distillation/cross-tokenizer/DSKDv2-main/code/criterions/dual_space_kd_v2_with_exact_token_alignment.py:96-238")
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

    def build_modules(self, info: ModelPairInfo, modules: Any) -> None:
        if info.teacher_hidden is None:
            raise ValueError("dskdv2_eta requires teacher_hidden")
        d_s, d_t = info.student_hidden, info.teacher_hidden
        modules.add(f"{self.alias}__t2s", nn.Linear(d_t, d_s, bias=self.cfg.projector_bias), group="projector", owner=self.alias)
        if not self.cfg.analytic_s2t:
            modules.add(f"{self.alias}__s2t", nn.Linear(d_s, d_t, bias=self.cfg.projector_bias), group="projector",
                        frozen=not self.cfg.train_s2t, owner=self.alias)

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

        tea_v = ctx.modules.get(f"{self.alias}__t2s")(h_t)
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
        s2t_kd_raw = (chunked_token_divergence(s2t_view, t_logits, div, mask=rows, chunk_size=self.cfg.chunk_size) * rows).sum()

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
            "t2s_acc": (_view_argmax(t2s_view, self.cfg.chunk_size).eq(labels) * pad_mask).sum() / pad_mask.sum().clamp(min=1),
        }
        return LossOutput(terms=terms, metrics=metrics)
