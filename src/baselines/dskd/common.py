"""Common utilities and base class for Dual-Space Knowledge Distillation (DSKD)."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn

from lmkd.core.plugin import BaseConfig
from lmkd.losses.base import Criterion, LossContext, PrepareContext
from lmkd.losses.cross_tokenizer._common import (
    vd_divergence,
)
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherSignalSpec


def _view_argmax(view: LogitsView, chunk_size: int = 1024) -> torch.Tensor:
    """Argmax index [B, T] from a LogitsView without materializing full logits."""
    return view.topk(1, chunk_size=chunk_size)[0].squeeze(-1)


def _get_heads(ctx: LossContext) -> tuple[nn.Module, nn.Module]:
    """Retrieve student and teacher lm_head modules from runtime or logits views."""
    rt = getattr(ctx, "rt", None)
    stu_head: nn.Module | None = None
    if (
        rt is not None
        and getattr(rt, "student_adapter", None) is not None
        and getattr(rt, "student_model", None) is not None
    ):
        stu_head = rt.student_adapter.lm_head(rt.student_model)
    if stu_head is None and ctx.student.logits is not None:
        stu_head = getattr(ctx.student.logits, "_head", None)

    tea_head: nn.Module | None = None
    if (
        rt is not None
        and getattr(rt, "teacher_adapter", None) is not None
        and getattr(rt, "teacher_model", None) is not None
    ):
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
    if (
        rt is not None
        and getattr(rt, "student_adapter", None) is not None
        and getattr(rt, "student_model", None) is not None
    ):
        stu_emb = rt.student_adapter.embed_tokens(rt.student_model)
    tea_emb = None
    if (
        rt is not None
        and getattr(rt, "teacher_adapter", None) is not None
        and getattr(rt, "teacher_model", None) is not None
    ):
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
        if (
            vm is None
            and getattr(rt, "student_tokenizer", None) is not None
            and getattr(rt, "teacher_tokenizer", None) is not None
        ):
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
        t2s_mod = criterion.get_module(rt, "t2s")
        if t2s_mod is not None:
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
    criterion.load_aux_checkpoint(rt)


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
            _init_pinv_and_buffers(criterion, PrepareContext(rt=ctx.rt))
        stu_w: torch.Tensor = cast(Any, stu_head).weight
        W_S_curr = stu_w.detach().transpose(0, 1)  # [D_s, V_s]
        if criterion.overlap_s_part is not None:
            W_S_curr = W_S_curr[:, criterion.overlap_s_part.to(W_S_curr.device)]
        elif getattr(criterion.cfg, "topk_vocab", -1) != -1:
            W_S_curr = W_S_curr[:, : criterion.cfg.topk_vocab]
        pinv_buf: torch.Tensor = cast(Any, criterion).part_T_pinv
        pinv_mat = pinv_buf.to(W_S_curr.device).float()
        s2t_mat = (W_S_curr.float() @ pinv_mat).to(h_s.dtype)
        return h_s @ s2t_mat
    s2t_mod = criterion.aux(ctx).s2t
    res: torch.Tensor = s2t_mod(h_s)
    return res

