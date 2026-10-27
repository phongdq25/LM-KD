"""Span Representation Alignment from IMPACT/code/criterions/SRA.py."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo, PrepareContext
from lmkd.losses.cross_tokenizer._common import pad_ragged_2d
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.modules import DSLProjector
from lmkd.teachers.signals import TeacherSignalSpec


def _last_attention(side: Any, layer: int) -> torch.Tensor:
    try:
        return side.attention_last_row(layer).detach()
    except (AttributeError, KeyError):
        return side.attention(layer).sum(1)[:, -1].detach()


def _pool(side: Any, layer: int, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = idx >= 0
    safe = idx.clamp_min(0)
    weights = _last_attention(side, layer)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
    span_w = weights.gather(1, safe.flatten(1)).view_as(safe) * mask
    hidden = side.hidden(layer)
    batch_idx = torch.arange(idx.shape[0], device=idx.device)[:, None, None]
    gathered = hidden[batch_idx, safe] * mask[..., None] * span_w[..., None]
    return gathered.sum(2) / span_w.sum(2, keepdim=True).clamp_min(1e-5), span_w.sum(2)


def _span_loss(s_pool: torch.Tensor, t_pool: torch.Tensor, t_weight: torch.Tensor,
               projector: Any) -> torch.Tensor:
    cos = F.cosine_similarity(projector(s_pool).float(), t_pool.float(), dim=-1, eps=1e-5)
    weights = t_weight / t_weight.sum(-1, keepdim=True).clamp_min(1e-9)
    return ((1 - cos) * weights).sum()


def _geometry(s_pool: torch.Tensor, t_pool: torch.Tensor, t_weight: torch.Tensor,
              power: float) -> torch.Tensor:
    weights = t_weight.pow(power)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
    pairs = weights[:, :, None] * weights[:, None, :]
    diagonal = torch.eye(weights.shape[1], device=weights.device, dtype=torch.bool)
    pairs = pairs.masked_fill(diagonal[None], 0)
    pairs = pairs / pairs.sum((1, 2), keepdim=True).clamp_min(1e-5)
    s_norm = F.normalize(s_pool.float(), dim=-1, eps=1e-5)
    t_norm = F.normalize(t_pool.float(), dim=-1, eps=1e-5)
    difference = s_norm @ s_norm.transpose(1, 2) - t_norm @ t_norm.transpose(1, 2)
    return (difference.square() * pairs).sum()


def _span_kd(s_pool: torch.Tensor, t_pool: torch.Tensor, s_head: Any, t_head: Any,
             ids: tuple[torch.Tensor, torch.Tensor] | None, temperature: float) -> torch.Tensor:
    s_logits, t_logits = s_head(s_pool), t_head(t_pool)
    if ids is not None:
        s_logits = s_logits[..., ids[0].to(s_logits.device)]
        t_logits = t_logits[..., ids[1].to(t_logits.device)]
    if s_logits.shape[-1] != t_logits.shape[-1]:
        raise ValueError("SRA needs a shared vocabulary when student/teacher vocab sizes differ")
    mask = (s_logits.abs().sum(-1) != 0).float()
    student = F.log_softmax(s_logits.float() / temperature, -1)
    teacher = F.softmax(t_logits.float() / temperature, -1)
    return (F.kl_div(student, teacher, reduction="none").sum(-1) * mask).sum()


def _shared_ids(rt: Any, mode: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    vm = getattr(getattr(rt, "data_ctx", None), "vocab_map", None)
    if mode == "auto":
        if vm is not None and vm.same_vocab:
            mode = "none"
        else:
            transforms = getattr(getattr(rt.config, "data", None), "transforms", []) if hasattr(rt, "config") else []
            has_mapping_file = any(t.get("type") == "load_vocab_map" and t.get("mapping_file") for t in transforms)
            mode = "id_map" if has_mapping_file else "exact"
    if mode == "none":
        return None
    if mode == "id_map":
        if vm is None or vm.stu2tea is None:
            raise ValueError("SRA shared_vocab=id_map needs load_vocab_map")
        student = torch.nonzero(vm.stu2tea >= 0).flatten()
        return student, vm.stu2tea[student]
    if mode == "exact":
        sv = rt.student_tokenizer.get_vocab()
        tv = rt.teacher_tokenizer.get_vocab()
        shared = sorted(set(sv) & set(tv))
        if not shared:
            raise ValueError("SRA found no exact shared vocabulary tokens")
        return torch.tensor([sv[k] for k in shared]), torch.tensor([tv[k] for k in shared])
    raise ValueError(f"unknown SRA shared_vocab mode {mode!r}")


@CRITERIONS.register("sra", source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/SRA.py:17-355")
class SRA(Criterion):
    """Attention-pooled span cosine + geometric similarity MSE + shared-vocabulary span KL.

    Source: IMPACT/code/criterions/SRA.py:17-355 and sra_span_utils.py:203-280.
    Original alpha is represented by aggregator kd_mix(kd_ratio=1-alpha, ce_alias=nll).
    legacy_scale reproduces its erroneous extra token-denominator division.
    """

    requires = ("align.sra_s_idx", "align.sra_t_idx")
    needs_teacher = True

    class Config(BaseConfig):
        geom_weight: float = 50.0
        span_power: float = 1.0
        student_layers: list[int] = [-1]
        teacher_layers: list[int] = [-1]
        hidden_loss_weights: list[float] | None = None
        temperature: float = 2.0
        use_span_loss: bool = True
        shared_vocab: str = "auto"
        legacy_scale: bool = False

    def features(self) -> FeatureRequest:
        # T03's last-row hook avoids full [B,H,T,T] attention tensors.
        student = SideRequest(logits="view", hidden=tuple(self.cfg.student_layers),
                              attention_last_row=tuple(self.cfg.student_layers))
        teacher = SideRequest(logits="view", hidden=tuple(self.cfg.teacher_layers),
                              attention_last_row=tuple(self.cfg.teacher_layers))
        return FeatureRequest(student=student, teacher=teacher, teacher_signal=TeacherSignalSpec(kind="hidden"))

    def build_modules(self, info: ModelPairInfo, modules: Any) -> None:
        if info.teacher_hidden is None:
            raise ValueError("SRA requires a teacher hidden dimension")
        cfg = DSLProjector.Config(structure="s-t", s=info.student_hidden, t=info.teacher_hidden, bias=True)
        modules.add(f"{self.alias}__sra_proj", DSLProjector(cfg), group="projector", owner=self.alias)

    def prepare(self, ctx: PrepareContext) -> None:
        rt = ctx.rt
        models_cfg = getattr(getattr(rt, "config", None), "models", None)
        for side, adapter in (("student", rt.student_adapter), ("teacher", rt.teacher_adapter)):
            spec = getattr(models_cfg, side, None)
            impl = getattr(spec, "attn_implementation", None)
            if impl not in (None, "eager") and not hasattr(adapter, "last_row_attention"):
                raise ValueError("SRA needs attention_last_row or models.*.attn_implementation: eager")
        self._vocab_ids = _shared_ids(rt, self.cfg.shared_vocab)

    def _layer_weights(self) -> list[float]:
        if len(self.cfg.student_layers) != len(self.cfg.teacher_layers) or not self.cfg.student_layers:
            raise ValueError("SRA student_layers and teacher_layers must have the same nonzero length")
        raw = self.cfg.hidden_loss_weights or [1.0] * len(self.cfg.student_layers)
        if len(raw) != len(self.cfg.student_layers) or sum(raw) <= 0:
            raise ValueError("SRA hidden_loss_weights must match layers and sum to a positive value")
        return [w / sum(raw) for w in raw]

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.side is None:
            raise ValueError("SRA requires teacher hidden states")
        s_idx = ctx.batch["align.sra_s_idx"]
        t_idx = ctx.batch["align.sra_t_idx"]
        s_idx = s_idx if isinstance(s_idx, torch.Tensor) else pad_ragged_2d(s_idx)
        t_idx = t_idx if isinstance(t_idx, torch.Tensor) else pad_ragged_2d(t_idx)
        assert s_idx.shape[:2] == t_idx.shape[:2], "SRA requires paired student/teacher spans"
        projector = ctx.modules.get(f"{self.alias}__sra_proj")
        span = torch.zeros((), device=s_idx.device)
        last_s = last_t = last_weight = None
        for weight, s_layer, t_layer in zip(self._layer_weights(), self.cfg.student_layers, self.cfg.teacher_layers):
            s_pool, _ = _pool(ctx.student, s_layer, s_idx)
            t_pool, t_weight = _pool(ctx.teacher.side, t_layer, t_idx)
            if self.cfg.use_span_loss:
                span = span + weight * _span_loss(s_pool, t_pool, t_weight, projector)
            last_s, last_t, last_weight = s_pool, t_pool, t_weight
        assert last_s is not None and last_t is not None and last_weight is not None
        geom = self.cfg.geom_weight * _geometry(last_s, last_t, last_weight, self.cfg.span_power)
        s_head = ctx.rt.student_adapter.lm_head(ctx.rt.student_model)
        t_head = ctx.rt.teacher_adapter.lm_head(ctx.rt.teacher_model)
        if s_head is None or t_head is None:
            raise ValueError("SRA requires student and teacher lm_head modules")
        kd = _span_kd(last_s, last_t, s_head, t_head,
                      getattr(self, "_vocab_ids", None), self.cfg.temperature)
        if self.cfg.legacy_scale:
            batch_size = s_idx.shape[0]
            return LossOutput({name: LossTerm(value / batch_size, "token_sum")
                               for name, value in (("span", span), ("geom", geom), ("kd", kd))})
        return LossOutput({name: LossTerm(value, "seq_sum")
                           for name, value in (("span", span), ("geom", geom), ("kd", kd))})
