"""Vectorized IMPACT boundary-preserving prefix increment alignment."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS, LAYER_MAPPERS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, PrepareContext
from lmkd.losses.cross_tokenizer._common import get_vocab_map, view_from_target_hidden
from lmkd.losses.layer_mappers import LayerMapper, MapperInputs
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherSignalSpec


def vocab_projection(rt: Any, lambda_reg: float, max_overlap: int | None) -> torch.Tensor:
    """Ridge W=(E_T^T E_T + lambda I)^-1 E_T^T E_S on sorted overlapping vocabulary."""
    vm = get_vocab_map(PrepareContext(rt))
    s_embed = rt.student_adapter.embed_tokens(rt.student_model).weight
    t_embed = rt.teacher_adapter.embed_tokens(rt.teacher_model).weight
    ov_s, ov_t = vm.overlap_sorted_s, vm.overlap_sorted_t
    if max_overlap is not None and max_overlap > 0 and len(ov_s) > max_overlap:
        take = ov_s.argsort()[:max_overlap]
        ov_s, ov_t = ov_s[take], ov_t[take]
    if len(ov_s) == 0:
        logging.getLogger(__name__).warning("IMPACT vocabulary overlap is empty; projection W is zero")
        return torch.zeros(t_embed.shape[1], s_embed.shape[1], device=t_embed.device)
    E_T = t_embed.detach()[ov_t.to(t_embed.device)].float()
    E_S = s_embed.detach()[ov_s.to(s_embed.device)].float().to(E_T.device)
    A = E_T.T @ E_T + lambda_reg * torch.eye(E_T.shape[1], device=E_T.device)
    return torch.linalg.solve(A, E_T.T @ E_S).detach()


def vectorized_teacher_increments(projected: torch.Tensor, lo: torch.Tensor,
                                  hi: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Gathers teacher prefix states at both student-token boundaries, then subtracts."""
    B, _, Ds = projected.shape
    Ts = lo.shape[1]
    prefix = torch.cat([projected.new_zeros(B, 1, Ds), projected], dim=1)
    def gather(index: torch.Tensor) -> torch.Tensor:
        return prefix.gather(1, index.reshape(B, -1, 1).expand(-1, -1, Ds)).reshape(B, Ts, 2, Ds)
    states = (1 - w)[..., None] * gather(lo) + w[..., None] * gather(hi)
    return states[:, :, 1] - states[:, :, 0]


def _student_increments(hidden: torch.Tensor) -> torch.Tensor:
    previous = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], dim=1)
    return hidden - previous


def _pair_term(hidden: torch.Tensor, projected: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
               w: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    delta_s = _student_increments(hidden.float())
    delta_t = vectorized_teacher_increments(projected, lo, hi, w)
    cosine = F.cosine_similarity(delta_s, delta_t, dim=-1, eps=1e-6)
    return (((1 - cosine) * mask).sum(dim=1) / lengths).sum()


def _project(ctx: LossContext, layer: int, W: torch.Tensor) -> torch.Tensor:
    key = ("impact_proj", layer)
    if key not in ctx.shared:
        assert ctx.teacher is not None and ctx.teacher.side is not None
        hidden = ctx.teacher.side.hidden(layer).float()
        ctx.shared[key] = hidden @ W.to(hidden.device)
    return ctx.shared[key]


@CRITERIONS.register("impact", source="white-box-distillation/cross-tokenizer/IMPACT/code/impact_utils.py:107-176, 779-896")
class Impact(Criterion):
    """IMPACT BPIA: weighted mean over sequence tokens of 1-cosine between student and teacher prefix increments.

    Source: IMPACT/code/impact_utils.py:107-176, 779-896 and criterions/IMPACT.py:20-80.
    Offsets come from the actual token ids, not retokenized text. seq_sum fixes the original extra token
    denominator; legacy_scale restores it. RAIL-KD has no criterion restriction; bi_ratio is an ablation extension.
    """

    requires = ("align.impact_lo", "align.impact_hi", "align.impact_w")
    needs_teacher = True
    supports_packing = False

    class Config(BaseConfig):
        layer_mapper: dict = {"type": "topk_bi", "k": 4}
        boundary_mode: Literal["linear", "nearest", "containing"] = "linear"
        lambda_reg: float = 1.0
        max_overlap: int | None = None
        legacy_scale: bool = False

    def __init__(self, cfg: BaseConfig | None = None):
        super().__init__(cfg)
        self.register_buffer("W", torch.empty(0), persistent=True)
        self.mapper: LayerMapper | None = None

    def features(self) -> FeatureRequest:
        ppl = self.cfg.layer_mapper.get("type") == "topk_ppl"
        return FeatureRequest(student=SideRequest(hidden="all"),
                              teacher=SideRequest(hidden="all", logits="view" if ppl else "none"),
                              teacher_signal=TeacherSignalSpec(kind="hidden"))

    def prepare(self, ctx: PrepareContext) -> None:
        self.W = vocab_projection(ctx.rt, self.cfg.lambda_reg, self.cfg.max_overlap)
        self.mapper = LAYER_MAPPERS.build(self.cfg.layer_mapper)
        self.mapper.prepare(ctx.rt)

    def _inputs(self, ctx: LossContext) -> MapperInputs:
        assert ctx.teacher is not None and ctx.teacher.side is not None
        teacher = ctx.teacher.side
        view: Callable[[int], LogitsView] | None = None
        if self.cfg.layer_mapper.get("type") == "topk_ppl":
            head = ctx.rt.teacher_adapter.lm_head(ctx.rt.teacher_model)
            if head is None:
                raise ValueError("IMPACT PPL requires a teacher lm_head")
            def layer_view(layer: int) -> LogitsView:
                return view_from_target_hidden(teacher.hidden(layer), head)
            view = layer_view
        return MapperInputs(ctx.student.num_layers, teacher.num_layers, ctx.state,
                            teacher_hidden=teacher.hidden, student_hidden=ctx.student.hidden,
                            teacher_labels=ctx.batch.get("teacher.labels"), teacher_view_at_layer=view,
                            W=self.W, rt=ctx.rt)

    def _validate_mode(self, ctx: LossContext) -> None:
        modes = ctx.batch.meta.get("meta.impact_mode")
        if modes is None:
            return
        values = [modes] if isinstance(modes, str) else modes
        if any(mode != self.cfg.boundary_mode for mode in values):
            raise ValueError(f"IMPACT boundary_mode={self.cfg.boundary_mode!r} differs from aligner {values!r}")

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.side is None:
            raise ValueError("IMPACT requires online teacher hidden states")
        if self.mapper is None or self.W.numel() == 0:
            raise RuntimeError("IMPACT.prepare must run before forward")
        self._validate_mode(ctx)
        inp = self._inputs(ctx)
        self.mapper.observe(inp)
        pairs = self.mapper.pairs(inp)
        batch = ctx.batch
        lo, hi, w = batch["align.impact_lo"], batch["align.impact_hi"], batch["align.impact_w"]
        mask = batch["student.attention_mask"].float()
        lengths = mask.sum(dim=1).clamp_min(1)
        total = mask.new_zeros(())
        if pairs and pairs[0][0] == -1:
            alpha = self.mapper.fuse_weights(inp)
            assert alpha is not None
            projected = torch.stack([_project(ctx, l, self.W) for l in range(1, inp.L_T + 1)], dim=1)
            for _, m, weight in pairs:
                fused = torch.einsum("bl,bltd->btd", alpha[:, m - 1], projected)
                total = total + weight * _pair_term(ctx.student.hidden(m), fused, lo, hi, w, mask, lengths)
        else:
            for l, m, weight in pairs:
                total = total + weight * _pair_term(ctx.student.hidden(min(m, inp.L_S)),
                                                   _project(ctx, l, self.W), lo, hi, w, mask, lengths)
        term = (LossTerm(total / mask.shape[0], "token_sum") if self.cfg.legacy_scale
                else LossTerm(total, "seq_sum"))
        frozen = self.mapper.state_dict().get("frozen") is not None
        metrics = {"impact_pairs": mask.new_tensor(len(pairs)),
                   "impact_layers": mask.new_tensor(pairs[0][0] if pairs else -1),
                   "impact_bi_frozen": mask.new_tensor(float(frozen))}
        return LossOutput({"impact": term}, metrics)

    def extra_state(self) -> dict:
        return {"W": self.W.detach().cpu(), "mapper": self.mapper.state_dict() if self.mapper is not None else {}}

    def load_extra_state(self, state: dict) -> None:
        if "W" in state:
            self.W = state["W"].to(self.W.device)
        if self.mapper is not None and "mapper" in state:
            self.mapper.load_state_dict(state["mapper"])
