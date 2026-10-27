"""SFT-style criteria. ``nll`` is the REFERENCE EXAMPLE every new criterion should imitate."""
from __future__ import annotations

import warnings

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.fused import fused_nll, liger_available
from lmkd.models.forward import FeatureRequest, SideRequest

_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, UserWarning, stacklevel=3)


@CRITERIONS.register("nll", source="standard causal LM cross-entropy")
class NLL(Criterion):
    """Token-level negative log-likelihood on supervised targets (labels != -100, x seq.loss_mask)."""

    class Config(BaseConfig):
        chunk_size: int = 1024
        fused: str = "off"  # off | auto | on

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def _should_fuse(self, ctx: LossContext, view) -> bool:
        if self.cfg.fused == "off":
            return False

        reason = None
        tw = ctx.batch.get("seq.token_weights")
        if tw is not None and not bool((tw == 1.0).all()):
            reason = "token weights are present in batch"
        elif not liger_available():
            reason = "liger_kernel is not available"
        elif view.hidden is None:
            reason = "student logits view lacks hidden states"
        elif not view.hidden.is_cuda:
            reason = "student hidden states must be on CUDA for fused kernel"

        if reason is not None:
            if self.cfg.fused == "on":
                raise ValueError(f"NLL fused execution failed: {reason}")
            _warn_once(f"nll_fallback_{reason}", f"NLL fused fallback: {reason}")
            return False
        return True

    def forward(self, ctx: LossContext) -> LossOutput:
        labels = ctx.batch["student.labels"]
        mask = ctx.supervised_mask()
        view = ctx.student_logits()

        if self._should_fuse(ctx, view):
            lm_head = ctx.rt.student_adapter.lm_head(ctx.rt.student_model) if ctx.rt is not None else None
            nll = fused_nll(view, labels, mask, lm_head=lm_head)
        else:
            logp = view.target_logp(labels, chunk_size=self.cfg.chunk_size)  # [B,T]
            nll = -(logp * mask).sum()

        return LossOutput(
            terms={"nll": LossTerm(nll, "token_sum")},
            metrics={"tokens": mask.sum().detach()},
        )


@CRITERIONS.register("weighted_nll", source="Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/training/losses.py:23")
class WeightedNLL(Criterion):
    """sum_t w_t * mask_t * (-log p_t). Weights do NOT change the denominator Z (SGL Eq. 9)."""

    requires = ("seq.token_weights",)

    class Config(BaseConfig):
        chunk_size: int = 1024
        fused: str = "off"  # off | auto | on

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx: LossContext) -> LossOutput:
        if self.cfg.fused == "on":
            raise ValueError("weighted_nll does not support fused execution (token weights are present)")
        if self.cfg.fused == "auto":
            _warn_once("weighted_nll_fallback", "weighted_nll fused fallback: token weights are present")

        labels = ctx.batch["student.labels"]
        mask = ctx.supervised_mask()
        w = ctx.batch["seq.token_weights"]
        logp = ctx.student_logits().target_logp(labels, chunk_size=self.cfg.chunk_size)
        return LossOutput(terms={"wnll": LossTerm(-(logp * mask * w).sum(), "token_sum")})
