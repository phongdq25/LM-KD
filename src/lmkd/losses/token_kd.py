"""Same-vocabulary token-level KD (DistiLLM family, CSD, GKD/OPSD JSD).

Formula: per-target divergence D(q_S || p_T) computed chunk-wise on the target-aligned views,
masked by supervised targets, summed -> ``token_sum`` (the aggregator divides by global Z).
Teacher signal: ``full`` online logits (hf_frozen / self).

Notes on differences from original codebases:
  * Reduction: original implementations reduce by the token count of the micro-batch;
    this framework sums numerators and divides by global Z across all micro-batches and ranks
    during the optimizer step. The two formulations are identical without gradient accumulation.
  * Precision: original CSD runs in bf16 without forcing fp32; this framework computes
    divergences in fp32. This is an intentional and accepted numerical difference.
  * Temperature: DistiLLM does not use temperature (tau=1). OPSD applies tau equally to both sides.
"""
from __future__ import annotations

import warnings
from typing import ClassVar

import torch
from pydantic import field_validator, model_validator

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.functional.divergences import DIVERGENCES, get_divergence, reverse_kl
from lmkd.losses.fused import fused_jsd, liger_available
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.logits_view import LogitsView, chunked_token_divergence
from lmkd.teachers.signals import TeacherSignalSpec

_WARNED_TOKEN_KD: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED_TOKEN_KD:
        _WARNED_TOKEN_KD.add(key)
        warnings.warn(message, UserWarning, stacklevel=3)


@CRITERIONS.register(
    "token_kd",
    paper="DistiLLM (Ko et al. 2024); CSD; GKD",
    source="white-box-distillation/same-tokenizer/distillm/losses.py",
)
class TokenKD(Criterion):
    """Same-vocabulary token-level knowledge distillation with configurable divergence.

    Supported divergences:
      - fkl: Forward KL, D_KL(p_T || q_S) = sum_v p_T(v) log(p_T(v) / q_S(v))
      - rkl: Reverse KL, D_KL(q_S || p_T) = sum_v q_S(v) log(q_S(v) / p_T(v))
      - symkl: Symmetric KL, 0.5 * (D_KL(p_T || q_S) + D_KL(q_S || p_T))
      - jsd: Jensen-Shannon Divergence, 0.5 * D_KL(p_T || M) + 0.5 * D_KL(q_S || M), M = 0.5*(p_T + q_S)
      - tvd: Total Variation Distance, 0.5 * sum_v |p_T(v) - q_S(v)|
      - sfkl: Skewed Forward KL with mixing parameter skew_alpha
      - srkl: Skewed Reverse KL with mixing parameter skew_alpha
      - csd: Cosine / Surrogate divergence
      - gjsd: Generalized JSD with mixing parameter beta and optional elementwise clipping
    """

    needs_teacher = True
    schedulable_fields: ClassVar[tuple[str, ...] | None] = ("temperature",)

    class Config(BaseConfig):
        divergence: str = "fkl"
        temperature: float = 1.0
        skew_alpha: float = 0.1          # sfkl / srkl
        beta: float = 0.5                # gjsd mixing
        token_clip: float | None = None  # gjsd clipping
        chunk_size: int = 1024
        jsd_lam: float = 0.9             # DistiLLM js_distance / symmetric_kl lam
        jsd_eps: float = 0.0             # DistiLLM js_distance eps
        clip_mode: str = "elementwise"   # gjsd only: elementwise | per_token
        log_rkl_for_csd: bool = False    # CSD: opt-in log real RKL value (no grad) as LossTerm.log_value
        fused: str = "off"               # off | auto | on (T26 Liger fused path)

        @field_validator("divergence")
        @classmethod
        def _known(cls, v: str) -> str:
            if v not in DIVERGENCES:
                raise ValueError(f"unknown/unimplemented divergence '{v}', available {sorted(DIVERGENCES)} (task T04)")
            return v

        @model_validator(mode="after")
        def _check_clip(self) -> TokenKD.Config:
            if self.token_clip is not None and self.divergence != "gjsd":
                raise ValueError(f"token_clip is only valid when divergence == 'gjsd', got '{self.divergence}'")
            return self

    def __init__(self, cfg: TokenKD.Config):
        super().__init__(cfg)
        params = {
            "alpha": cfg.skew_alpha,
            "beta": cfg.beta,
            "lam": cfg.jsd_lam,
            "eps": cfg.jsd_eps,
            "token_clip": cfg.token_clip,
            "clip_mode": cfg.clip_mode,
        }
        self.fn = get_divergence(cfg.divergence, **params)

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="full"),
        )

    def _check_fused_support(
        self, ctx: LossContext, s_view: LogitsView, t_view: LogitsView, head_s, head_t
    ) -> tuple[bool, str | None, float | None]:
        tw = ctx.batch.get("seq.token_weights")
        if tw is not None and not bool((tw == 1.0).all()):
            return False, "token weights are present in batch", None
        if not liger_available():
            return False, "liger_kernel is not available", None
        if s_view.hidden is None or t_view.hidden is None:
            return False, "student or teacher view lacks hidden states", None
        if not s_view.hidden.is_cuda:
            return False, "tensors must be on CUDA for fused kernel", None
        if self.cfg.divergence not in ("gjsd", "rkl"):
            return False, f"divergence '{self.cfg.divergence}' is not supported by fused path", None
        if self.cfg.token_clip is not None:
            return False, "token_clip is not supported by fused path", None
        if self.cfg.temperature <= 0.0:
            return False, "invalid temperature for fused path", None
        if (
            head_s is not None
            and head_t is not None
            and getattr(head_s, "weight", None) is not None
            and getattr(head_t, "weight", None) is not None
            and head_s.weight.shape[0] != head_t.weight.shape[0]
        ):
            return False, "student and teacher vocabulary size must match", None

        beta = self.cfg.beta if self.cfg.divergence == "gjsd" else 1.0
        return True, None, beta

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.logits is None:
            raise ValueError("token_kd needs online teacher logits (teacher.provider: hf_frozen | self)")
        mask = ctx.supervised_mask()
        s_view = ctx.student_logits().with_temperature(self.cfg.temperature)
        t_view = ctx.teacher.logits.with_temperature(self.cfg.temperature)

        if self.cfg.fused in ("auto", "on"):
            head_s = ctx.rt.student_adapter.lm_head(ctx.rt.student_model) if ctx.rt is not None else None
            head_t = (
                ctx.rt.teacher_adapter.lm_head(ctx.rt.teacher_model)
                if (ctx.rt is not None and getattr(ctx.rt, "teacher_adapter", None) is not None)
                else getattr(t_view, "_head", None)
            )
            can_fuse, reason, beta = self._check_fused_support(ctx, s_view, t_view, head_s, head_t)
            if not can_fuse:
                if self.cfg.fused == "on":
                    raise ValueError(f"TokenKD fused execution failed: {reason}")
                _warn_once(f"token_kd_fallback_{reason}", f"TokenKD fused fallback: {reason}")
            else:
                assert beta is not None
                labels = ctx.batch["student.labels"]
                fused_loss = fused_jsd(
                    s_view, t_view, labels, mask, beta=beta, temperature=self.cfg.temperature,
                    lm_head_s=head_s, lm_head_t=head_t,
                )
                return LossOutput(terms={self.cfg.divergence: LossTerm(fused_loss, "token_sum")})

        per_tok = chunked_token_divergence(s_view, t_view, self.fn, mask=mask, chunk_size=self.cfg.chunk_size)
        log_val = None
        if self.cfg.divergence == "csd" and self.cfg.log_rkl_for_csd:
            with torch.no_grad():
                rkl_per_tok = chunked_token_divergence(
                    s_view, t_view, reverse_kl, mask=mask, chunk_size=self.cfg.chunk_size
                )
                log_val = rkl_per_tok.sum()

        return LossOutput(terms={self.cfg.divergence: LossTerm(per_tok.sum(), "token_sum", log_value=log_val)})
