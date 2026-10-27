"""Top-k sparse knowledge distillation (SDPO / OPSD).

Evaluates KL divergence / Generalized JSD restricted to top-k tokens of teacher or student,
with either an added tail bucket (SDPO) or renormalization over top-k (SDPO / OPSD).
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import field_validator

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


@CRITERIONS.register(
    "topk_kd",
    paper="SDPO; OPSD top_k_loss",
    source="Self-Distill/SDPO-main/verl/trainer/ppo/core_algos.py:1085-1182",
)
class TopKKD(Criterion):
    """KL/JSD on top-k support (+tail bucket or renorm)."""

    needs_teacher = True

    class Config(BaseConfig):
        alpha: float = 0.0                # 0 = KL(T||S) (fwd), 1 = KL(S||T) (rev), else generalized JSD (lerp)
        k: int = 100
        support: Literal["teacher", "student"] = "teacher"  # teacher | student
        tail: Literal["bucket", "renorm"] = "bucket"        # bucket | renorm
        temperature: float = 1.0
        token_clip: float | None = None   # elementwise clip (OPSD style), applied before summing over K(+1)

        @field_validator("support")
        @classmethod
        def _check_support(cls, v: str) -> str:
            if v not in ("teacher", "student"):
                raise ValueError(f"support must be 'teacher' or 'student', got '{v}'")
            return v

        @field_validator("tail")
        @classmethod
        def _check_tail(cls, v: str) -> str:
            if v not in ("bucket", "renorm"):
                raise ValueError(f"tail must be 'bucket' or 'renorm', got '{v}'")
            return v

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(
                kind="topk",
                k=self.cfg.k,
                support=self.cfg.support,
                tail="none",
                temperature=self.cfg.temperature,
            ),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.topk_ids is None or ctx.teacher.topk_logp is None:
            raise ValueError("topk_kd requires teacher topk signals (topk_ids and topk_logp)")

        ids = ctx.teacher.topk_ids
        t_lp = ctx.teacher.topk_logp
        s_lp = ctx.student_logits().with_temperature(self.cfg.temperature).gather(ids)

        if self.cfg.tail == "bucket":
            def add_tail(x: torch.Tensor) -> torch.Tensor:
                log_s = torch.logsumexp(x, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)
                tail_log = torch.log(-torch.expm1(log_s))
                return torch.cat([x, tail_log], dim=-1)

            s_distill = add_tail(s_lp)
            t_distill = add_tail(t_lp)
        elif self.cfg.tail == "renorm":
            def renorm(x: torch.Tensor) -> torch.Tensor:
                return x - torch.logsumexp(x, dim=-1, keepdim=True)

            s_distill = renorm(s_lp)
            t_distill = renorm(t_lp)
        else:
            raise ValueError(f"unknown tail mode: '{self.cfg.tail}'")

        if self.cfg.alpha == 0.0:
            kl = F.kl_div(s_distill, t_distill, reduction="none", log_target=True)
        elif self.cfg.alpha == 1.0:
            kl = F.kl_div(t_distill, s_distill, reduction="none", log_target=True)
        else:
            alpha_tensor = torch.tensor(self.cfg.alpha, dtype=s_distill.dtype, device=s_distill.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([
                    s_distill + torch.log(1.0 - alpha_tensor),
                    t_distill + torch.log(alpha_tensor),
                ]),
                dim=0,
            )
            kl_t = F.kl_div(mixture_log_probs, t_distill, reduction="none", log_target=True)
            kl_s = F.kl_div(mixture_log_probs, s_distill, reduction="none", log_target=True)
            kl = torch.lerp(kl_s, kl_t, alpha_tensor)

        if self.cfg.token_clip is not None:
            kl = kl.clamp(max=self.cfg.token_clip)

        per_tok = kl.sum(-1)
        mask = ctx.supervised_mask()
        return LossOutput(terms={"topk_kd": LossTerm((per_tok * mask).sum(), "token_sum")})

