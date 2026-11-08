"""SDPO: Self-Distillation with Preference Optimization - Distillation Loss Method.

Paper: Self-Distillation with Preference Optimization (2024)

Key Method Characteristics:
1. Distillation on student's top-k support with an explicit residual probability tail bucket (1 - sum P_k).
2. Generalized JSD / KL divergence parameterized by alpha:
   - alpha = 0: Forward KL (T || S)
   - alpha = 1: Reverse KL (S || T)
   - 0 < alpha < 1: Generalized Jensen-Shannon Divergence mixture
3. Importance sampling (IS) clipping ratio against rollout log probabilities:
   ratio = clamp(exp(student_logp - old_logp), max=is_clip)
4. Self-distillation gating via `has_signal` mask (only reprompted samples with valid feedback/solution contribute).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import Field

from baselines.self_distill.common import add_tail, masked_mean, renorm_topk
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.topk_kd import TopKKD
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import SignalKind, TeacherSignalSpec


def sdpo_per_token(student: torch.Tensor, teacher: torch.Tensor, alpha: float) -> torch.Tensor:
    """Evaluate SDPO divergence per token: KL(T||S), generalized JSD, or KL(S||T).

    Args:
        student: Student log-probabilities [..., V_sparse]
        teacher: Teacher log-probabilities [..., V_sparse]
        alpha: Divergence mode (0: FKL, 1: RKL, (0,1): G-JSD mixture)
    """
    teacher = teacher.detach()
    if alpha == 0:
        return F.kl_div(student, teacher, reduction="none", log_target=True).sum(-1)
    if alpha == 1:
        return F.kl_div(teacher, student, reduction="none", log_target=True).sum(-1)
    mixture = torch.logsumexp(torch.stack([student + math.log(1 - alpha), teacher + math.log(alpha)]), 0)
    ks = F.kl_div(mixture, student, reduction="none", log_target=True)
    kt = F.kl_div(mixture, teacher, reduction="none", log_target=True)
    return torch.lerp(ks, kt, alpha).sum(-1)


def compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_config: Any,
    old_log_probs: torch.Tensor | None = None,
    student_topk_log_probs: torch.Tensor | None = None,
    teacher_topk_log_probs: torch.Tensor | None = None,
    self_distillation_mask: torch.Tensor | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SDPO loss calculation over student and teacher log-probabilities."""
    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)

    metrics: dict[str, torch.Tensor] = {}

    if getattr(self_distillation_config, "full_logit_distillation", True):
        use_topk = getattr(self_distillation_config, "distillation_topk", None) is not None
        if use_topk:
            assert student_topk_log_probs is not None and teacher_topk_log_probs is not None
            s_lp = student_topk_log_probs
            t_lp = teacher_topk_log_probs

            if getattr(self_distillation_config, "distillation_add_tail", True):
                s_lp = add_tail(s_lp)
                t_lp = add_tail(t_lp)
            else:
                s_lp = renorm_topk(s_lp)
                t_lp = renorm_topk(t_lp)
        else:
            s_lp = student_log_probs
            t_lp = teacher_log_probs

        alpha = getattr(self_distillation_config, "alpha", 0.5)
        per_token_loss = sdpo_per_token(s_lp, t_lp, alpha)
    else:
        # Sampled reverse-KL
        per_token_loss = (student_log_probs - teacher_log_probs).detach() * student_log_probs

    # Importance sampling correction
    is_clip = getattr(self_distillation_config, "is_clip", None)
    if is_clip is not None:
        assert old_log_probs is not None, "is_clip requires old_log_probs"
        ratio = (student_log_probs.detach() - old_log_probs).clamp(-20, 20).exp().clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio
        metrics["sdpo/ratio_mean"] = masked_mean(ratio, loss_mask).detach()

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    total_loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1)
    metrics["sdpo/loss"] = total_loss.detach()
    return total_loss, metrics


@CRITERIONS.register("sdpo", force=True)
@CRITERIONS.register("sdpo_distill", force=True)
class SDPOLoss(Criterion):
    """SDPO self-distillation criterion with student top-k support, tail bucket, and IS correction."""

    requires = ("student.labels", "rollout.has_signal")
    needs_teacher = True
    supports_packing = False

    class Config(BaseConfig):
        alpha: float = Field(default=0.5, ge=0, le=1)
        k: int | None = Field(default=100, ge=0)
        add_tail: bool = True
        is_clip: float | None = Field(default=2.0, gt=0)
        full_logit: bool = True
        temperature: float = 1.0

    @classmethod
    def required_keys(cls, cfg: Config) -> tuple[str, ...]:
        return cls.requires + (("rollout.old_logp",) if cfg.is_clip is not None else ())

    def features(self) -> FeatureRequest:
        if not self.cfg.full_logit and self.cfg.alpha != 1:
            raise ValueError("sampled SDPO requires alpha=1")
        kind: SignalKind = "topk" if self.cfg.k else "full"
        if not self.cfg.full_logit:
            kind = "sampled_logp"
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(
                kind=kind, k=self.cfg.k or 0, support="student", tail="none", temperature=self.cfg.temperature
            ),
        )

    def _distribution_loss(self, ctx: LossContext) -> torch.Tensor:
        if ctx.teacher is None:
            raise ValueError("SDPO requires teacher output")
        student, teacher = ctx.student_logits(), ctx.teacher
        t_view = teacher.logits
        if self.cfg.temperature != 1.0:
            student = student.with_temperature(self.cfg.temperature)
            if t_view is not None:
                t_view = t_view.with_temperature(self.cfg.temperature)
        if self.cfg.k:
            ids, s = student.topk(self.cfg.k)
            t = teacher.topk_logp
            if t is None or teacher.topk_ids is None or not teacher.topk_ids.equal(ids):
                if t_view is None:
                    raise ValueError("SDPO top-k distillation requires teacher topk_logp or teacher logits")
                t = t_view.gather(ids)
            if self.cfg.add_tail:
                s, t = add_tail(s), add_tail(t)
            else:
                s, t = renorm_topk(s), renorm_topk(t)
        else:
            if t_view is None:
                raise ValueError("SDPO full_logit distillation requires teacher logits")
            s, t = student.full().log_softmax(-1), t_view.full().log_softmax(-1)
        return sdpo_per_token(s, t, self.cfg.alpha)

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None:
            raise ValueError("SDPO requires teacher output")
        labels = ctx.batch["student.labels"]
        s_view = ctx.student_logits()
        if self.cfg.temperature != 1.0:
            s_view = s_view.with_temperature(self.cfg.temperature)
        s_tok = s_view.target_logp(labels)

        if self.cfg.full_logit:
            per_token = self._distribution_loss(ctx)
        else:
            t_tok = ctx.teacher.token_logp
            if t_tok is None:
                if ctx.teacher.logits is None:
                    raise ValueError("SDPO sampled distillation requires teacher token_logp or logits")
                t_view = ctx.teacher.logits
                if self.cfg.temperature != 1.0:
                    t_view = t_view.with_temperature(self.cfg.temperature)
                t_tok = t_view.target_logp(labels)
            per_token = (s_tok - t_tok).detach() * s_tok

        metrics: dict[str, torch.Tensor] = {}
        if self.cfg.is_clip is not None:
            old = ctx.batch.get("rollout.old_logp")
            if old is None:
                raise ValueError("SDPO importance correction requires rollout.old_logp")
            ratio = (s_tok.detach() - old.detach()).clamp(-20, 20).exp().clamp(max=self.cfg.is_clip)
            per_token = per_token * ratio
            mask_raw = ctx.supervised_mask() * ctx.batch["rollout.has_signal"][:, None]
            metrics["sdpo/ratio_mean"] = masked_mean(ratio, mask_raw).detach()

        per_token = per_token * ctx.batch.get("rollout.is_weight", 1.0)
        mask = ctx.supervised_mask() * ctx.batch["rollout.has_signal"][:, None]
        metrics["sdpo/distill_mean"] = masked_mean(per_token, mask).detach()

        return LossOutput({"distill": LossTerm((per_token * mask).sum(), "token_sum")}, metrics=metrics)


# Alias for backward compatibility
SDPODistill = SDPOLoss



__all__ = [
    "SDPOLoss",
    "SDPODistill",
    "TopKKD",
    "sdpo_per_token",
    "compute_self_distillation_loss",
    "add_tail",
]
