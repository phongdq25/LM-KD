"""Total Variation Preference Distillation (TVKD).

Value-based preference distillation objective with optional compressed token KD.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from baselines.preference.common import _check_pairwise, _reduction_term, pairwise
from baselines.preference.dckd.loss import _project_support, compressed_projection_kl
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.models.forward import FeatureRequest, SideRequest


def _tvkd_values(
    s: torch.Tensor,
    p: torch.Tensor,
    keep: torch.Tensor,
    func: str,
    temp: float,
    s_wt: float,
    t_wt: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    s_log = s.clamp_min(eps).log()
    p_log = p.clamp_min(eps).log()
    if func == "entropy":
        values = -(torch.where(keep, s * s_log, torch.zeros_like(s))).sum(dim=-1)
    elif func == "soft":
        comb = (s_wt * s_log + t_wt * p_log) / temp
        comb = torch.where(keep, comb, torch.full_like(comb, -1e9))
        values = temp * torch.logsumexp(comb, dim=-1)
    else:
        raise ValueError(f"Unknown TVKD value function: {func}")
    return values, s_log


def tvkd_scores(
    student_probs: torch.Tensor,
    labels: torch.Tensor,
    ids: torch.Tensor,
    p: torch.Tensor,
    tail_p: torch.Tensor,
    *,
    value_function: str = "entropy",
    value_discount: float = 1.0,
    value_temperature: float = 1.0,
    student_value_weight: float = 0.0,
    teacher_value_weight: float = 0.7,
    eps: float = 1e-8,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Vectorized sequence score for TVKD: sum_t (action_t - discount * next_value_t)."""
    mask = labels.ne(ignore_index).clone()
    mask[:, 0] = False
    if not mask.any(dim=-1).all():
        raise ValueError("TVKD teacher probabilities do not align with response labels")

    b, t, v_s = student_probs.shape
    k = ids.shape[-1]
    s, p_proj, keep, in_range = _project_support(
        student_probs.reshape(-1, v_s), ids.reshape(-1, k), p.reshape(-1, k), tail_p.reshape(-1), eps
    )
    values, s_log = _tvkd_values(
        s, p_proj, keep, value_function, value_temperature, student_value_weight, teacher_value_weight, eps
    )

    flat_labels = labels.reshape(-1)
    matched = (ids.reshape(-1, k) == flat_labels.unsqueeze(-1)) & in_range
    action_index = torch.where(matched.any(dim=-1), matched.long().argmax(dim=-1), k)
    actions = s_log.gather(-1, action_index.unsqueeze(-1)).squeeze(-1)

    actions = torch.where(mask, actions.reshape(b, t), torch.zeros_like(mask, dtype=actions.dtype))
    values = torch.where(mask, values.reshape(b, t), torch.zeros_like(mask, dtype=values.dtype))

    first_idx = mask.long().argmax(dim=-1)
    first_values = values.gather(-1, first_idx.unsqueeze(-1)).squeeze(-1)
    next_value_sums = values.sum(dim=-1) - first_values
    return actions.sum(dim=-1) - value_discount * next_value_sums


def tvkd_loss(
    chosen_student_probs: torch.Tensor,
    rejected_student_probs: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    chosen_ids: torch.Tensor,
    chosen_p: torch.Tensor,
    chosen_tail_p: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_p: torch.Tensor,
    rejected_tail_p: torch.Tensor,
    *,
    value_function: str = "entropy",
    value_discount: float = 1.0,
    value_temperature: float = 1.0,
    student_value_weight: float = 0.0,
    teacher_value_weight: float = 0.7,
    eps: float = 1e-8,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute TVKD preference distillation loss from chosen and rejected sequences."""
    chosen_scores = tvkd_scores(
        chosen_student_probs,
        chosen_labels,
        chosen_ids,
        chosen_p,
        chosen_tail_p,
        value_function=value_function,
        value_discount=value_discount,
        value_temperature=value_temperature,
        student_value_weight=student_value_weight,
        teacher_value_weight=teacher_value_weight,
        eps=eps,
        ignore_index=ignore_index,
    )
    rejected_scores = tvkd_scores(
        rejected_student_probs,
        rejected_labels,
        rejected_ids,
        rejected_p,
        rejected_tail_p,
        value_function=value_function,
        value_discount=value_discount,
        value_temperature=value_temperature,
        student_value_weight=student_value_weight,
        teacher_value_weight=teacher_value_weight,
        eps=eps,
        ignore_index=ignore_index,
    )
    return -F.logsigmoid(chosen_scores - rejected_scores)


@CRITERIONS.register(
    "tvkd",
)
class TVKD(Criterion):
    """Value-based preference distillation objective (TVKD).

    Chosen response is index 0, rejected is index 1.
    Formula: -log sigmoid(score_chosen - score_rejected) where score = sum_t (action_t - gamma * next_value_t).
    """

    needs_teacher: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ("cache.topk_ids", "cache.topk_p", "cache.tail_p", "group.size")

    class Config(BaseConfig):
        value_function: str = "entropy"
        value_discount: float = 1.0
        value_temperature: float = 1.0
        student_value_weight: float = 0.0
        teacher_value_weight: float = 0.7
        chosen_kd_weight: float = 0.0
        rejected_kd_weight: float = 0.0
        eps: float = 1e-8
        reduction: str = "mean"
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx: LossContext) -> LossOutput:
        n = _check_pairwise(ctx.batch, "tvkd")
        labels = ctx.batch["student.labels"]
        ids = ctx.batch["cache.topk_ids"]
        p = ctx.batch["cache.topk_p"]
        tail_p = ctx.batch["cache.tail_p"]

        target_probs = torch.cat(
            [c.logits.softmax(dim=-1) for c in ctx.student_logits().chunks(size=self.cfg.chunk_size)], dim=1
        )

        probs_c, probs_r = pairwise(target_probs)
        labels_c, labels_r = pairwise(labels)
        ids_c, ids_r = pairwise(ids)
        p_c, p_r = pairwise(p)
        tail_c, tail_r = pairwise(tail_p)

        loss = tvkd_loss(
            probs_c,
            probs_r,
            labels_c,
            labels_r,
            ids_c,
            p_c,
            tail_c,
            ids_r,
            p_r,
            tail_r,
            value_function=self.cfg.value_function,
            value_discount=self.cfg.value_discount,
            value_temperature=self.cfg.value_temperature,
            student_value_weight=self.cfg.student_value_weight,
            teacher_value_weight=self.cfg.teacher_value_weight,
            eps=self.cfg.eps,
        )

        if self.cfg.chosen_kd_weight > 0 or self.cfg.rejected_kd_weight > 0:
            mask = ctx.supervised_mask().bool()
            mask_c, mask_r = pairwise(mask)
            sub_c = mask_c.any(dim=-1)
            sub_r = mask_r.any(dim=-1)
            kl_c = (
                compressed_projection_kl(
                    probs_c[mask_c], ids_c[mask_c], p_c[mask_c], tail_c[mask_c], eps=self.cfg.eps
                ).mean()
                if sub_c.any()
                else loss.new_zeros(())
            )
            kl_r = (
                compressed_projection_kl(
                    probs_r[mask_r], ids_r[mask_r], p_r[mask_r], tail_r[mask_r], eps=self.cfg.eps
                ).mean()
                if sub_r.any()
                else loss.new_zeros(())
            )
            loss = loss + self.cfg.chosen_kd_weight * kl_c + self.cfg.rejected_kd_weight * kl_r

        term = _reduction_term(loss, self.cfg.reduction, n)
        return LossOutput({"tvkd": term}, metrics={"loss": loss.mean().detach()})
