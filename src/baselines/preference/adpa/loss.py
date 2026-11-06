"""Advantage Distillation Preference Alignment (ADPA).

Maximize the expected teacher-reference token advantage on rejected responses.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from baselines.preference.common import _check_pairwise, _reduction_term, pairwise
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.models.forward import FeatureRequest, SideRequest


def advantage_expectation_loss(
    student_probs: torch.Tensor,
    labels: torch.Tensor,
    adv_ids: torch.Tensor,
    adv_val: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Maximize the expected teacher-reference token advantage used by ADPA."""
    mask = labels.ne(ignore_index).clone()
    mask[:, 0] = False
    v_s = student_probs.shape[-1]
    valid = (adv_ids >= 0) & (adv_ids < v_s)
    has_valid = valid.any(dim=-1)
    keep_mask = mask & has_valid
    if not keep_mask.any():
        raise ValueError("No ADPA advantages align with non-masked labels")

    safe_ids = adv_ids.clamp(min=0, max=v_s - 1)
    q = student_probs.gather(-1, safe_ids)
    term = (q * adv_val * valid.to(student_probs.dtype)).sum(dim=-1)
    return -term[keep_mask].mean()


@CRITERIONS.register(
    "adpa",
)
class ADPA(Criterion):
    """Advantage expectation loss on rejected responses (ADPA).

    Chosen response is index 0, rejected is index 1.
    Formula: -mean_{(r,t) rejected, supervised} sum_k q[ids_k] * A_k.
    """

    needs_teacher: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ("cache.adv_ids", "cache.adv_val", "group.size")

    class Config(BaseConfig):
        reduction: str = "mean"
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx: LossContext) -> LossOutput:
        _check_pairwise(ctx.batch, "adpa")
        labels = ctx.batch["student.labels"]
        adv_ids = ctx.batch["cache.adv_ids"]
        adv_val = ctx.batch["cache.adv_val"]

        target_probs = torch.cat(
            [c.logits.softmax(dim=-1) for c in ctx.student_logits().chunks(size=self.cfg.chunk_size)], dim=1
        )
        _, rej_probs = pairwise(target_probs)
        _, rej_labels = pairwise(labels)
        _, rej_ids = pairwise(adv_ids)
        _, rej_val = pairwise(adv_val)

        loss = advantage_expectation_loss(rej_probs, rej_labels, rej_ids, rej_val)
        term = _reduction_term(loss.unsqueeze(0), self.cfg.reduction, 2)
        return LossOutput({"adpa": term}, metrics={"loss": loss.detach()})
