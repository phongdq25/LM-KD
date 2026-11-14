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
from lmkd.models.logits_view import LogitsView


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


def _chunk_expectation(
    logits: torch.Tensor,
    ids: torch.Tensor,
    val: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    safe_ids = ids.clamp(min=0, max=logits.shape[-1] - 1)
    q = probs.gather(-1, safe_ids)
    return (q * val * valid.to(probs.dtype)).sum(dim=-1)


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

        _, rej_labels = pairwise(labels)
        _, rej_ids = pairwise(adv_ids)
        _, rej_val = pairwise(adv_val)
        mask = rej_labels.ne(-100).clone()
        mask[:, 0] = False

        view = ctx.student_logits()
        if view._logits is not None:
            rej_view = LogitsView(
                logits=view._logits[1::2],
                temperature=view.temperature,
                vocab_limit=view.vocab_limit,
            )
        else:
            assert view._hidden is not None
            rej_view = LogitsView(
                view._hidden[1::2],
                view._head,
                temperature=view.temperature,
                vocab_limit=view.vocab_limit,
            )

        if rej_view._efficient_ok():
            assert rej_view._head is not None
            w_v = rej_view._head.weight.shape[0]
            v_s = min(w_v, rej_view.vocab_limit) if rej_view.vocab_limit is not None else w_v
            valid = (rej_ids >= 0) & (rej_ids < v_s)
            keep_mask = mask & valid.any(dim=-1)
            if not keep_mask.any():
                raise ValueError("No ADPA advantages align with non-masked labels")

            def chunk_fn(x: torch.Tensor, s: int, e: int) -> torch.Tensor:
                return _chunk_expectation(x, rej_ids[:, s:e], rej_val[:, s:e], valid[:, s:e])

            terms = rej_view._apply_chunked(chunk_fn, self.cfg.chunk_size, keep_mask)
            loss = -terms[keep_mask].mean()
        else:
            terms_list: list[torch.Tensor] = []
            keep_parts: list[torch.Tensor] = []
            for c in rej_view.chunks(size=self.cfg.chunk_size):
                c_v = c.logits.shape[-1]
                c_ids = rej_ids[:, c.start : c.end]
                c_valid = (c_ids >= 0) & (c_ids < c_v)
                c_keep = mask[:, c.start : c.end] & c_valid.any(dim=-1)
                keep_parts.append(c_keep)
                if not c_keep.any():
                    terms_list.append(c.logits.new_zeros(c.logits.shape[:2]))
                    continue
                terms_list.append(_chunk_expectation(c.logits, c_ids, rej_val[:, c.start : c.end], c_valid))
            keep_mask = torch.cat(keep_parts, dim=1)
            if not keep_mask.any():
                raise ValueError("No ADPA advantages align with non-masked labels")
            loss = -torch.cat(terms_list, dim=1)[keep_mask].mean()

        term = _reduction_term(loss.unsqueeze(0), self.cfg.reduction, 2)
        return LossOutput({"adpa": term}, metrics={"loss": loss.detach()})
