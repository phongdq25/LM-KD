"""Dual-Constraint Knowledge Distillation (DCKD).

Compressed top-k projection KL + Chosen response NLL.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from baselines.preference.common import _check_pairwise
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest


def compressed_projection_kl(
    student_probs: torch.Tensor,
    ids: torch.Tensor,
    p: torch.Tensor,
    tail_p: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Token-level KL(teacher || student) using compressed top-k teacher distribution."""
    if student_probs.ndim != 2:
        raise ValueError("student_probs must have shape [tokens, vocab]")
    v_s = student_probs.shape[-1]
    in_range = (ids >= 0) & (ids < v_s)
    res_prime = tail_p + (p * (~in_range).to(p.dtype)).sum(dim=-1)
    safe_ids = ids.clamp(min=0, max=v_s - 1)
    q_sel = student_probs.gather(-1, safe_ids) * in_range.to(student_probs.dtype)
    q_res = (1.0 - q_sel.sum(dim=-1, keepdim=True)).clamp_min(0.0)

    keep = torch.cat([in_range, in_range.new_ones(*in_range.shape[:-1], 1)], dim=-1)
    concat_s = torch.cat([q_sel, q_res], dim=-1)
    concat_p = torch.cat([p * in_range.to(p.dtype), res_prime.unsqueeze(-1)], dim=-1)

    s = torch.where(keep, concat_s.clamp_min(eps), concat_s.new_zeros(1))
    s = s / s.sum(dim=-1, keepdim=True)
    p_proj = torch.where(keep, concat_p.clamp_min(0.0), concat_p.new_zeros(1))
    p_sum = p_proj.sum(dim=-1, keepdim=True)
    if (p_sum <= 0).any():
        raise ValueError("Compressed teacher distribution has no positive probability mass")
    p_proj = p_proj / p_sum

    return torch.where(
        keep & (p_proj > 0),
        p_proj * (p_proj.clamp_min(eps).log() - s.log()),
        torch.zeros_like(p_proj),
    ).sum(dim=-1)


def _project_support(
    flat_probs: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_p: torch.Tensor,
    flat_tail: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    v_s = flat_probs.shape[-1]
    in_range = (flat_ids >= 0) & (flat_ids < v_s)
    res_prime = flat_tail + (flat_p * (~in_range).to(flat_p.dtype)).sum(dim=-1)
    safe_ids = flat_ids.clamp(min=0, max=v_s - 1)
    q_sel = flat_probs.gather(-1, safe_ids) * in_range.to(flat_probs.dtype)
    q_res = (1.0 - q_sel.sum(dim=-1, keepdim=True)).clamp_min(0.0)

    keep = torch.cat([in_range, in_range.new_ones(*in_range.shape[:-1], 1)], dim=-1)
    concat_s = torch.cat([q_sel, q_res], dim=-1)
    concat_p = torch.cat([flat_p * in_range.to(flat_p.dtype), res_prime.unsqueeze(-1)], dim=-1)

    s = torch.where(keep, concat_s.clamp_min(eps), concat_s.new_zeros(1))
    s = s / s.sum(dim=-1, keepdim=True)
    p_proj = torch.where(keep, concat_p.clamp_min(0.0), concat_p.new_zeros(1))
    p_sum = p_proj.sum(dim=-1, keepdim=True)
    if (p_sum <= 0).any():
        raise ValueError("Compressed teacher distribution has no positive probability mass")
    p_proj = p_proj / p_sum
    return s, p_proj, keep, in_range


@CRITERIONS.register(
    "compressed_kl",
)
class CompressedKL(Criterion):
    """DCKD top-k + residual-bin KL on chosen and rejected responses.

    Chosen response is index 0, rejected is index 1.
    Formula: w_c * KL_c + w_r * KL_r on compressed teacher distributions.
    """

    needs_teacher: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ("cache.topk_ids", "cache.topk_p", "cache.tail_p", "group.size")

    class Config(BaseConfig):
        chosen_weight: float = 0.1
        rejected_weight: float = 0.1
        eps: float = 1e-8
        reduction: str = "mean"
        check_unique: bool = False
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx: LossContext) -> LossOutput:
        _check_pairwise(ctx.batch, "compressed_kl")
        labels = ctx.batch["student.labels"]
        mask = ctx.supervised_mask().bool()
        topk_ids = ctx.batch["cache.topk_ids"]
        topk_p = ctx.batch["cache.topk_p"]
        tail_p = ctx.batch["cache.tail_p"]

        c_sum = labels.new_zeros((), dtype=torch.float32)
        r_sum = labels.new_zeros((), dtype=torch.float32)
        c_cnt = 0
        r_cnt = 0

        is_chosen = (torch.arange(labels.shape[0], device=labels.device) % 2) == 0

        for chunk in ctx.student_logits().chunks(size=self.cfg.chunk_size):
            s, e = chunk.start, chunk.end
            chunk_mask = mask[:, s:e]
            if not chunk_mask.any():
                continue
            probs = chunk.logits.softmax(dim=-1)
            c_mask = chunk_mask & is_chosen.unsqueeze(-1)
            r_mask = chunk_mask & (~is_chosen).unsqueeze(-1)

            if c_mask.any():
                kl_c = compressed_projection_kl(
                    probs[c_mask],
                    topk_ids[:, s:e][c_mask],
                    topk_p[:, s:e][c_mask],
                    tail_p[:, s:e][c_mask],
                    eps=self.cfg.eps,
                )
                c_sum = c_sum + kl_c.sum()
                c_cnt += int(c_mask.sum())

            if r_mask.any():
                kl_r = compressed_projection_kl(
                    probs[r_mask],
                    topk_ids[:, s:e][r_mask],
                    topk_p[:, s:e][r_mask],
                    tail_p[:, s:e][r_mask],
                    eps=self.cfg.eps,
                )
                r_sum = r_sum + kl_r.sum()
                r_cnt += int(r_mask.sum())

        if c_cnt == 0 or r_cnt == 0:
            raise ValueError("No teacher-token probabilities align with non-masked labels")

        mean_kl_c = c_sum / c_cnt
        mean_kl_r = r_sum / r_cnt
        loss = self.cfg.chosen_weight * mean_kl_c + self.cfg.rejected_weight * mean_kl_r

        if self.cfg.reduction == "mean":
            term = LossTerm(loss, "mean")
        elif self.cfg.reduction == "seq_sum":
            term = LossTerm(loss * labels.shape[0], "seq_sum")
        else:
            raise ValueError(f"Unsupported reduction: {self.cfg.reduction}")

        return LossOutput(
            {"compressed_kl": term},
            metrics={"kl_c": mean_kl_c.detach(), "kl_r": mean_kl_r.detach()},
        )


@CRITERIONS.register(
    "chosen_nll",
)
class ChosenNLL(Criterion):
    """Supervised token negative log-likelihood on chosen response rows only.

    Chosen response is index 0, rejected is index 1.
    Formula: -mean_{(r,t) in chosen, supervised} log p(y_t).
    """

    needs_teacher: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ("group.size",)

    class Config(BaseConfig):
        index: int = 0
        chunk_size: int = 1024
        reduction: str = "mean"

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx: LossContext) -> LossOutput:
        labels = ctx.batch["student.labels"]
        mask = ctx.supervised_mask()
        if "group.index" in ctx.batch:
            chosen_row = ctx.batch["group.index"] == self.cfg.index
        else:
            n = int(ctx.batch["group.size"][0])
            chosen_row = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
            chosen_row[self.cfg.index :: n] = True

        chosen_mask = mask * chosen_row.unsqueeze(-1)
        logp = ctx.student_logits().target_logp(labels, chunk_size=self.cfg.chunk_size)
        nll = -(logp * chosen_mask).sum()
        denom = chosen_mask.sum().clamp_min(1.0)

        if self.cfg.reduction == "mean":
            term = LossTerm(nll / denom, "mean")
        elif self.cfg.reduction == "seq_sum":
            n = int(ctx.batch["group.size"][0])
            term = LossTerm((nll / denom) * (labels.shape[0] // n), "seq_sum")
        else:
            raise ValueError(f"Unsupported reduction: {self.cfg.reduction}")

        return LossOutput({"chosen_nll": term}, metrics={"chosen_nll/tokens": denom.detach()})


DCKD = CompressedKL

