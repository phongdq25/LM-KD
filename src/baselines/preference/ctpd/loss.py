"""Contrastive Teacher Preference Distillation (CTPD).

Weighted parent-token KD-TIS-DPO losses across tokenizers.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from baselines.preference.common import _check_pairwise, _reduction_term, pairwise, sequence_logps
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def _parent_sums_tensor(tok: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    b, g, _ = groups.shape
    valid_idx = (groups >= 0) & (groups < tok.shape[-1])
    safe_groups = groups.clamp(min=0, max=tok.shape[-1] - 1)
    gathered = tok.unsqueeze(1).expand(-1, g, -1).gather(-1, safe_groups)
    return (gathered * valid_idx.to(tok.dtype)).sum(dim=-1)


def _pad_groups(groups: torch.Tensor | list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(groups, torch.Tensor):
        return groups.to(device=device, dtype=torch.long)
    ts = [g.to(device=device, dtype=torch.long) for g in groups]
    ts = [g.unsqueeze(-1) if g.ndim == 1 else g for g in ts]
    max_g = max((g.shape[0] for g in ts), default=1)
    max_m = max((g.shape[1] for g in ts if g.ndim >= 2 and g.shape[0] > 0), default=1)
    out = torch.full((len(ts), max(1, max_g), max(1, max_m)), -1, dtype=torch.long, device=device)
    for i, g in enumerate(ts):
        if g.numel() > 0:
            out[i, : g.shape[0], : g.shape[1]] = g
    return out


def _pad_weights(weights: torch.Tensor | list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(weights, torch.Tensor):
        return weights.to(device=device, dtype=dtype)
    ts = [w.to(device=device, dtype=dtype).reshape(-1) for w in weights]
    max_g = max((w.shape[0] for w in ts), default=1)
    out = torch.zeros((len(ts), max(1, max_g)), dtype=dtype, device=device)
    for i, w in enumerate(ts):
        if w.numel() > 0:
            out[i, : w.shape[0]] = w
    return out


def ctpd_loss(
    policy_chosen_token_logps: torch.Tensor,
    policy_rejected_token_logps: torch.Tensor,
    reference_chosen_token_logps: torch.Tensor,
    reference_rejected_token_logps: torch.Tensor,
    chosen_parent_groups: torch.Tensor,
    rejected_parent_groups: torch.Tensor,
    chosen_weights: torch.Tensor,
    rejected_weights: torch.Tensor,
    *,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute weighted parent-token KD-TIS-DPO losses for CTPD."""
    p_c = _parent_sums_tensor(policy_chosen_token_logps, chosen_parent_groups)
    r_c = _parent_sums_tensor(reference_chosen_token_logps, chosen_parent_groups)
    p_r = _parent_sums_tensor(policy_rejected_token_logps, rejected_parent_groups)
    r_r = _parent_sums_tensor(reference_rejected_token_logps, rejected_parent_groups)

    chosen_count = p_c.shape[1]
    rejected_count = p_r.shape[1]
    if chosen_weights.shape[1] < chosen_count or rejected_weights.shape[1] < rejected_count:
        raise ValueError("CTPD weight count does not match its parent-token groups")

    chosen_margin = ((p_c - r_c) * chosen_weights[:, :chosen_count]).sum(dim=-1)
    rejected_margin = ((p_r - r_r) * rejected_weights[:, :rejected_count]).sum(dim=-1)
    return -F.logsigmoid(beta * (chosen_margin - rejected_margin))


@CRITERIONS.register(
    "ctpd",
)
class CTPD(Criterion):
    """Cross-tokenizer preference distillation with weighted parent-token groups.

    Chosen response is index 0, rejected is index 1.
    Formula: -log sigmoid(beta * (margin_chosen - margin_rejected)).
    """

    needs_teacher: ClassVar[bool] = True
    requires: ClassVar[tuple[str, ...]] = ("pref.ctpd_groups", "pref.ctpd_w", "group.size")

    class Config(BaseConfig):
        beta: float = 0.5
        average_log_prob: bool = False
        reduction: str = "mean"
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="sampled_logp"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        n = _check_pairwise(ctx.batch, "ctpd")
        if ctx.teacher is None or ctx.teacher.token_logp is None:
            raise ValueError("CTPD requires teacher with sampled_logp signal")

        _, tok, _ = sequence_logps(
            ctx.student_logits(),
            ctx.batch["student.labels"],
            average=False,
            chunk_size=self.cfg.chunk_size,
        )
        ref_tok = ctx.teacher.token_logp

        tok_c, tok_r = pairwise(tok)
        ref_c, ref_r = pairwise(ref_tok)
        groups_c, groups_r = pairwise(_pad_groups(ctx.batch["pref.ctpd_groups"], tok.device))
        w_c, w_r = pairwise(_pad_weights(ctx.batch["pref.ctpd_w"], tok.device, tok.dtype))

        loss = ctpd_loss(tok_c, tok_r, ref_c, ref_r, groups_c, groups_r, w_c, w_r, beta=self.cfg.beta)
        term = _reduction_term(loss, self.cfg.reduction, n)
        return LossOutput({"ctpd": term}, metrics={"loss": loss.mean().detach()})
