"""Pairwise/Probabilistic Preference Distillation (PPD).

Exact ranking-distribution JSD.
Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:83-130
"""

from __future__ import annotations

from itertools import permutations
from typing import ClassVar

import torch

from baselines.preference.common import (
    _reduction_term,
    _score_matrix,
    plackett_luce_log_probability,
    sequence_logps,
)
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def ppd_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    beta: float = 1.0,
    exact_ranking_limit: int = 8,
) -> torch.Tensor:
    """Probabilistic preference distillation using exact ranking-distribution JSD."""
    _score_matrix(student_scores, "student_scores")
    _score_matrix(teacher_scores, "teacher_scores")
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student_scores and teacher_scores must have the same shape")
    batch_size, response_count = student_scores.shape
    if response_count > exact_ranking_limit:
        raise ValueError(
            f"Exact PPD enumerates rankings and supports at most {exact_ranking_limit} responses; received {response_count}"
        )

    ranking_table = torch.tensor(
        tuple(permutations(range(response_count))),
        device=student_scores.device,
        dtype=torch.long,
    )
    ranking_count = ranking_table.shape[0]
    expanded_rankings = ranking_table.unsqueeze(0).expand(batch_size, -1, -1)

    def _distribution(scores: torch.Tensor) -> torch.Tensor:
        flat_scores = scores[:, None, :].expand(-1, ranking_count, -1).reshape(-1, response_count)
        flat_rankings = expanded_rankings.reshape(-1, response_count)
        log_probs = plackett_luce_log_probability(flat_scores, flat_rankings)
        log_probs = log_probs.reshape(batch_size, ranking_count)
        return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)

    student_log_probs = _distribution(beta * student_scores)
    teacher_log_probs = _distribution(beta * teacher_scores)
    mixture_log_probs = torch.logaddexp(student_log_probs, teacher_log_probs) - torch.log(
        student_scores.new_tensor(2.0)
    )
    student_kl = (student_log_probs.exp() * (student_log_probs - mixture_log_probs)).sum(dim=-1)
    teacher_kl = (teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)
    return 0.5 * (student_kl + teacher_kl)


@CRITERIONS.register(
    "ppd",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:83",
)
class PPD(Criterion):
    """Probabilistic preference distillation using exact ranking-distribution JSD.

    Chosen response is index 0, rejected is index 1.
    Formula: 0.5 * (KL(P_S || M) + KL(P_T || M)) where P is PL distribution over permutations.
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:83-130
    """

    needs_teacher: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ("group.size", "group.valid", "pref.teacher_score")

    class Config(BaseConfig):
        beta: float = 10.0
        exact_ranking_limit: int = 8
        average_log_prob: bool = True
        reduction: str = "mean"
        fill_missing_from_teacher: bool = False
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        spec = TeacherSignalSpec(kind="sampled_logp") if self.cfg.fill_missing_from_teacher else None
        return FeatureRequest(student=SideRequest(logits="view"), teacher_signal=spec)

    def forward(self, ctx: LossContext) -> LossOutput:
        seq, _, _ = sequence_logps(
            ctx.student_logits(),
            ctx.batch["student.labels"],
            average=self.cfg.average_log_prob,
            chunk_size=self.cfg.chunk_size,
        )
        n = int(ctx.batch["group.size"][0])
        b = seq.shape[0] // n
        s = seq.view(b, n)
        t = ctx.batch["pref.teacher_score"].view(b, n)
        valid = ctx.batch["group.valid"].view(b, n)

        missing = valid & t.isnan()
        if missing.any():
            if not self.cfg.fill_missing_from_teacher or ctx.teacher is None or ctx.teacher.token_logp is None:
                raise ValueError("VPD/PPD requires teacher scores")
            mask = ctx.supervised_mask()
            ref_seq = (ctx.teacher.token_logp * mask).sum(-1)
            if self.cfg.average_log_prob:
                ref_seq = ref_seq / mask.sum(-1).clamp_min(1.0)
            t = torch.where(missing, ref_seq.view(b, n), t)

        limit = self.cfg.exact_ranking_limit
        if valid.all():
            loss = ppd_loss(s, t, beta=self.cfg.beta, exact_ranking_limit=limit)
        else:
            losses = [
                ppd_loss(
                    s[i : i + 1, : int(valid[i].sum())],
                    t[i : i + 1, : int(valid[i].sum())],
                    beta=self.cfg.beta,
                    exact_ranking_limit=limit,
                )
                for i in range(b)
            ]
            loss = torch.cat(losses)

        term = _reduction_term(loss, self.cfg.reduction, n)
        return LossOutput({"ppd": term}, metrics={"loss": loss.mean().detach()})
