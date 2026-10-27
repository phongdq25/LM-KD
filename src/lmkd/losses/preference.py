"""Preference distillation criteria for LM-KD.

Convention: chosen response is index 0, rejected response is index 1.
"""
from __future__ import annotations

from typing import ClassVar

import torch

from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.functional.preference import (
    advantage_expectation_loss,
    compressed_projection_kl,
    ctpd_loss,
    dp3o_loss,
    dpo_loss,
    pbsd_loss,
    ppd_loss,
    tvkd_loss,
    vpd_loss,
)
from lmkd.losses.functional.sequence import pairwise, sequence_logps
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def _check_pairwise(batch: Batch, name: str) -> int:
    n = int(batch["group.size"][0])
    valid = batch.get("group.valid")
    if n != 2 or (valid is not None and not valid.all()):
        raise ValueError(f"{name.upper()} requires exactly chosen and rejected responses")
    return n


def _reduction_term(loss: torch.Tensor, reduction: str, n: int) -> LossTerm:
    if reduction == "mean":
        return LossTerm(loss.mean(), "mean")
    if reduction == "seq_sum":
        return LossTerm(loss.sum() * n, "seq_sum")
    raise ValueError(f"Unsupported reduction: {reduction}")


@CRITERIONS.register(
    "vpd",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:67",
)
class VPD(Criterion):
    """Plackett-Luce negative log-likelihood of teacher ranking over response groups.

    Chosen response is index 0, rejected is index 1.
    Formula: -log PL(argsort_desc(teacher_scores) | beta * student_scores).
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:67-80
    """

    needs_teacher: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ("group.size", "group.valid", "pref.teacher_score")

    class Config(BaseConfig):
        beta: float = 10.0
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

        if valid.all():
            loss = vpd_loss(s, t, beta=self.cfg.beta)
        else:
            losses = [
                vpd_loss(s[i : i + 1, : int(valid[i].sum())], t[i : i + 1, : int(valid[i].sum())], beta=self.cfg.beta)
                for i in range(b)
            ]
            loss = torch.cat(losses)

        term = _reduction_term(loss, self.cfg.reduction, n)
        return LossOutput({"vpd": term}, metrics={"loss": loss.mean().detach()})


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
                ppd_loss(s[i : i + 1, : int(valid[i].sum())], t[i : i + 1, : int(valid[i].sum())], beta=self.cfg.beta, exact_ranking_limit=limit)
                for i in range(b)
            ]
            loss = torch.cat(losses)

        term = _reduction_term(loss, self.cfg.reduction, n)
        return LossOutput({"ppd": term}, metrics={"loss": loss.mean().detach()})


@CRITERIONS.register(
    "compressed_kl",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:200",
)
class CompressedKL(Criterion):
    """DCKD top-k + residual-bin KL on chosen and rejected responses.

    Chosen response is index 0, rejected is index 1.
    Formula: w_c * KL_c + w_r * KL_r on compressed teacher distributions.
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:158-221
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
                    probs[c_mask], topk_ids[:, s:e][c_mask], topk_p[:, s:e][c_mask], tail_p[:, s:e][c_mask], eps=self.cfg.eps
                )
                c_sum = c_sum + kl_c.sum()
                c_cnt += int(c_mask.sum())

            if r_mask.any():
                kl_r = compressed_projection_kl(
                    probs[r_mask], topk_ids[:, s:e][r_mask], topk_p[:, s:e][r_mask], tail_p[:, s:e][r_mask], eps=self.cfg.eps
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
    "tvkd",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:352",
)
class TVKD(Criterion):
    """Value-based preference distillation objective (TVKD).

    Chosen response is index 0, rejected is index 1.
    Formula: -log sigmoid(score_chosen - score_rejected) where score = sum_t (action_t - gamma * next_value_t).
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:306-388
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
            kl_c = compressed_projection_kl(probs_c[mask_c], ids_c[mask_c], p_c[mask_c], tail_c[mask_c], eps=self.cfg.eps).mean() if sub_c.any() else loss.new_zeros(())
            kl_r = compressed_projection_kl(probs_r[mask_r], ids_r[mask_r], p_r[mask_r], tail_r[mask_r], eps=self.cfg.eps).mean() if sub_r.any() else loss.new_zeros(())
            loss = loss + self.cfg.chosen_kd_weight * kl_c + self.cfg.rejected_kd_weight * kl_r

        term = _reduction_term(loss, self.cfg.reduction, n)
        return LossOutput({"tvkd": term}, metrics={"loss": loss.mean().detach()})


@CRITERIONS.register(
    "adpa",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:224",
)
class ADPA(Criterion):
    """Advantage expectation loss on rejected responses (ADPA).

    Chosen response is index 0, rejected is index 1.
    Formula: -mean_{(r,t) rejected, supervised} sum_k q[ids_k] * A_k.
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:224-248
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


@CRITERIONS.register(
    "ctpd",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:268",
)
class CTPD(Criterion):
    """Cross-tokenizer preference distillation with weighted parent-token groups.

    Chosen response is index 0, rejected is index 1.
    Formula: -log sigmoid(beta * (margin_chosen - margin_rejected)).
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:251-303
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
        groups_c, groups_r = pairwise(ctx.batch["pref.ctpd_groups"])
        w_c, w_r = pairwise(ctx.batch["pref.ctpd_w"])

        loss = ctpd_loss(tok_c, tok_r, ref_c, ref_r, groups_c, groups_r, w_c, w_r, beta=self.cfg.beta)
        term = _reduction_term(loss, self.cfg.reduction, n)
        return LossOutput({"ctpd": term}, metrics={"loss": loss.mean().detach()})


@CRITERIONS.register(
    "dpo",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/preference/losses.py:127",
)
class DPO(Criterion):
    """Direct Preference Optimization (DPO) family with reference model logps.

    Chosen response is index 0, rejected is index 1.
    Formula: -log sigmoid(beta * (lr_chosen - lr_rejected)) on policy-minus-reference log ratios.
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/preference/losses.py:127-130
    """

    needs_teacher: ClassVar[bool] = True
    requires: ClassVar[tuple[str, ...]] = ("group.size",)

    class Config(BaseConfig):
        beta: float = 0.1
        loss_type: str = "sigmoid"
        label_smoothing: float = 0.0
        average_log_prob: bool = False
        gamma: float = 0.5
        reduction: str = "mean"
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        teacher_spec = None if self.cfg.loss_type == "simpo" else TeacherSignalSpec(kind="sampled_logp")
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher_signal=teacher_spec,
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        n = _check_pairwise(ctx.batch, "dpo")
        if self.cfg.loss_type != "simpo" and (ctx.teacher is None or ctx.teacher.token_logp is None):
            raise ValueError("DPO requires teacher with sampled_logp signal")

        seq_pol, _, mask = sequence_logps(
            ctx.student_logits(),
            ctx.batch["student.labels"],
            average=self.cfg.average_log_prob,
            chunk_size=self.cfg.chunk_size,
        )
        if self.cfg.loss_type == "simpo":
            lr_c, lr_r = pairwise(seq_pol)
        else:
            assert ctx.teacher is not None and ctx.teacher.token_logp is not None
            ref_seq = (ctx.teacher.token_logp * mask).sum(-1)
            if self.cfg.average_log_prob:
                ref_seq = ref_seq / mask.sum(-1).clamp_min(1.0)
            lr = seq_pol - ref_seq
            lr_c, lr_r = pairwise(lr)

        loss = dpo_loss(
            lr_c,
            lr_r,
            beta=self.cfg.beta,
            loss_type=self.cfg.loss_type,
            label_smoothing=self.cfg.label_smoothing,
            gamma=self.cfg.gamma,
        )
        term = _reduction_term(loss, self.cfg.reduction, n)
        metrics = {
            "accuracy": (lr_c > lr_r).float().mean().detach(),
            "margin": (self.cfg.beta * (lr_c - lr_r)).mean().detach(),
            "rewards/chosen": (self.cfg.beta * lr_c).mean().detach(),
            "rewards/rejected": (self.cfg.beta * lr_r).mean().detach(),
        }
        return LossOutput({"dpo": term}, metrics=metrics)


@CRITERIONS.register(
    "dp3o",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/preference/losses.py:109",
)
class DP3O(Criterion):
    """Direct Preference Distillation Optimization (DP3O) blending hard and soft DPO.

    Chosen response is index 0, rejected is index 1.
    Formula: alpha * DPO_hard + (1 - alpha) * soft_label_DPO(sigmoid(reward_diff)).
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/preference/losses.py:109-124
    """

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        alpha: float = 0.2
        beta: float = 0.05
        average_log_prob: bool = False
        reduction: str = "mean"
        chunk_size: int = 1024

    @classmethod
    def required_keys(cls, cfg: BaseConfig) -> tuple[str, ...]:
        c = cfg if isinstance(cfg, cls.Config) else cls.Config(**cfg.model_dump())
        return ("group.size", "pref.reward") if c.alpha < 1.0 else ("group.size",)

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="sampled_logp"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        n = _check_pairwise(ctx.batch, "dp3o")
        if ctx.teacher is None or ctx.teacher.token_logp is None:
            raise ValueError("DP3O requires teacher with sampled_logp signal")

        seq_pol, _, mask = sequence_logps(
            ctx.student_logits(),
            ctx.batch["student.labels"],
            average=self.cfg.average_log_prob,
            chunk_size=self.cfg.chunk_size,
        )
        ref_seq = (ctx.teacher.token_logp * mask).sum(-1)
        if self.cfg.average_log_prob:
            ref_seq = ref_seq / mask.sum(-1).clamp_min(1.0)

        lr_c, lr_r = pairwise(seq_pol - ref_seq)
        reward_diff = None
        if self.cfg.alpha < 1.0:
            rew_c, rew_r = pairwise(ctx.batch["pref.reward"])
            reward_diff = rew_c - rew_r

        loss = dp3o_loss(lr_c, lr_r, reward_diff=reward_diff, beta=self.cfg.beta, alpha=self.cfg.alpha)
        term = _reduction_term(loss, self.cfg.reduction, n)
        metrics = {
            "accuracy": (lr_c > lr_r).float().mean().detach(),
            "margin": (self.cfg.beta * (lr_c - lr_r)).mean().detach(),
        }
        return LossOutput({"dp3o": term}, metrics=metrics)


@CRITERIONS.register(
    "pbsd",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:391",
)
class PBSD(Criterion):
    """Preference-Based Self-Distillation (PBSD) on teacher and student rollouts.

    Teacher rollout is index 0, student rollout is index 1.
    Formula: delta = (T(y_t) - S(y_t)) - (T(y_s) - S(y_s)); loss = -log sigmoid(-beta * delta).
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/distillation_losses.py:391-422
    """

    needs_teacher: ClassVar[bool] = True
    requires: ClassVar[tuple[str, ...]] = ("group.size", "pref.gate")

    class Config(BaseConfig):
        beta: float = 0.1
        loss_type: str = "logistic_gap"
        normalize_by_length: bool = False
        reduction: str = "mean"
        chunk_size: int = 1024

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="sampled_logp"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        n = _check_pairwise(ctx.batch, "pbsd")
        if ctx.teacher is None or ctx.teacher.token_logp is None:
            raise ValueError("PBSD requires teacher with sampled_logp signal")

        seq_s, _, mask = sequence_logps(
            ctx.student_logits(),
            ctx.batch["student.labels"],
            average=self.cfg.normalize_by_length,
            chunk_size=self.cfg.chunk_size,
        )
        seq_t = (ctx.teacher.token_logp * mask).sum(-1)
        if self.cfg.normalize_by_length:
            seq_t = seq_t / mask.sum(-1).clamp_min(1.0)

        t_on_t, t_on_s = pairwise(seq_t)
        s_on_t, s_on_s = pairwise(seq_s)
        gate, _ = pairwise(ctx.batch["pref.gate"])

        loss = pbsd_loss(t_on_t, s_on_t, t_on_s, s_on_s, beta=self.cfg.beta, loss_type=self.cfg.loss_type, gate=gate)
        term = _reduction_term(loss.unsqueeze(0), self.cfg.reduction, n)
        return LossOutput({"pbsd": term}, metrics={"loss": loss.detach()})


@CRITERIONS.register(
    "chosen_nll",
    source="Awesome-Human-AI-Alignment-main/src/human_alignment/integrations/preference_distillation.py:607",
)
class ChosenNLL(Criterion):
    """Supervised token negative log-likelihood on chosen response rows only.

    Chosen response is index 0, rejected is index 1.
    Formula: -mean_{(r,t) in chosen, supervised} log p(y_t).
    Source: Awesome-Human-AI-Alignment-main/src/human_alignment/integrations/preference_distillation.py:607-618
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
