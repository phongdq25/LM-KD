"""Reference implementation from Self-Distill/SDPO-main/verl/trainer/ppo/core_algos.py:1085-1163.
Extracts the top-k / full-logit self distillation loss, returning per_token_loss.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F


def compute_self_distillation_loss(
    student_topk_log_probs: torch.Tensor | None = None,
    teacher_topk_log_probs: torch.Tensor | None = None,
    student_all_log_probs: torch.Tensor | None = None,
    teacher_all_log_probs: torch.Tensor | None = None,
    alpha: float = 0.0,
    distillation_topk: int | None = None,
    distillation_add_tail: bool = False,
    full_logit_distillation: bool = True,
) -> torch.Tensor:
    cfg = SimpleNamespace(
        full_logit_distillation=full_logit_distillation,
        distillation_topk=distillation_topk,
        distillation_add_tail=distillation_add_tail,
        alpha=alpha,
    )

    if cfg.full_logit_distillation:
        use_topk = cfg.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError("top-k distillation requires student_topk_log_probs and teacher_topk_log_probs.")

            def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                # Compute tail log-probability using logsumexp for numerical stability
                # log(1 - sum(p_i)) = log(1 - exp(log_sum_exp(log(p_i))))
                log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)  # Clamp to avoid log_s >= 0 (which implies sum(probs) >= 1)
                tail_log = torch.log(-torch.expm1(log_s))  # 1 - exp(x) = -(exp(x) - 1)
                return torch.cat([log_probs, tail_log], dim=-1)

            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ

            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs
            if cfg.distillation_add_tail:
                student_distill_log_probs = add_tail(student_distill_log_probs)
                teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError("full_logit_distillation requires student_all_log_probs and teacher_all_log_probs.")
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

        if cfg.alpha == 0.0:
            kl_loss = F.kl_div(
                student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True
            )
        elif cfg.alpha == 1.0:
            kl_loss = F.kl_div(
                teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True
            )
        else:
            # Compute the log of the mixture distribution
            alpha_tensor = torch.tensor(
                cfg.alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack([
                    student_distill_log_probs + torch.log(1 - alpha_tensor),
                    teacher_distill_log_probs + torch.log(alpha_tensor),
                ]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha_tensor)

        per_token_loss = kl_loss.sum(-1)
        return per_token_loss
    raise NotImplementedError

