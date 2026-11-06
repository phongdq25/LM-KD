"""Reference implementation of Thinking Machines (Tinker) loss from OPSD.
Source: Self-Distill/OPSD-main/opsd_trainer.py:705-726
"""
from __future__ import annotations

import torch


def opsd_tinker_loss_ref(
    student_log_probs_sampled: torch.Tensor,
    teacher_log_probs_sampled: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact logic from OPSD-main/opsd_trainer.py:705-726."""
    advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()

    if labels is not None:
        mask = labels != -100
        advantage = advantage[mask]
        student_log_probs_sampled_masked = student_log_probs_sampled[mask]
    else:
        student_log_probs_sampled_masked = student_log_probs_sampled

    loss = -(advantage * student_log_probs_sampled_masked).mean()
    return loss

