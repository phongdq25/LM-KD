"""Reference implementation of compute_universal_logit_distillation_loss."""
from __future__ import annotations

import torch


def compute_universal_logit_distillation_loss_ref(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_target: torch.Tensor,
    teacher_target: torch.Tensor,
    padding_id: int = -100,
    clone_inputs: bool = True,
) -> torch.Tensor:
    if clone_inputs:
        student_logits = student_logits.clone()
        teacher_logits = teacher_logits.clone()
        student_target = student_target.clone()
        teacher_target = teacher_target.clone()

    for i in range(student_target.shape[0]):
        stu_start_idx = student_target[i].ne(padding_id).nonzero()[0][0]
        tea_start_idx = teacher_target[i].ne(padding_id).nonzero()[0][0]
        student_target[i] = torch.cat(
            [student_target[i][stu_start_idx:], student_target[i][:stu_start_idx]], dim=0
        )
        student_logits[i] = torch.cat(
            [student_logits[i][stu_start_idx:, :], student_logits[i][:stu_start_idx, :]], dim=0
        )
        teacher_target[i] = torch.cat(
            [teacher_target[i][tea_start_idx:], teacher_target[i][:tea_start_idx]], dim=0
        )
        teacher_logits[i] = torch.cat(
            [teacher_logits[i][tea_start_idx:, :], teacher_logits[i][:tea_start_idx, :]], dim=0
        )

    student_probs = torch.softmax(student_logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    sorted_student_probs = student_probs.sort(-1, descending=True).values
    sorted_teacher_probs = teacher_probs.sort(-1, descending=True).values

    vocab_size_gap = sorted_student_probs.shape[-1] - sorted_teacher_probs.shape[-1]
    bsz, slen = sorted_student_probs.shape[0], sorted_student_probs.shape[1]
    if vocab_size_gap > 0:
        sorted_teacher_probs = torch.cat(
            [
                sorted_teacher_probs,
                torch.zeros(
                    bsz, slen, vocab_size_gap, device=teacher_probs.device, dtype=teacher_probs.dtype
                ),
            ],
            dim=-1,
        )
    elif vocab_size_gap < 0:
        sorted_student_probs = torch.cat(
            [
                sorted_student_probs,
                torch.zeros(
                    bsz, slen, -vocab_size_gap, device=student_probs.device, dtype=student_probs.dtype
                ),
            ],
            dim=-1,
        )

    uld_loss = (sorted_student_probs - sorted_teacher_probs).abs().sum(-1)
    pad_mask = student_target.ne(padding_id) & teacher_target.ne(padding_id)
    uld_loss = (uld_loss * pad_mask).sum()
    return uld_loss

