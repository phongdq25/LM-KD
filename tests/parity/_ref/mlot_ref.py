"""Reference implementation of MultiLevelOT helpers and compute_multi_level_ot_loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(value: torch.Tensor) -> torch.Tensor:
    stds = value.std(dim=-1, keepdim=True)
    return value / (stds + 0.0001)


def improved_sort(value: torch.Tensor) -> torch.Tensor:
    sums = value.sum(dim=(0, 1))
    sorted_indices = torch.argsort(sums, descending=True)
    sorted_values = value[:, :, sorted_indices]
    return sorted_values


def KL_wo(y_s: torch.Tensor, y_t: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    p_s = F.log_softmax(y_s / T, dim=-1)
    p_t = F.softmax(y_t / T, dim=-1)
    loss = -torch.sum(p_t * p_s, dim=-1).mean()
    return loss


class SinkhornSeqRef(nn.Module):
    def __init__(self, T: float = 2.0) -> None:
        super().__init__()
        self.T = T

    def sinkhorn_normalized(self, x: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(
        self, x: torch.Tensor, y: torch.Tensor, epsilon: float = 0.1, n_iters: int = 10
    ) -> torch.Tensor:
        Wxy = torch.cdist(x, y, p=1)
        K = torch.exp(-Wxy / epsilon)
        P = self.sinkhorn_normalized(K, n_iters)
        return torch.sum(P * Wxy)

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        softmax = nn.Softmax(dim=-1)
        p_s = softmax(y_s / self.T)
        p_t = softmax(y_t / self.T)
        emd_loss = torch.tensor(0.0, device=y_s.device, dtype=torch.float32)
        for i in range(p_s.shape[0]):
            emd_loss = emd_loss + 0.001 * self.sinkhorn_loss(x=p_s[i], y=p_t[i])
        return emd_loss


def compute_multi_level_ot_loss_ref(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_target: torch.Tensor,
    teacher_target: torch.Tensor,
    student_temperature: float = 2.0,
    teacher_temperature: float = 2.0,
    top_k: int = 50,
    f: int = 1,
    padding_id: int = -100,
    clone_inputs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (multi_loss, ot_loss, kl_loss, l1_loss)."""
    if clone_inputs:
        student_logits = student_logits.clone()
        teacher_logits = teacher_logits.clone()
        student_target = student_target.clone()
        teacher_target = teacher_target.clone()

    student_logits = normalize(student_logits)
    teacher_logits = normalize(teacher_logits)

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

    student_probs = torch.softmax(student_logits / student_temperature, dim=-1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits / teacher_temperature, dim=-1, dtype=torch.float32)

    student_probs = improved_sort(student_probs)
    teacher_probs = improved_sort(teacher_probs)

    student_probs = student_probs[:, :, :top_k]
    teacher_probs = teacher_probs[:, :, :top_k]

    diff_size = student_probs.size(2) - teacher_probs.size(2)
    if diff_size > 0:
        teacher_probs = F.pad(teacher_probs, (0, diff_size), value=0)
    elif diff_size < 0:
        student_probs = F.pad(student_probs, (0, abs(diff_size)), value=0)

    sinkhorn = SinkhornSeqRef()
    if f == 1:
        ot_loss = sinkhorn(student_probs, teacher_probs) * 0.1
    else:
        raise ValueError("Invalid value for f. Use 1.")

    kl_loss = KL_wo(student_probs, teacher_probs) * 0.1
    l1_loss = (student_probs - teacher_probs).abs().sum(-1).mean()

    multi_loss = ot_loss + kl_loss + l1_loss
    pad_mask = student_target.ne(padding_id) & teacher_target.ne(padding_id)
    multi_loss_token_scaled = (multi_loss * pad_mask).sum()

    return multi_loss_token_scaled, ot_loss, kl_loss, l1_loss

