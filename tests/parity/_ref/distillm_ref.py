"""Reference implementation of DistiLLM divergence losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def forward_kl(student_logits, teacher_logits, no_model_batch):
    mask = (no_model_batch["label"] != -100).float()
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    loss = -torch.sum(teacher_probs * student_log_probs, dim=-1)
    return (loss * mask).sum() / mask.sum()


def reverse_kl(student_logits, teacher_logits, no_model_batch):
    mask = (no_model_batch["label"] != -100).float()
    student_probs = F.softmax(student_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    loss = torch.sum(student_probs * (student_log_probs - teacher_log_probs), dim=-1)
    return (loss * mask).sum() / mask.sum()


def symmetric_kl(student_logits, teacher_logits, no_model_batch, lam=0.9):
    return (1.0 - lam) * forward_kl(student_logits, teacher_logits, no_model_batch) + lam * reverse_kl(
        student_logits, teacher_logits, no_model_batch
    )


def js_distance(student_logits, teacher_logits, no_model_batch, lam=0.9):
    mask = (no_model_batch["label"] != -100).float()
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    m = (1.0 - lam) * teacher_probs + lam * student_probs
    log_m = torch.log(m)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    loss = lam * torch.sum(student_probs * (student_log_probs - log_m), dim=-1) + (
        1.0 - lam
    ) * torch.sum(teacher_probs * (teacher_log_probs - log_m), dim=-1)
    return (loss * mask).sum() / mask.sum()


def tv_distance(student_logits, teacher_logits, no_model_batch):
    mask = (no_model_batch["label"] != -100).float()
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    loss = 0.5 * torch.sum(torch.abs(student_probs - teacher_probs), dim=-1)
    return (loss * mask).sum() / mask.sum()


def skewed_forward_kl(student_logits, teacher_logits, no_model_batch, lam=0.1):
    mask = (no_model_batch["label"] != -100).float()
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    m = lam * teacher_probs + (1.0 - lam) * student_probs
    loss = -torch.sum(teacher_probs * torch.log(m), dim=-1)
    return (loss * mask).sum() / mask.sum()


def skewed_reverse_kl(student_logits, teacher_logits, no_model_batch, lam=0.1):
    mask = (no_model_batch["label"] != -100).float()
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    m = (1.0 - lam) * teacher_probs + lam * student_probs
    loss = torch.sum(student_probs * (F.log_softmax(student_logits, dim=-1) - torch.log(m)), dim=-1)
    return (loss * mask).sum() / mask.sum()


def csd(student_logits, teacher_logits, no_model_batch):
    mask = (no_model_batch["label"] != -100).float()
    student_probs = F.softmax(student_logits, dim=-1)
    delta = student_logits - teacher_logits
    c = torch.sum(student_probs * delta, dim=-1, keepdim=True)
    target = (delta - c) * student_probs
    loss = torch.sum(target.detach() * student_logits, dim=-1)
    return (loss * mask).sum() / mask.sum()

