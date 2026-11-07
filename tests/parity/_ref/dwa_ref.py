"""Reference implementation of DWA-KD."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from numba import jit, prange


@jit(nopython=True, parallel=True)
def _ref_compute_softdtw(D: np.ndarray, gamma: float, bandwidth: float) -> np.ndarray:
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0.0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                if 0 < bandwidth < np.abs(i - j):
                    continue
                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


@jit(nopython=True, parallel=True)
def _ref_compute_softdtw_backward(D_: np.ndarray, R: np.ndarray, gamma: float, bandwidth: float) -> np.ndarray:
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1 : N + 1, 1 : M + 1] = D_
    E[:, -1, -1] = 1.0
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf
                if 0 < bandwidth < np.abs(i - j):
                    continue
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1 : N + 1, 1 : M + 1]


class RefSoftDTWFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        D_ = D.detach().cpu().float().numpy()
        R = torch.tensor(_ref_compute_softdtw(D_, float(gamma), float(bandwidth)), device=dev, dtype=dtype)
        ctx.save_for_backward(D, R)
        ctx.gamma = float(gamma)
        ctx.bandwidth = float(bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R = ctx.saved_tensors
        D_ = D.detach().cpu().float().numpy()
        R_ = R.detach().cpu().float().numpy()
        E = torch.tensor(
            _ref_compute_softdtw_backward(D_, R_, ctx.gamma, ctx.bandwidth),
            device=dev,
            dtype=dtype,
        )
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


def ref_softdtw(D: torch.Tensor, gamma: float = 1.0, bandwidth: float = 0.0) -> torch.Tensor:
    """Reference Soft-DTW forward pass."""
    return RefSoftDTWFunction.apply(D, gamma, bandwidth)


def ref_compute_dwa_band_penalty(
    A: torch.Tensor,
    s_len: int,
    t_len: int,
    band_width: float = 5.0,
    band_penalty: float = 1.0,
    center_blend: float = 0.7,
    entropy_coef: float = 2.0,
    pen_scale: float = 1.0,
) -> torch.Tensor:
    """Reference implementation of DWA banding."""
    eps = 1e-9
    A_slice = A[:s_len, :t_len]
    A_clamped = (A_slice + eps) / (A_slice.sum(dim=-1, keepdim=True) + eps)
    row_entropy = -(A_clamped * torch.log(A_clamped)).sum(dim=-1)

    lin_center = torch.arange(s_len, device=A.device, dtype=torch.float32) * (float(t_len) / float(s_len))
    soft_center = (A_clamped * torch.arange(t_len, device=A.device).view(1, -1)).sum(dim=-1)
    alpha = float(center_blend)
    centers = alpha * soft_center + (1.0 - alpha) * lin_center

    base_w = float(band_width)
    width = base_w + float(entropy_coef) * row_entropy

    j = torch.arange(t_len, device=A.device).view(1, -1).float()
    dist = (j - centers.view(-1, 1)).abs()
    band = dist <= width.view(-1, 1)

    penalty = float(band_penalty) * pen_scale
    return (~band).float() * penalty


def ref_calculate_alignment_loss(
    student_embs: torch.Tensor,
    teacher_embs: torch.Tensor,
    student_mask: torch.Tensor,
    teacher_mask: torch.Tensor,
    last_align: torch.Tensor | None = None,
    gamma: float = 1.0,
    band_width: float = 5.0,
    band_penalty: float = 1.0,
    center_blend: float = 0.7,
    entropy_coef: float = 2.0,
    pen_scale: float = 1.0,
) -> torch.Tensor:
    """Reference calculation of alignment loss."""
    batch_size = student_embs.size(0)
    total_loss = torch.tensor(0.0, device=student_embs.device, requires_grad=True)
    non_empty_pairs = 0

    for i in range(batch_size):
        s_len = student_mask[i].sum().item()
        t_len = teacher_mask[i].sum().item()

        if s_len == 0 or t_len == 0:
            continue

        non_empty_pairs += 1

        s_seq = student_embs[i, :s_len, :]
        t_seq = teacher_embs[i, :t_len, :]

        if (
            torch.isnan(s_seq).any()
            or torch.isinf(s_seq).any()
            or torch.isnan(t_seq).any()
            or torch.isinf(t_seq).any()
        ):
            non_empty_pairs -= 1
            continue

        c_stu_tea = 1.0 - F.cosine_similarity(s_seq.unsqueeze(1), t_seq.unsqueeze(0), dim=-1)
        c_stu_stu = 1.0 - F.cosine_similarity(s_seq.unsqueeze(1), s_seq.unsqueeze(0), dim=-1)
        c_tea_tea = 1.0 - F.cosine_similarity(t_seq.unsqueeze(1), t_seq.unsqueeze(0), dim=-1)

        c_stu_tea = c_stu_tea.clamp(0.0, 2.0)
        c_stu_stu = c_stu_stu.clamp(0.0, 2.0)
        c_tea_tea = c_tea_tea.clamp(0.0, 2.0)

        if last_align is not None and band_width > 0:
            pen = ref_compute_dwa_band_penalty(
                last_align[i],
                s_len=s_len,
                t_len=t_len,
                band_width=band_width,
                band_penalty=band_penalty,
                center_blend=center_blend,
                entropy_coef=entropy_coef,
                pen_scale=pen_scale,
            )
            c_stu_tea = c_stu_tea + pen

        s2t = ref_softdtw(c_stu_tea.unsqueeze(0), gamma=gamma)
        s2s = ref_softdtw(c_stu_stu.unsqueeze(0), gamma=gamma)
        t2t = ref_softdtw(c_tea_tea.unsqueeze(0), gamma=gamma)

        pair_loss = s2t - 0.5 * (s2s + t2t)

        if torch.isnan(pair_loss) or torch.isinf(pair_loss):
            non_empty_pairs -= 1
            continue

        total_loss = total_loss + pair_loss.squeeze()

    if non_empty_pairs == 0:
        return torch.tensor(0.0, device=student_embs.device, requires_grad=True)

    return total_loss

