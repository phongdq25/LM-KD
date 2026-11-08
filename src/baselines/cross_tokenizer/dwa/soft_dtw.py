"""Differentiable Soft-DTW sequence representation alignment for DWA-KD.

Provides:
- ``SoftDTW``: Numba CUDA GPU kernel, Numba parallel CPU JIT, and vectorized PyTorch Soft-DTW.
- ``pairwise_cosine_distance``: Clamped pairwise cosine distance matrix between representations.
- ``compute_dwa_band_penalty``: Entropy-guided adaptive dynamic window (soft banding) penalty.
- ``soft_dtw_divergence``: Full soft-DTW divergence computation between student and teacher representations.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Numba Imports & CUDA initialization
# -----------------------------------------------------------------------------
try:
    from numba import config, jit, prange

    config.CUDA_DEFAULT_PTX_CC = (8, 0)
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

try:
    from numba import cuda

    # cuCtxGetDevice_v2 is broken on some CUDA builds (returns wrong/garbage device IDs).
    # numba always prefers _v2 variants; override the binding to use the plain variant.
    def _patch_numba_ctx_get_device():
        import ctypes
        try:
            from numba.cuda.cudadrv import driver as _drv_mod, drvapi as _drvapi

            _drv = _drv_mod.driver  # module-level Driver singleton
            _drv.ensure_initialized()
            _fn = ctypes.CDLL(_drv.lib._name).cuCtxGetDevice
            _fn.restype = ctypes.c_int
            _fn.argtypes = (ctypes.POINTER(_drvapi.cu_device),)

            def _wrapped(*args):
                _drv._check_ctypes_error("cuCtxGetDevice", _fn(*args))

            setattr(_drv, "cuCtxGetDevice", _wrapped)
        except Exception:
            pass

    _patch_numba_ctx_get_device()
    del _patch_numba_ctx_get_device

    _NUMBA_CUDA_AVAILABLE = cuda.is_available()
except Exception:
    _NUMBA_CUDA_AVAILABLE = False


# -----------------------------------------------------------------------------
# Numba CUDA Kernels
# -----------------------------------------------------------------------------
if _NUMBA_CUDA_AVAILABLE:

    @cuda.jit
    def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
        """CUDA kernel computing Soft-DTW forward along anti-diagonals.

        Each block processes one sample pair; threads process anti-diagonals.
        """
        b = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        I = tid
        inv_gamma = 1.0 / gamma

        for p in range(n_passes):
            J = max(0, min(p - tid, max_j - 1))
            i = I + 1
            j = J + 1
            if I + J == p and (I < max_i and J < max_j):
                if not (abs(i - j) > bandwidth > 0):
                    r0 = -R[b, i - 1, j - 1] * inv_gamma
                    r1 = -R[b, i - 1, j] * inv_gamma
                    r2 = -R[b, i, j - 1] * inv_gamma
                    rmax = max(max(r0, r1), r2)
                    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                    softmin = -gamma * (math.log(rsum) + rmax)
                    R[b, i, j] = D[b, i - 1, j - 1] + softmin
            cuda.syncthreads()

    @cuda.jit
    def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
        """CUDA kernel computing Soft-DTW backward gradient."""
        k = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        I = tid

        for p in range(n_passes):
            rev_p = n_passes - p - 1
            J = max(0, min(rev_p - tid, max_j - 1))
            i = I + 1
            j = J + 1
            if I + J == rev_p and (I < max_i and J < max_j):
                if math.isinf(R[k, i, j]):
                    R[k, i, j] = -math.inf
                if not (abs(i - j) > bandwidth > 0):
                    a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                    b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                    c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                    E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
            cuda.syncthreads()

    class _SoftDTWCUDA(torch.autograd.Function):
        """Numba CUDA autograd function for GPU-accelerated Soft-DTW."""

        @staticmethod
        def forward(ctx: Any, D: torch.Tensor, gamma: float, bandwidth: float) -> torch.Tensor:
            dev = D.device
            dtype = D.dtype
            gamma_t = torch.tensor([gamma], device=dev, dtype=torch.float32)
            bw_t = torch.tensor([bandwidth], device=dev, dtype=torch.float32)

            B = D.shape[0]
            N = D.shape[1]
            M = D.shape[2]
            threads_per_block = max(N, M)
            n_passes = 2 * threads_per_block - 1

            R = torch.ones((B, N + 2, M + 2), device=dev, dtype=torch.float32) * math.inf
            R[:, 0, 0] = 0

            D_cuda = D.detach().to(torch.float32)
            compute_softdtw_cuda[B, threads_per_block](
                cuda.as_cuda_array(D_cuda),
                gamma_t.item(),
                bw_t.item(),
                N,
                M,
                n_passes,
                cuda.as_cuda_array(R),
            )
            ctx.save_for_backward(D, R.clone(), gamma_t, bw_t)
            result = R[:, -2, -2]
            if dtype != torch.float32:
                result = result.to(dtype)
            return result

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
            dev = grad_output.device
            dtype = grad_output.dtype
            D, R, gamma_t, bw_t = ctx.saved_tensors

            B = D.shape[0]
            N = D.shape[1]
            M = D.shape[2]
            threads_per_block = max(N, M)
            n_passes = 2 * threads_per_block - 1

            D_ = torch.zeros((B, N + 2, M + 2), dtype=torch.float32, device=dev)
            D_[:, 1 : N + 1, 1 : M + 1] = D.to(torch.float32)

            R_local = R.to(torch.float32).clone()
            R_local[:, :, -1] = -math.inf
            R_local[:, -1, :] = -math.inf
            R_local[:, -1, -1] = R_local[:, -2, -2]

            E = torch.zeros((B, N + 2, M + 2), dtype=torch.float32, device=dev)
            E[:, -1, -1] = 1

            compute_softdtw_backward_cuda[B, threads_per_block](
                cuda.as_cuda_array(D_),
                cuda.as_cuda_array(R_local),
                1.0 / gamma_t.item(),
                bw_t.item(),
                N,
                M,
                n_passes,
                cuda.as_cuda_array(E),
            )
            E = E[:, 1 : N + 1, 1 : M + 1]
            if dtype != torch.float32:
                E = E.to(dtype)
            return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# -----------------------------------------------------------------------------
# Numba Parallel CPU Kernels
# -----------------------------------------------------------------------------
if _NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True)
    def _compute_softdtw_numba(D: np.ndarray, gamma: float, bandwidth: float) -> np.ndarray:
        B, N, M = D.shape
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
    def _compute_softdtw_backward_numba(
        D_: np.ndarray, R: np.ndarray, gamma: float, bandwidth: float
    ) -> np.ndarray:
        B, N, M = D_.shape
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

    class _SoftDTWNumba(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, D: torch.Tensor, gamma: float, bandwidth: float) -> torch.Tensor:
            dev = D.device
            dtype = D.dtype
            D_ = D.detach().cpu().float().numpy()
            R = torch.tensor(_compute_softdtw_numba(D_, float(gamma), float(bandwidth)), device=dev, dtype=dtype)
            ctx.save_for_backward(D, R)
            ctx.gamma = float(gamma)
            ctx.bandwidth = float(bandwidth)
            return R[:, -2, -2]

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
            dev = grad_output.device
            dtype = grad_output.dtype
            D, R = ctx.saved_tensors
            D_ = D.detach().cpu().float().numpy()
            R_ = R.detach().cpu().float().numpy()
            E = torch.tensor(
                _compute_softdtw_backward_numba(D_, R_, ctx.gamma, ctx.bandwidth),
                device=dev,
                dtype=dtype,
            )
            return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# -----------------------------------------------------------------------------
# Pairwise distance & DWA Soft Banding
# -----------------------------------------------------------------------------
def pairwise_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine distance in [0, 2] between sequence x and sequence y."""
    cos = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)
    return (1.0 - cos).clamp(0.0, 2.0)


def compute_dwa_band_penalty(
    align_matrix: torch.Tensor,
    s_len: int,
    t_len: int,
    band_width: float = 5.0,
    band_penalty: float = 1.0,
    center_blend: float = 0.7,
    entropy_coef: float = 2.0,
    warmup_scale: float = 1.0,
) -> torch.Tensor:
    """Calculate entropy-guided adaptive window (soft band) penalty matrix.

    """
    if s_len <= 0 or t_len <= 0 or band_width <= 0:
        return torch.zeros((s_len, t_len), device=align_matrix.device, dtype=align_matrix.dtype)

    A = align_matrix[:s_len, :t_len]
    eps = 1e-9
    A_clamped = (A + eps) / (A.sum(dim=-1, keepdim=True) + eps)
    row_entropy = -(A_clamped * torch.log(A_clamped + eps)).sum(dim=-1)

    device = A.device
    lin_center = torch.arange(s_len, device=device, dtype=torch.float32) * (float(t_len) / float(s_len))
    soft_center = (A_clamped * torch.arange(t_len, device=device, dtype=torch.float32).view(1, -1)).sum(dim=-1)
    alpha = float(center_blend)
    centers = alpha * soft_center + (1.0 - alpha) * lin_center

    base_w = float(band_width)
    width = base_w + float(entropy_coef) * row_entropy

    j = torch.arange(t_len, device=device, dtype=torch.float32).view(1, -1)
    dist = (j - centers.view(-1, 1)).abs()
    band = dist <= width.view(-1, 1)

    penalty = float(band_penalty) * float(warmup_scale)
    return (~band).to(A.dtype) * penalty


# -----------------------------------------------------------------------------
# Anti-diagonal Vectorized PyTorch Soft-DTW (Fallback & Non-CUDA)
# -----------------------------------------------------------------------------
def softdtw_antidiag_forward(
    D: torch.Tensor,
    gamma: float = 1.0,
    bandwidth: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass of soft-DTW along anti-diagonals in pure PyTorch."""
    B, N, M = D.shape
    device = D.device
    dtype = D.dtype

    calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    D_c = D.to(calc_dtype)
    R = torch.full((B, N + 2, M + 2), 1e9, device=device, dtype=calc_dtype)
    R[:, 0, 0] = 0.0

    gamma_f = float(gamma)
    bw = float(bandwidth)

    for p in range(2, N + M + 1):
        i_min = max(1, p - M)
        i_max = min(N, p - 1)
        if i_min > i_max:
            continue
        i = torch.arange(i_min, i_max + 1, device=device)
        j = p - i

        r0 = R[:, i - 1, j - 1]
        r1 = R[:, i - 1, j]
        r2 = R[:, i, j - 1]
        stacked = torch.stack([r0, r1, r2], dim=-1)
        softmin = -gamma_f * torch.logsumexp(-stacked / gamma_f, dim=-1)
        d_val = D_c[:, i - 1, j - 1]

        if bw > 0:
            in_band = (i - j).abs() <= bw
            R[:, i, j] = torch.where(in_band.unsqueeze(0), d_val + softmin, R[:, i, j])
        else:
            R[:, i, j] = d_val + softmin

    val = R[:, N, M].to(dtype)
    return val, R


# -----------------------------------------------------------------------------
# Main SoftDTW Module
# -----------------------------------------------------------------------------
class SoftDTW(nn.Module):
    """Differentiable Soft-DTW with Numba CUDA, Numba CPU, or PyTorch anti-diagonal backend."""

    def __init__(
        self,
        gamma: float = 1.0,
        bandwidth: float = 0.0,
        normalize: bool = False,
        backend: Literal["auto", "cuda", "numba", "antidiag"] = "auto",
        alignment_postprocess: Literal["row", "none"] = "row",
        use_cuda: bool | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.bandwidth = float(bandwidth) if bandwidth is not None else 0.0
        self.normalize = normalize
        self.backend = backend
        self.alignment_postprocess = alignment_postprocess
        self.use_cuda = use_cuda

    def _select_func(self, C: torch.Tensor):
        # Determine whether to use CUDA kernel
        if self.use_cuda is True or (self.use_cuda is None and self.backend in ("auto", "cuda")):
            if C.is_cuda and _NUMBA_CUDA_AVAILABLE:
                max_len = max(C.shape[1], C.shape[2])
                if max_len <= 1024:
                    return _SoftDTWCUDA.apply
        # Numba CPU JIT
        if self.backend == "numba" and _NUMBA_AVAILABLE and not C.is_cuda:
            return _SoftDTWNumba.apply
        # Fallback to PyTorch anti-diagonal
        return None

    def forward_with_cost_matrix(
        self,
        C: torch.Tensor,
        return_alignment: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the soft-DTW value using a pre-computed cost matrix C."""
        assert C.dim() == 3, f"Cost matrix C must be 3-dimensional [B, N, M], got {C.shape}"

        func_dtw = self._select_func(C)
        if func_dtw is not None:
            if not return_alignment:
                return func_dtw(C, self.gamma, self.bandwidth)
            C_det = C.clone().detach().requires_grad_(True)
            vals = func_dtw(C_det, self.gamma, self.bandwidth)
            grad_out = torch.ones_like(vals)
            A = torch.autograd.grad(vals, C_det, grad_outputs=grad_out, retain_graph=True)[0]
            if self.alignment_postprocess == "row":
                A = A.clamp_min(0.0)
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-9)
            return vals, A

        # PyTorch anti-diagonal vectorized implementation
        if not return_alignment:
            val, _ = softdtw_antidiag_forward(C, self.gamma, self.bandwidth)
            return val

        C_det = C.clone().detach().requires_grad_(True)
        val, _ = softdtw_antidiag_forward(C_det, self.gamma, self.bandwidth)
        grad_out = torch.ones_like(val)
        A = torch.autograd.grad(val, C_det, grad_outputs=grad_out, retain_graph=True)[0]
        if self.alignment_postprocess == "row":
            A = A.clamp_min(0.0)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-9)
        return val, A

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        return_alignment: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute soft-DTW between representations X and Y."""
        if self.normalize:
            if return_alignment:
                raise NotImplementedError("Alignment return is not implemented for normalize=True")
            d_xy = pairwise_cosine_distance(X[0], Y[0]).unsqueeze(0)
            d_xx = pairwise_cosine_distance(X[0], X[0]).unsqueeze(0)
            d_yy = pairwise_cosine_distance(Y[0], Y[0]).unsqueeze(0)
            v_xy = self.forward_with_cost_matrix(d_xy, return_alignment=False)
            v_xx = self.forward_with_cost_matrix(d_xx, return_alignment=False)
            v_yy = self.forward_with_cost_matrix(d_yy, return_alignment=False)
            return v_xy - 0.5 * (v_xx + v_yy)

        C = pairwise_cosine_distance(X[0], Y[0]).unsqueeze(0)
        return self.forward_with_cost_matrix(C, return_alignment=return_alignment)


def soft_dtw_divergence(
    dtw: SoftDTW,
    s_seq: torch.Tensor,
    t_seq: torch.Tensor,
    align_matrix: torch.Tensor | None = None,
    band_width: float = 5.0,
    band_penalty: float = 1.0,
    center_blend: float = 0.7,
    entropy_coef: float = 2.0,
    band_warmup_scale: float = 1.0,
    band_source: Literal["cma", "sdtw"] = "cma",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute pairwise Soft-DTW divergence D_sDTW(s, t) with DWA soft banding penalty."""
    s_len = s_seq.shape[0]
    t_len = t_seq.shape[0]

    c_stu_tea = pairwise_cosine_distance(s_seq, t_seq)
    c_stu_stu = pairwise_cosine_distance(s_seq, s_seq)
    c_tea_tea = pairwise_cosine_distance(t_seq, t_seq)

    A_used = None
    if band_width > 0:
        if band_source == "cma" and align_matrix is not None:
            A_used = align_matrix[:s_len, :t_len]
            pen = compute_dwa_band_penalty(
                A_used,
                s_len=s_len,
                t_len=t_len,
                band_width=band_width,
                band_penalty=band_penalty,
                center_blend=center_blend,
                entropy_coef=entropy_coef,
                warmup_scale=band_warmup_scale,
            )
            c_stu_tea = c_stu_tea + pen
        elif band_source == "sdtw":
            _, A_sdtw = dtw.forward_with_cost_matrix(c_stu_tea.unsqueeze(0), return_alignment=True)
            A_used = A_sdtw[0]
            pen = compute_dwa_band_penalty(
                A_used,
                s_len=s_len,
                t_len=t_len,
                band_width=band_width,
                band_penalty=band_penalty,
                center_blend=center_blend,
                entropy_coef=entropy_coef,
                warmup_scale=band_warmup_scale,
            )
            c_stu_tea = c_stu_tea + pen

    s2t = dtw.forward_with_cost_matrix(c_stu_tea.unsqueeze(0))
    s2s = dtw.forward_with_cost_matrix(c_stu_stu.unsqueeze(0))
    t2t = dtw.forward_with_cost_matrix(c_tea_tea.unsqueeze(0))
    pair_loss = s2t - 0.5 * (s2s + t2t)
    return pair_loss.squeeze(), A_used
