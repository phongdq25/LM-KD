"""E-SCOT: Entropic Soft Context-aware Optimal Transport criterion (T13).

Source: white-box-distillation/cross-tokenizer/IMPACT/code/criterions/E_SCOT.py:114-280.
"""
from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.cross_tokenizer._common import pairs_to_index
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def compute_renyi_entropies(
    probs: torch.Tensor, V: int, alphas: list[float]
) -> torch.Tensor:
    """Normalized Rényi entropies for specified orders (E_SCOT.py:265-280)."""
    p = probs.clamp(min=1e-12)
    device = probs.device
    log_V = (
        torch.log(torch.tensor(float(V), device=device, dtype=torch.float32))
        if V > 1
        else torch.tensor(1.0, device=device, dtype=torch.float32)
    )
    outputs: list[torch.Tensor] = []
    for alpha in alphas:
        if abs(alpha - 1.0) < 1e-6:
            H = -(p * torch.log(p)).sum(dim=-1) / log_V
        else:
            lse = torch.logsumexp(alpha * torch.log(p), dim=-1)
            H = lse / (1.0 - alpha) / log_V
        outputs.append(H)
    return torch.stack(outputs, dim=-1)


def escot_weights(b_idx: torch.Tensor, q: torch.Tensor, lam: float = 0.25) -> torch.Tensor:
    """Compute per-student-position weights: w_i = lam + (1-lam) * M_b * q_i / sum(q)."""
    w = torch.ones_like(q)
    for b in torch.unique(b_idx):
        mask_b = b_idx == b
        m_b = mask_b.sum().float().clamp(min=1.0)
        q_b = q[mask_b]
        q_sum = q_b.sum().clamp(min=1e-8)
        w[mask_b] = lam + (1.0 - lam) * m_b * q_b / q_sum
    return w.detach()


def _pad_vocab_dim(ss: torch.Tensor, st: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Pad student and teacher sorted probabilities along vocabulary dimension."""
    v_s = ss.shape[-1]
    v_t = st.shape[-1]
    v_max = max(v_s, v_t)
    if v_s < v_max:
        ss = F.pad(ss, (0, v_max - v_s), value=0.0)
    elif v_t < v_max:
        st = F.pad(st, (0, v_max - v_t), value=0.0)
    return ss, st, v_s, v_t


@CRITERIONS.register(
    "escot",
    source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/E_SCOT.py:114-280",
)
class ESCOT(Criterion):
    """E-SCOT: Entropic Soft Context-aware Optimal Transport loss.

    Formula:
        For character-span aligned token pairs (t_S, t_T):
        d_ij = ||sort_desc(softmax(z_S[u_i] / T)) - sort_desc(softmax(z_T[u_j] / T))||_1
        delta_ij = ||H_Renyi(softmax(z_S[u_i] / T)) - H_Renyi(softmax(z_T[u_j] / T))||_1 (stop-grad)
        q_i = mean_{j in N(i)} delta_ij
        w_i = lam + (1 - lam) * M_b * q_i / sum_k q_k (stop-grad)
        loss = sum_i sum_{j in N(i)} (w_i / |N(i)|) * d_ij

    Original: IMPACT criterions/E_SCOT.py:114-280.
    """

    needs_teacher: ClassVar[bool] = True
    requires: ClassVar[tuple[str, ...]] = ("align.span_pairs",)

    class Config(BaseConfig):
        lam: float = 0.25
        alphas: list[float] = [0.5, 1.0, 2.0]
        temperature: float = 2.0
        pair_chunk: int = 512

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="full"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        device = ctx.batch["student.labels"].device
        bi, ts, tt = pairs_to_index(ctx.batch["align.span_pairs"], device)
        p_count = len(bi)
        if p_count == 0:
            return LossOutput(
                terms={self.alias: LossTerm(torch.tensor(0.0, device=device, requires_grad=True), "token_sum")},
                metrics={"escot_pairs": torch.tensor(0.0, device=device)},
            )

        pairs_s = torch.stack([bi, ts], dim=1)
        us, inv_s = torch.unique(pairs_s, dim=0, return_inverse=True)
        pairs_t = torch.stack([bi, tt], dim=1)
        ut, inv_t = torch.unique(pairs_t, dim=0, return_inverse=True)

        assert ctx.student.logits is not None
        s_rows = ctx.student.logits.at(us)
        assert ctx.teacher is not None and ctx.teacher.logits is not None
        with torch.no_grad():
            t_rows = ctx.teacher.logits.at(ut)

        ps = torch.softmax(s_rows.float() / self.cfg.temperature, dim=-1)
        pt = torch.softmax(t_rows.float() / self.cfg.temperature, dim=-1)
        ss = ps.sort(dim=-1, descending=True).values
        st = pt.sort(dim=-1, descending=True).values
        ss, st, v_s, v_t = _pad_vocab_dim(ss, st)

        h_s = compute_renyi_entropies(ps, v_s, self.cfg.alphas)
        h_t = compute_renyi_entropies(pt, v_t, self.cfg.alphas)

        d = (ss[inv_s] - st[inv_t]).abs().sum(dim=-1)
        dl = (h_s[inv_s] - h_t[inv_t]).abs().sum(dim=-1).detach()

        gid = inv_s
        u_len = us.shape[0]
        cnt = torch.zeros(u_len, device=device, dtype=torch.float32)
        cnt.scatter_add_(0, gid, torch.ones_like(gid, dtype=torch.float32))
        dsum = torch.zeros(u_len, device=device, dtype=torch.float32)
        dsum.scatter_add_(0, gid, dl)
        q = dsum / cnt

        w = escot_weights(us[:, 0], q, self.cfg.lam)
        per_pair_w = w[gid] / cnt[gid]
        loss = (per_pair_w * d).sum()

        return LossOutput(
            terms={self.alias: LossTerm(loss, "token_sum")},
            metrics={"escot_pairs": torch.tensor(float(p_count), device=device)},
        )
