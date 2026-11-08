"""Reference implementation of E-SCOT loss and Renyi entropy helper."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.parity._ref.xtok_ref import _span_align


def compute_renyi_entropies_ref(
    probs: torch.Tensor, V: int, alphas: list[float]
) -> torch.Tensor:
    """Normalized Rényi entropies for orders in alphas."""
    p = probs.clamp(min=1e-12)
    log_V = (
        torch.log(torch.tensor(float(V), device=probs.device))
        if V > 1
        else torch.tensor(1.0, device=probs.device)
    )
    outputs = []
    for alpha in alphas:
        if abs(alpha - 1.0) < 1e-6:
            H = -(p * p.log()).sum(dim=-1) / log_V
        else:
            lse = torch.logsumexp(alpha * p.log(), dim=-1)
            H = lse / (1.0 - alpha) / log_V
        outputs.append(H)
    return torch.stack(outputs, dim=-1)


def compute_escot_loss_ref(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_labels: torch.Tensor,
    student_offsets_batch: list[list[tuple[int, int] | list[int]]],
    teacher_offsets_batch: list[list[tuple[int, int] | list[int]]],
    kd_temperature: float = 2.0,
    lam: float = 0.25,
    alphas: list[float] | None = None,
    padding_id: int = -100,
) -> torch.Tensor:
    """Compute the E-SCOT distillation loss for a batch (SUM-scale)."""
    if alphas is None:
        alphas = [0.5, 1.0, 2.0]
    device = student_logits.device
    B = student_logits.shape[0]
    tau = kd_temperature

    # Softmax at temperature tau. Cast to fp32 for stable Rényi/W-1.
    s_probs = torch.softmax(student_logits / tau, dim=-1, dtype=torch.float32)
    t_probs = torch.softmax(teacher_logits / tau, dim=-1, dtype=torch.float32)

    # Sort descending
    sorted_s = s_probs.sort(dim=-1, descending=True).values  # [B, L_S, V_S]
    sorted_t = t_probs.sort(dim=-1, descending=True).values  # [B, L_T, V_T]

    V_S = sorted_s.shape[-1]
    V_T = sorted_t.shape[-1]
    L_common = max(V_S, V_T)
    if V_S < L_common:
        sorted_s = torch.cat(
            [sorted_s, sorted_s.new_zeros(B, sorted_s.shape[1], L_common - V_S)], dim=-1
        )
    if V_T < L_common:
        sorted_t = torch.cat(
            [sorted_t, sorted_t.new_zeros(B, sorted_t.shape[1], L_common - V_T)], dim=-1
        )

    # Normalized Rényi entropies per position.
    s_ent = compute_renyi_entropies_ref(s_probs, V_S, alphas)  # [B, L_S, A]
    t_ent = compute_renyi_entropies_ref(t_probs, V_T, alphas)  # [B, L_T, A]

    total_loss = student_logits.new_zeros((), dtype=torch.float32)

    for b in range(B):
        s_valid = student_labels[b] != padding_id
        t_valid = teacher_labels[b] != padding_id
        if s_valid.sum() == 0 or t_valid.sum() == 0:
            continue

        s_offsets_b = student_offsets_batch[b]
        t_offsets_b = teacher_offsets_batch[b]
        if len(s_offsets_b) == 0 or len(t_offsets_b) == 0:
            continue

        s_pos_all = s_valid.nonzero(as_tuple=False).flatten().tolist()
        t_pos_all = t_valid.nonzero(as_tuple=False).flatten().tolist()
        s_pos = [p for p in s_pos_all if p < len(s_offsets_b)]
        t_pos = [p for p in t_pos_all if p < len(t_offsets_b)]
        if not s_pos or not t_pos:
            continue

        s_spans = [tuple(s_offsets_b[p]) for p in s_pos]
        t_spans = [tuple(t_offsets_b[p]) for p in t_pos]

        neighborhoods = _span_align(s_spans, t_spans)

        pair_i, pair_j = [], []
        for i_local, N_i in enumerate(neighborhoods):
            for j_local in N_i:
                pair_i.append(i_local)
                pair_j.append(j_local)
        if not pair_i:
            continue

        s_pos_t = torch.tensor(s_pos, device=device, dtype=torch.long)
        t_pos_t = torch.tensor(t_pos, device=device, dtype=torch.long)
        pair_i_t = torch.tensor(pair_i, device=device, dtype=torch.long)
        pair_j_t = torch.tensor(pair_j, device=device, dtype=torch.long)

        s_sort_sel = sorted_s[b].index_select(0, s_pos_t)  # [I, L]
        t_sort_sel = sorted_t[b].index_select(0, t_pos_t)  # [J, L]
        s_ent_sel = s_ent[b].index_select(0, s_pos_t)  # [I, A]
        t_ent_sel = t_ent[b].index_select(0, t_pos_t)  # [J, A]

        s_pairs = s_sort_sel.index_select(0, pair_i_t)  # [P, L]
        t_pairs = t_sort_sel.index_select(0, pair_j_t)  # [P, L]
        d_ij = (s_pairs - t_pairs).abs().sum(dim=-1)  # [P]

        s_ent_p = s_ent_sel.index_select(0, pair_i_t)  # [P, A]
        t_ent_p = t_ent_sel.index_select(0, pair_j_t)  # [P, A]
        delta_ij = (s_ent_p - t_ent_p).abs().sum(dim=-1).detach()  # [P]

        I = len(neighborhoods)
        counts = torch.zeros(I, device=device, dtype=torch.float32)
        delta_sum = torch.zeros(I, device=device, dtype=torch.float32)
        counts.scatter_add_(0, pair_i_t, torch.ones_like(pair_i_t, dtype=torch.float32))
        delta_sum.scatter_add_(0, pair_i_t, delta_ij)
        valid = counts > 0
        q = torch.zeros_like(counts)
        q[valid] = delta_sum[valid] / counts[valid]

        M_mask = valid
        M_size = M_mask.sum().float().clamp(min=1.0)
        q_sum = q[M_mask].sum().clamp(min=1e-8)
        w = torch.ones_like(q)
        w[M_mask] = lam + (1.0 - lam) * M_size * q[M_mask] / q_sum
        w = w.detach()

        per_pair_w = w.index_select(0, pair_i_t) / counts.index_select(0, pair_i_t).clamp(min=1.0)
        item_loss = (per_pair_w * d_ij).sum()
        total_loss = total_loss + item_loss

    return total_loss

