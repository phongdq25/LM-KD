"""Helpers shared by the cross-tokenizer criteria (T12-T16)."""
from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmkd.models.logits_view import LogitsView


class DetachedLinear(nn.Module):
    """y = x @ W.detach().T (+ b.detach()). Used for 'lm_head.weight.detach()' (t2s, DSKD lines 70-72)."""

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = [head]  # list: NOT registered as submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = cast(Any, self.head[0])
        w: torch.Tensor = head.weight
        b: torch.Tensor | None = getattr(head, "bias", None)
        b_det = b.detach().to(x.dtype) if b is not None else None
        return F.linear(x, w.detach().to(x.dtype), b_det)


def view_from_target_hidden(h_tgt: torch.Tensor, head: nn.Module) -> LogitsView:
    """h_tgt is target-aligned ([:, t] predicts token t). LogitsView shifts right by 1,
    so we shift left beforehand: pos = cat([h_tgt[:, 1:], zeros(B, 1, D)]). Target 0 is masked.
    """
    if h_tgt.shape[1] == 0:
        pos = h_tgt
    else:
        zeros = torch.zeros_like(h_tgt[:, :1])
        pos = torch.cat([h_tgt[:, 1:], zeros], dim=1)
    return LogitsView(hidden=pos, lm_head=head)


def target_hidden(h: torch.Tensor) -> torch.Tensor:
    """Hidden positional ``[B,T,D]`` -> target-aligned: ``out[:, t] = h[:, t-1]``, ``out[:, 0] = 0``."""
    out = torch.zeros_like(h)
    out[:, 1:] = h[:, :-1]
    return out


def target_logits_full(view: LogitsView) -> torch.Tensor:
    """Target-aligned ``[B,T,V]`` logits. Materializes the full tensor: only for criteria that must see all of V."""
    return view.full()  # [B,T,V] fp32 on purpose: callers (OT / sorting / projections) need the whole vocabulary


def _masked(kld: torch.Tensor, logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    bad = logits.isinf() | logits.isnan() | teacher_logits.isinf() | teacher_logits.isnan()
    return kld.masked_fill(bad, 0.0).sum(-1)


def _scaled(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kd_temp: float,
    tea_temp: float,
    use_tea_temp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled_logits = logits / kd_temp
    scaled_tea = teacher_logits / kd_temp
    return scaled_logits, (scaled_tea / tea_temp if use_tea_temp else scaled_tea)


def _forward_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kd_temp: float,
    tea_temp: float,
    use_tea_temp: bool,
) -> torch.Tensor:
    l_s, l_t = _scaled(logits, teacher_logits, kd_temp, tea_temp, use_tea_temp)
    lprobs = torch.log_softmax(l_s, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(l_t, -1, dtype=torch.float32)
    teacher_lprobs = torch.log_softmax(l_t, -1, dtype=torch.float32)
    return _masked(teacher_probs * (teacher_lprobs - lprobs), l_s, l_t)


def _reverse_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kd_temp: float,
    tea_temp: float,
    use_tea_temp: bool,
) -> torch.Tensor:
    l_s, l_t = _scaled(logits, teacher_logits, kd_temp, tea_temp, use_tea_temp)
    probs = torch.softmax(l_s, -1, dtype=torch.float32)
    lprobs = torch.log_softmax(l_s, -1, dtype=torch.float32)
    teacher_lprobs = torch.log_softmax(l_t, -1, dtype=torch.float32)
    return _masked(probs * (lprobs - teacher_lprobs), l_s, l_t)


def _adaptive_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kd_temp: float,
    tea_temp: float,
    use_tea_temp: bool,
    alpha: float,
) -> torch.Tensor:
    probs = torch.softmax(logits / kd_temp, dim=-1, dtype=torch.float32)
    t_div = teacher_logits / tea_temp / kd_temp if use_tea_temp else teacher_logits / kd_temp
    teacher_probs = torch.softmax(t_div, dim=-1, dtype=torch.float32)
    sorted_teacher_probs, sorted_idx = teacher_probs.sort(-1)
    sorted_probs = probs.gather(-1, sorted_idx)
    gap = (sorted_teacher_probs - sorted_probs).abs()
    tail_mask = torch.cumsum(sorted_teacher_probs, -1).le(alpha).float()
    g_head = (gap * (1 - tail_mask)).sum(-1).detach()
    g_tail = (gap * tail_mask).sum(-1).detach()
    fkl = _forward_kl(logits, teacher_logits, kd_temp, tea_temp, use_tea_temp)
    rkl = _reverse_kl(logits, teacher_logits, kd_temp, tea_temp, use_tea_temp)
    total_g = g_head + g_tail
    return (g_head / total_g) * fkl + (g_tail / total_g) * rkl


def _skewed_forward_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kd_temp: float,
    tea_temp: float,
    use_tea_temp: bool,
    lam: float,
) -> torch.Tensor:
    l_s, l_t = _scaled(logits, teacher_logits, kd_temp, tea_temp, use_tea_temp)
    student_probs = torch.softmax(l_s, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(l_t, -1, dtype=torch.float32)
    mixed_lprobs = torch.log(lam * teacher_probs + (1 - lam) * student_probs)
    teacher_lprobs = torch.log_softmax(l_t, -1, dtype=torch.float32)
    return _masked(teacher_probs * (teacher_lprobs - mixed_lprobs), l_s, l_t)


def _skewed_reverse_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kd_temp: float,
    tea_temp: float,
    use_tea_temp: bool,
    lam: float,
) -> torch.Tensor:
    l_s, l_t = _scaled(logits, teacher_logits, kd_temp, tea_temp, use_tea_temp)
    student_probs = torch.softmax(l_s, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(l_t, -1, dtype=torch.float32)
    mixed_lprobs = torch.log((1 - lam) * teacher_probs + lam * student_probs)
    student_lprobs = torch.log_softmax(l_s, -1, dtype=torch.float32)
    return _masked(student_probs * (student_lprobs - mixed_lprobs), l_s, l_t)


def _js_divergence(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kd_temp: float,
    tea_temp: float,
    use_tea_temp: bool,
) -> torch.Tensor:
    l_s, l_t = _scaled(logits, teacher_logits, kd_temp, tea_temp, use_tea_temp)
    probs = torch.softmax(l_s, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(l_t, -1, dtype=torch.float32)
    m_lprobs = torch.log((probs + teacher_probs) / 2 + 1e-9)
    kld1 = teacher_probs * (torch.log(teacher_probs + 1e-9) - m_lprobs)
    kld2 = probs * (torch.log(probs + 1e-9) - m_lprobs)
    return _masked((kld1 + kld2) / 2, l_s, l_t)


def vd_divergence(name: str, logits: torch.Tensor, teacher_logits: torch.Tensor, *, kd_temp: float = 1.0,
                  tea_temp: float = 1.0, use_tea_temp: bool = False, skew_lambda: float = 0.1,
                  adaptive_alpha: float = 0.5) -> torch.Tensor:
    """Per-token ``[B,T]`` port of IMPACT's ``VariousDivergence`` (IMPACT/code/criterions/various_divergence.py:102-307,
    ``reduction='none'``); ``name`` in forward_kl | reverse_kl | adaptive_kl | skewed_forward_kl | skewed_reverse_kl |
    js_divergence.

    NOT the same as ``losses/functional/divergences.py`` (distillm): here ``forward_kl`` is the true KL (includes
    the teacher entropy term), divides the logits by ``kd_temp`` WITHOUT the ``T^2`` factor, and masks inf/nan
    entries of BOTH student and teacher logits (the distillm versions mask only the teacher side and rescale).
    """
    args = (logits, teacher_logits, kd_temp, tea_temp, use_tea_temp)
    if name == "forward_kl":
        return _forward_kl(*args)
    if name == "reverse_kl":
        return _reverse_kl(*args)
    if name == "adaptive_kl":
        return _adaptive_kl(*args, adaptive_alpha)
    if name == "skewed_forward_kl":
        return _skewed_forward_kl(*args, skew_lambda)
    if name == "skewed_reverse_kl":
        return _skewed_reverse_kl(*args, skew_lambda)
    if name == "js_divergence":
        return _js_divergence(*args)
    raise ValueError(f"unknown divergence {name!r}")


def pad_ragged_2d(items: list[torch.Tensor], pad: int = -1) -> torch.Tensor:
    """List of ``[N_i, K_i]`` tensors -> ``[B, max N, max K]`` padded with ``pad``."""
    n = max(t.shape[0] for t in items)
    k = max(t.shape[1] for t in items)
    out = torch.full((len(items), n, k), pad, dtype=items[0].dtype, device=items[0].device)
    for i, t in enumerate(items):
        out[i, : t.shape[0], : t.shape[1]] = t
    return out


def pairs_to_index(items: list[torch.Tensor], device: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """List of ``[P_b, 2]`` tensors -> ``(batch_idx[P], col0[P], col1[P])`` over the whole batch."""
    pairs = [t.reshape(-1, 2).to(device) for t in items]
    batch_idx = torch.cat([torch.full((p.shape[0],), b, dtype=torch.long, device=device) for b, p in enumerate(pairs)])
    cat = torch.cat(pairs) if pairs else torch.zeros(0, 2, dtype=torch.long, device=device)
    return batch_idx, cat[:, 0], cat[:, 1]


def get_vocab_map(ctx: Any) -> Any:
    """``ctx.rt.data_ctx.vocab_map`` or a clear error."""
    vm = getattr(getattr(getattr(ctx, "rt", None), "data_ctx", None), "vocab_map", None)
    if vm is None:
        raise RuntimeError("no vocab map: add the `vocab_map` pipeline stage or the `load_vocab_map` transform")
    return vm

