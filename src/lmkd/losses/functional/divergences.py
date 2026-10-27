"""Pure per-token divergence functions on fp32 LOGITS chunks (DESIGN §5.3).

Signature: ``fn(s_logits [B,t,V], t_logits [B,t,V]) -> [B,t]``. Teacher logits arrive detached.
They are plugged into :func:`lmkd.models.logits_view.chunked_token_divergence`.

Implemented divergences (TASK T04):
| Registry name | Function | Per-token formula (matching original code) | Parameters |
|---|---|---|---|
| `fkl` | `forward_kl` | `−Σ p·lq` (inf mask evaluates `s`) | — |
| `rkl` | `reverse_kl` | `Σ q·(lq − lp)` | — |
| `symkl` | `symmetric_kl(lam)` | `(1−lam)·fkl + lam·rkl` | `lam` (0.9) |
| `jsd` | `js_distance(lam)` | `m = (1−lam)·p + lam·q`; `lm = log(m)`; `L = lam·Σ q(lq − lm) + (1−lam)·Σ p(lp − lm)` | `lam` (0.9), `eps` |
| `tvd` | `tv_distance` | `0.5·Σ |p − q|` | — |
| `sfkl` | `skewed_forward_kl(alpha)` | `m = α·p + (1−α)·q`; `−Σ p·log(m)` | `alpha` (0.1) |
| `srkl` | `skewed_reverse_kl(alpha)` | `m = (1−α)·p + α·q`; `Σ q·(lq − log m)` | `alpha` (0.1) |
| `csd` | `csd_surrogate` | `Δ = s − t`; `c = Σ q·Δ` (keepdim); `L = Σ_v sg[(Δ − c)·q]·s` (mode "SS") | — |
| `gjsd` | `generalized_jsd(beta, token_clip, clip_mode)` | TRL/OPSD: `beta==0` → `p·(lp − lq)`; `beta==1` → `q·(lq − lp)`; else: mixture. clip elementwise/per_token | `beta` (0.5), `token_clip` (None), `clip_mode` ("elementwise") |
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F

TokenFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _inf_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.isinf(x)


def forward_kl(s_logits: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
    """DistiLLM ``forward_kl`` (distillm/losses.py:4): cross-entropy form ``-sum_v p_T log q_S``
    (teacher entropy omitted: same gradient). Infinite student logits are masked."""
    t_prob = F.softmax(t_logits, dim=-1)
    s_logp = F.log_softmax(s_logits, dim=-1)
    prod = (t_prob * s_logp).masked_fill(_inf_mask(s_logits), 0.0)
    return -prod.sum(-1)


def reverse_kl(s_logits: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
    """``sum_v q_S (log q_S - log p_T)`` (distillm/losses.py:14)."""
    s_prob = F.softmax(s_logits, dim=-1)
    s_logp = F.log_softmax(s_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    prod = (s_prob * (s_logp - t_logp)).masked_fill(inf, 0.0)
    return prod.sum(-1)


def symmetric_kl(s_logits: torch.Tensor, t_logits: torch.Tensor, lam: float = 0.9) -> torch.Tensor:
    """DistiLLM ``symmetric_kl`` (distillm/losses.py:26): ``(1 - lam) * fkl + lam * rkl``."""
    return (1.0 - lam) * forward_kl(s_logits, t_logits) + lam * reverse_kl(s_logits, t_logits)


def js_distance(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    lam: float = 0.9,
    eps: float = 0.0,
) -> torch.Tensor:
    """Jensen-Shannon distance (distillm/losses.py:32).
    m = (1 - lam) * p + lam * q; lm = log(m + eps)
    L = lam * sum q(lq - lm) + (1 - lam) * sum p(lp - lm)
    """
    p = F.softmax(t_logits, dim=-1)
    q = F.softmax(s_logits, dim=-1)
    m = (1.0 - lam) * p + lam * q
    lm = torch.log(m + eps) if eps > 0.0 else torch.log(m)

    lp = F.log_softmax(t_logits, dim=-1)
    lq = F.log_softmax(s_logits, dim=-1)
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)

    term_s = (q * (lq - lm)).masked_fill(inf, 0.0)
    term_t = (p * (lp - lm)).masked_fill(inf, 0.0)
    return lam * term_s.sum(-1) + (1.0 - lam) * term_t.sum(-1)


def tv_distance(s_logits: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
    """Total Variation distance: ``0.5 * sum |p - q|`` (distillm/losses.py:55)."""
    p = F.softmax(t_logits, dim=-1)
    q = F.softmax(s_logits, dim=-1)
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    diff = (0.5 * torch.abs(p - q)).masked_fill(inf, 0.0)
    return diff.sum(-1)


def skewed_forward_kl(s_logits: torch.Tensor, t_logits: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Skewed forward KL (distillm/losses.py:66).
    m = alpha * p + (1 - alpha) * q; -sum p * log(m)
    """
    p = F.softmax(t_logits, dim=-1)
    q = F.softmax(s_logits, dim=-1)
    m = alpha * p + (1.0 - alpha) * q
    lm = torch.log(m)
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    prod = (p * lm).masked_fill(inf, 0.0)
    return -prod.sum(-1)


def skewed_reverse_kl(s_logits: torch.Tensor, t_logits: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Skewed reverse KL (distillm/losses.py:80).
    m = (1 - alpha) * p + alpha * q; sum q * (lq - log(m))
    """
    p = F.softmax(t_logits, dim=-1)
    q = F.softmax(s_logits, dim=-1)
    m = (1.0 - alpha) * p + alpha * q
    lq = F.log_softmax(s_logits, dim=-1)
    lm = torch.log(m)
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    prod = (q * (lq - lm)).masked_fill(inf, 0.0)
    return prod.sum(-1)


def csd_surrogate(s_logits: torch.Tensor, t_logits: torch.Tensor, mode: str = "SS") -> torch.Tensor:
    """CSD surrogate loss (distillm/losses.py:97). Mode 'SS'.
    Delta = s - t; c = sum q * Delta (keepdim); L = sum_v sg[(Delta - c) * q] * s
    """
    if mode != "SS":
        raise NotImplementedError(f"CSD mode '{mode}' is not implemented (dead code in original)")
    q = F.softmax(s_logits, dim=-1)
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    delta = torch.where(inf, torch.zeros_like(s_logits), s_logits - t_logits)
    c = (q * delta).sum(dim=-1, keepdim=True)
    loss = (delta - c).detach() * q.detach() * s_logits
    loss = loss.masked_fill(inf, 0.0)
    return loss.sum(dim=-1)


def generalized_jsd(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    beta: float = 0.5,
    token_clip: float | None = None,
    clip_mode: str = "elementwise",
) -> torch.Tensor:
    """Generalized Jensen-Shannon Divergence (TRL / OPSD opsd_trainer.py:382).
    beta=0: KL(T||S) full; beta=1: KL(S||T) full.
    otherwise: mixture distribution with student weight (1 - beta) and teacher weight beta.
    """
    if clip_mode not in ("elementwise", "per_token"):
        raise ValueError(f"unknown clip_mode: {clip_mode}, expected 'elementwise' or 'per_token'")

    s_logp = F.log_softmax(s_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)

    if beta == 0.0:
        jsd = F.kl_div(s_logp, t_logp, reduction="none", log_target=True)
    elif beta == 1.0:
        jsd = F.kl_div(t_logp, s_logp, reduction="none", log_target=True)
    else:
        beta_tensor = torch.tensor(beta, dtype=s_logp.dtype, device=s_logp.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([s_logp + torch.log1p(-beta_tensor), t_logp + torch.log(beta_tensor)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, t_logp, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, s_logp, reduction="none", log_target=True)
        jsd = beta * kl_teacher + (1.0 - beta) * kl_student

    inf = _inf_mask(s_logits) | _inf_mask(t_logits)

    if token_clip is not None and clip_mode == "elementwise":
        jsd = jsd.clamp(max=token_clip)

    jsd = jsd.masked_fill(inf, 0.0)
    per_tok = jsd.sum(-1)

    if token_clip is not None and clip_mode == "per_token":
        per_tok = per_tok.clamp(max=token_clip)

    return per_tok


DIVERGENCES: dict[str, Callable[..., TokenFn]] = {
    "fkl": lambda **kw: forward_kl,
    "rkl": lambda **kw: reverse_kl,
    "symkl": lambda lam=0.9, **kw: partial(symmetric_kl, lam=lam),
    "jsd": lambda lam=0.9, eps=0.0, **kw: partial(js_distance, lam=lam, eps=eps),
    "tvd": lambda **kw: tv_distance,
    "sfkl": lambda alpha=0.1, **kw: partial(skewed_forward_kl, alpha=alpha),
    "srkl": lambda alpha=0.1, **kw: partial(skewed_reverse_kl, alpha=alpha),
    "csd": lambda **kw: csd_surrogate,
    "gjsd": lambda beta=0.5, token_clip=None, clip_mode="elementwise", **kw: partial(
        generalized_jsd, beta=beta, token_clip=token_clip, clip_mode=clip_mode
    ),
}


def get_divergence(name: str, **params) -> TokenFn:
    if name not in DIVERGENCES:
        raise NotImplementedError(
            f"divergence '{name}' not implemented yet. "
            f"Available: {sorted(DIVERGENCES)}"
        )
    return DIVERGENCES[name](**params)
