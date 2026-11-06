"""Pure per-token divergence functions on fp32 LOGITS chunks.

Signature: ``fn(s_logits [B,t,V], t_logits [B,t,V]) -> [B,t]``. Teacher logits arrive detached.
They are plugged into :func:`lmkd.models.logits_view.chunked_token_divergence`.

Implemented divergences:
| Registry name | Function | Per-token formula (matching original code) | Parameters |
|---|---|---|---|
| `fkl` | `forward_kl` | `−Σ p·lq` (inf mask evaluates `s`) | `temperature` (1.0) |
| `rkl` | `reverse_kl` | `Σ q·(lq − lp)` | `temperature` (1.0) |
| `symkl` | `symmetric_kl(lam)` | `(1−lam)·fkl + lam·rkl` | `lam` (0.9), `temperature` (1.0) |
| `jsd` | `js_distance(lam)` | `m = (1−lam)·p + lam·q`; `lm = log(m)`; `L = lam·Σ q(lq − lm) + (1−lam)·Σ p(lp − lm)` | `lam` (0.9), `eps`, `temperature` (1.0) |
| `tvd` | `tv_distance` | `0.5·Σ |p − q|` | `temperature` (1.0) |
| `sfkl` | `skewed_forward_kl(alpha)` | `m = α·p + (1−α)·q`; `−Σ p·log(m)` | `alpha` (0.1), `temperature` (1.0) |
| `srkl` | `skewed_reverse_kl(alpha)` | `m = (1−α)·p + α·q`; `Σ q·(lq − log m)` | `alpha` (0.1), `temperature` (1.0) |
| `csd` | `csd` | `Δ = s − t`; `c = Σ q·Δ` (keepdim); `L = Σ_v sg[(Δ − c)·q]·s` (mode "SS") | `temperature` (1.0) |
| `gjsd` | `generalized_jsd(beta, token_clip, clip_mode)` | TRL/OPSD: `beta==0` → `p·(lp − lq)`; `beta==1` → `q·(lq − lp)`; else: mixture. clip elementwise/per_token | `beta` (0.5), `token_clip` (None), `clip_mode` ("elementwise"), `temperature` (1.0) |
| `amid` | `amid(alpha, lam, div_name, div_order, ab_alpha, ab_beta)` | AMID assistant distribution mixture with FKL or AB divergence | `alpha` (0.5), `lam` (0.5), `div_name` ("fkl"), `div_order` ("pr"), `ab_alpha` (0.2), `ab_beta` (0.7), `temperature` (1.0) |
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

TokenFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _inf_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.isinf(x)


def forward_kl(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """DistiLLM ``forward_kl`` (distillm/losses.py:4): cross-entropy form ``-sum_v p_T log q_S``
    (teacher entropy omitted: same gradient). Infinite student logits are masked.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature != 1.0:
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    t_prob = F.softmax(t_logits, dim=-1).masked_fill(inf, 0.0)
    s_logp = F.log_softmax(s_logits, dim=-1).masked_fill(inf, 0.0)
    prod = (t_prob * s_logp).masked_fill(inf, 0.0)
    return -prod.sum(-1)


def reverse_kl(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """``sum_v q_S (log q_S - log p_T)`` (distillm/losses.py:14)."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature != 1.0:
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    s_prob = F.softmax(s_logits, dim=-1).masked_fill(inf, 0.0)
    s_logp = F.log_softmax(s_logits, dim=-1).masked_fill(inf, 0.0)
    t_logp = F.log_softmax(t_logits, dim=-1).masked_fill(inf, 0.0)
    prod = (s_prob * (s_logp - t_logp)).masked_fill(inf, 0.0)
    return prod.sum(-1)


def symmetric_kl(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    lam: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """DistiLLM ``symmetric_kl`` (distillm/losses.py:26): ``(1 - lam) * fkl + lam * rkl``."""
    return (1.0 - lam) * forward_kl(s_logits, t_logits, temperature=temperature) + lam * reverse_kl(
        s_logits, t_logits, temperature=temperature
    )


def js_distance(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    lam: float = 0.9,
    eps: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Jensen-Shannon distance (distillm/losses.py:32).
    m = (1 - lam) * p + lam * q; lm = log(m + eps)
    L = lam * sum q(lq - lm) + (1 - lam) * sum p(lp - lm)
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature != 1.0:
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    p = F.softmax(t_logits, dim=-1).masked_fill(inf, 0.0)
    q = F.softmax(s_logits, dim=-1).masked_fill(inf, 0.0)
    m = ((1.0 - lam) * p + lam * q).masked_fill(inf, 1.0)
    lm = (torch.log(m + eps) if eps > 0.0 else torch.log(m)).masked_fill(inf, 0.0)

    lp = F.log_softmax(t_logits, dim=-1).masked_fill(inf, 0.0)
    lq = F.log_softmax(s_logits, dim=-1).masked_fill(inf, 0.0)

    term_s = (q * (lq - lm)).masked_fill(inf, 0.0)
    term_t = (p * (lp - lm)).masked_fill(inf, 0.0)
    return lam * term_s.sum(-1) + (1.0 - lam) * term_t.sum(-1)


def tv_distance(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Total Variation distance: ``0.5 * sum |p - q|`` (distillm/losses.py:55)."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature != 1.0:
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    p = F.softmax(t_logits, dim=-1).masked_fill(inf, 0.0)
    q = F.softmax(s_logits, dim=-1).masked_fill(inf, 0.0)
    diff = (0.5 * torch.abs(p - q)).masked_fill(inf, 0.0)
    return diff.sum(-1)


def skewed_forward_kl(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    alpha: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Skewed forward KL (distillm/losses.py:66).
    m = alpha * p + (1 - alpha) * q; -sum p * log(m)
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature != 1.0:
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    p = F.softmax(t_logits, dim=-1).masked_fill(inf, 0.0)
    q = F.softmax(s_logits, dim=-1).masked_fill(inf, 0.0)
    m = (alpha * p + (1.0 - alpha) * q).masked_fill(inf, 1.0)
    lm = torch.log(m).masked_fill(inf, 0.0)
    prod = (p * lm).masked_fill(inf, 0.0)
    return -prod.sum(-1)


def skewed_reverse_kl(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    alpha: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Skewed reverse KL (distillm/losses.py:80).
    m = (1 - alpha) * p + alpha * q; sum q * (lq - log(m))
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature != 1.0:
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    p = F.softmax(t_logits, dim=-1).masked_fill(inf, 0.0)
    q = F.softmax(s_logits, dim=-1).masked_fill(inf, 0.0)
    m = ((1.0 - alpha) * p + alpha * q).masked_fill(inf, 1.0)
    lq = F.log_softmax(s_logits, dim=-1).masked_fill(inf, 0.0)
    lm = torch.log(m).masked_fill(inf, 0.0)
    prod = (q * (lq - lm)).masked_fill(inf, 0.0)
    return prod.sum(-1)


def csd(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    mode: str = "SS",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cosine / Surrogate divergence (DistiLLM / CSD distillm/losses.py:97).

    Formula (mode "SS"):
        student_probs = softmax(s_logits / temperature)
        delta = s_logits - t_logits
        loss = (delta - sum(student_probs * delta, keepdim=True)).detach() * student_probs.detach() * s_logits
        return sum_vocab(loss)
    """
    if mode != "SS":
        raise NotImplementedError(f"CSD mode '{mode}' is not implemented (dead code in original)")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    student_probs = F.softmax(s_logits / temperature, dim=-1, dtype=torch.float32)
    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    s_safe = s_logits.masked_fill(inf, 0.0)
    t_safe = t_logits.masked_fill(inf, 0.0)
    delta = s_safe - t_safe
    c = torch.sum(student_probs * delta, dim=-1, keepdim=True)
    loss = (delta - c).detach() * student_probs.detach() * s_safe
    loss = loss.masked_fill(inf, 0.0)
    return loss.sum(dim=-1)


def generalized_jsd(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    beta: float = 0.5,
    token_clip: float | None = None,
    clip_mode: str = "elementwise",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generalized Jensen-Shannon Divergence (TRL / OPSD opsd_trainer.py:382).
    beta=0: KL(T||S) full; beta=1: KL(S||T) full.
    otherwise: mixture distribution with student weight (1 - beta) and teacher weight beta.
    """
    if clip_mode not in ("elementwise", "per_token"):
        raise ValueError(f"unknown clip_mode: {clip_mode}, expected 'elementwise' or 'per_token'")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature != 1.0:
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature

    inf = _inf_mask(s_logits) | _inf_mask(t_logits)
    s_logp = F.log_softmax(s_logits, dim=-1).masked_fill(inf, 0.0)
    t_logp = F.log_softmax(t_logits, dim=-1).masked_fill(inf, 0.0)

    if beta == 0.0:
        jsd = F.kl_div(s_logp, t_logp, reduction="none", log_target=True)
    elif beta == 1.0:
        jsd = F.kl_div(t_logp, s_logp, reduction="none", log_target=True)
    else:
        beta_tensor = torch.tensor(beta, dtype=s_logp.dtype, device=s_logp.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([s_logp + torch.log1p(-beta_tensor), t_logp + torch.log(beta_tensor)]),
            dim=0,
        ).masked_fill(inf, 0.0)
        kl_teacher = F.kl_div(mixture_log_probs, t_logp, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, s_logp, reduction="none", log_target=True)
        jsd = beta * kl_teacher + (1.0 - beta) * kl_student

    if token_clip is not None and clip_mode == "elementwise":
        jsd = jsd.clamp(max=token_clip)

    jsd = jsd.masked_fill(inf, 0.0)
    per_tok = jsd.sum(-1)

    if token_clip is not None and clip_mode == "per_token":
        per_tok = per_tok.clamp(max=token_clip)

    return per_tok


def _amid_assistant_distribution(
    p: torch.Tensor,
    q: torch.Tensor,
    logp: torch.Tensor,
    logq: torch.Tensor,
    alpha: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lam <= 0.0:
        return q, logq
    if lam >= 1.0:
        return p, logp
    if alpha >= 1.0:
        logr_unnorm = lam * logp + (1.0 - lam) * logq
    else:
        t1 = math.log(lam) + 0.5 * (1.0 - alpha) * logp
        t2 = math.log(1.0 - lam) + 0.5 * (1.0 - alpha) * logq
        logr_unnorm = 2.0 / (1.0 - alpha) * torch.logaddexp(t1, t2)
    return F.softmax(logr_unnorm, dim=-1), F.log_softmax(logr_unnorm, dim=-1)


def _amid_divergence_loss(
    div_name: str,
    div_order: str,
    p: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    logp: torch.Tensor,
    logq: torch.Tensor,
    logr: torch.Tensor,
    inf_mask: torch.Tensor,
    ab_alpha: float,
    ab_beta: float,
) -> torch.Tensor:
    if div_name == "fkl":
        pairs = {"pr": (p, logp, logr), "qr": (q, logq, logr), "rp": (r, logr, logp), "rq": (r, logr, logq)}
        if div_order not in pairs:
            raise ValueError(f"Unknown div_order: '{div_order}', expected 'pr', 'qr', 'rp', or 'rq'")
        dist, l1, l2 = pairs[div_order]
        return torch.sum(torch.masked_fill(dist * (l1 - l2), inf_mask, 0.0), dim=-1)

    if div_name == "ab":
        apb = ab_alpha + ab_beta
        if apb == 0.0 or ab_alpha * ab_beta == 0.0:
            raise ValueError(f"Invalid ab_alpha={ab_alpha}, ab_beta={ab_beta}: require non-zero and sum != 0")
        if div_order == "pr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logp + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logp, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
        elif div_order == "qr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logq + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(ab_alpha * logq, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(ab_beta * logr, dim=-1))
        elif div_order == "rp":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logp, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(ab_alpha * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(ab_beta * logp, dim=-1))
        elif div_order == "rq":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logq, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(ab_alpha * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(ab_beta * logq, dim=-1))
        else:
            raise ValueError(f"Unknown div_order: '{div_order}', expected 'pr', 'qr', 'rp', or 'rq'")
        divergence = -(term1 - term2 - term3) / (ab_alpha * ab_beta)
        return torch.where(torch.isfinite(divergence), divergence, torch.zeros_like(divergence))

    raise ValueError(f"Unknown div_name: '{div_name}', expected 'fkl' or 'ab'")


def amid(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    alpha: float = 0.5,
    lam: float = 0.5,
    div_name: str = "fkl",
    div_order: str = "pr",
    ab_alpha: float = 0.2,
    ab_beta: float = 0.7,
    temperature: float = 1.0,
    **kwargs: Any,
) -> torch.Tensor:
    """AMID (Assistant Model / Intermediate Distribution) distillation divergence.

    Computes an assistant distribution r via geometric mixture (alpha >= 1) or
    Rényi / power mean mixture (alpha < 1) between teacher p and student q with weight lam.
    Then evaluates divergence between distributions according to div_name ('fkl' or 'ab')
    and div_order ('pr', 'qr', 'rp', 'rq').

    Returns per-token loss [B, t].
    """
    temperature = float(kwargs.get("temperature", temperature))
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature != 1.0:
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature

    alpha = float(kwargs.get("amid_alpha", alpha))
    lam = float(kwargs.get("amid_lam", lam))
    div_name = str(kwargs.get("amid_div_name", div_name))
    div_order = str(kwargs.get("amid_div_order", div_order))
    ab_alpha = float(kwargs.get("ab_alpha", ab_alpha))
    ab_beta = float(kwargs.get("ab_beta", ab_beta))

    p = F.softmax(t_logits, dim=-1)
    q = F.softmax(s_logits, dim=-1)
    logp = F.log_softmax(t_logits, dim=-1)
    logq = F.log_softmax(s_logits, dim=-1)

    inf_mask = _inf_mask(t_logits) | _inf_mask(s_logits)
    r, logr = _amid_assistant_distribution(p, q, logp, logq, alpha, lam)
    return _amid_divergence_loss(
        div_name, div_order, p, q, r, logp, logq, logr, inf_mask, ab_alpha, ab_beta
    )


DIVERGENCES: dict[str, Callable[..., TokenFn]] = {
    "fkl": lambda temperature=1.0, **kw: partial(forward_kl, temperature=temperature),
    "rkl": lambda temperature=1.0, **kw: partial(reverse_kl, temperature=temperature),
    "symkl": lambda lam=0.9, temperature=1.0, **kw: partial(symmetric_kl, lam=lam, temperature=temperature),
    "jsd": lambda lam=0.9, eps=0.0, temperature=1.0, **kw: partial(
        js_distance,
        lam=lam,
        eps=eps,
        temperature=temperature,
    ),
    "tvd": lambda temperature=1.0, **kw: partial(tv_distance, temperature=temperature),
    "sfkl": lambda alpha=0.1, temperature=1.0, **kw: partial(
        skewed_forward_kl,
        alpha=alpha,
        temperature=temperature,
    ),
    "srkl": lambda alpha=0.1, temperature=1.0, **kw: partial(
        skewed_reverse_kl,
        alpha=alpha,
        temperature=temperature,
    ),
    "csd": lambda mode="SS", temperature=1.0, **kw: partial(csd, mode=mode, temperature=temperature),
    "gjsd": lambda beta=0.5, token_clip=None, clip_mode="elementwise", temperature=1.0, **kw: partial(
        generalized_jsd,
        beta=beta,
        token_clip=token_clip,
        clip_mode=clip_mode,
        temperature=temperature,
    ),
    "amid": lambda alpha=0.5, lam=0.5, div_name="fkl", div_order="pr", ab_alpha=0.2, ab_beta=0.7, temperature=1.0, **kw: partial(
        amid,
        alpha=alpha,
        lam=lam,
        div_name=div_name,
        div_order=div_order,
        ab_alpha=ab_alpha,
        ab_beta=ab_beta,
        temperature=temperature,
        **kw,
    ),
}


def get_divergence(name: str, **params) -> TokenFn:
    if name not in DIVERGENCES:
        raise NotImplementedError(
            f"divergence '{name}' not implemented yet. "
            f"Available: {sorted(DIVERGENCES)}"
        )
    if "temperature" in params and params["temperature"] <= 0.0:
        raise ValueError(f"temperature must be positive, got {params['temperature']}")
    return DIVERGENCES[name](**params)
