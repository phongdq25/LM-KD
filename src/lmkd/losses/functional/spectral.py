"""Closed-form causal LM gradient and spectral primitives from SGL."""
from __future__ import annotations

import torch


def analytic_gradients_and_entropies(
    hidden: torch.Tensor, targets: torch.Tensor, unembedding: torch.Tensor, chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Wᵀ(softmax(Wh)-onehot(y)) and predictive entropy in token chunks.

    Source: SGL/src/sgl/signals/gradients.py:44-74.
    """
    if chunk < 1:
        raise ValueError("chunk must be positive")
    weight = unembedding.float()
    gradients = torch.empty((len(targets), weight.shape[1]), device=hidden.device, dtype=torch.float32)
    entropies = torch.empty(len(targets), device=hidden.device, dtype=torch.float32)
    for start in range(0, len(targets), chunk):
        stop = min(start + chunk, len(targets))
        logits = hidden[start:stop].float() @ weight.T
        probs = torch.softmax(logits, dim=-1)
        entropies[start:stop] = torch.logsumexp(logits, dim=-1) - (probs * logits).sum(-1)
        probs[torch.arange(stop - start, device=hidden.device), targets[start:stop]] -= 1
        gradients[start:stop] = probs @ weight
    return gradients, entropies


def effective_rank(singular_values: torch.Tensor, energy_cutoff: float = 0.95) -> int:
    """Return the smallest rank containing the requested squared singular-value energy.

    Source: SGL/src/sgl/signals/spectral.py:19-22.
    """
    if not 0 < energy_cutoff <= 1:
        raise ValueError("energy_cutoff must be in (0, 1]")
    if singular_values.numel() == 0:
        return 0
    energy = torch.cumsum(singular_values.float().square(), 0)
    total = energy[-1].clamp_min(1e-12)
    return min(len(energy), int(torch.searchsorted(energy / total, energy_cutoff).item()) + 1)


def step_strengths(values: torch.Tensor, spans: list[tuple[int, int]]) -> list[float]:
    """Average token values within each response-relative step span.

    Source: SGL/src/sgl/signals/spectral.py:30-44.
    """
    return [float(values[a:b].float().mean()) if b > a else 0.0 for a, b in spans]
