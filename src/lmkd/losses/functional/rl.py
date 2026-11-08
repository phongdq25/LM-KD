"""Pure PPO/GRPO token objectives (verl base)."""

from __future__ import annotations

import torch


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Finite masked mean, including batches with no supervised tokens."""
    return (value * mask).sum() / mask.sum().clamp_min(1)


def ppo_clip_per_token(logp, old_logp, adv, clip_low, clip_high, clip_c=None):
    """Clipped PPO numerator per token (verl actor)."""
    ratio = (logp - old_logp.detach()).clamp(-20, 20).exp()
    pg1 = -adv.detach() * ratio
    pg2 = -adv.detach() * ratio.clamp(1 - clip_low, 1 + clip_high)
    loss = torch.maximum(pg1, pg2)
    if clip_c is not None:
        loss = torch.where(adv < 0, torch.minimum(loss, -adv.detach() * clip_c), loss)
    return loss, (pg2 > pg1).float(), old_logp.detach() - logp


def grpo_loss(
    logits: torch.Tensor,
    old_logp: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
) -> torch.Tensor:
    """Group Relative Policy Optimization (GRPO) clipped surrogate loss.

    Computes token-level log probabilities from target-aligned student logits and applies
    clipped policy gradient against advantages, averaged over supervised tokens.
    """
    logp = logits.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = labels.ne(-100)
    per_token_loss, _, _ = ppo_clip_per_token(logp, old_logp, advantages, clip_low, clip_high)
    return (per_token_loss * mask).sum() / mask.sum().clamp_min(1)


__all__ = [
    "masked_mean",
    "ppo_clip_per_token",
    "grpo_loss",
]
