"""Pure RLSD/PPO/SDPO token objectives; callers own masking and global normalization."""
import math

import torch
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Finite masked mean, including batches with no supervised tokens."""
    return (value * mask).sum() / mask.sum().clamp_min(1)


def ppo_clip_per_token(logp, old_logp, adv, clip_low, clip_high, clip_c=None):
    """Clipped PPO numerator per token (RLSD dp_opsd_actor.py:426-579)."""
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


def rlsd_reweight(adv, teacher_logp, student_logp, mask, lam, clip, negative_only=False, thought_mask=None):
    """RLSD sign-dependent detached token weighting (dp_opsd_actor.py:210-269)."""
    delta = (teacher_logp.detach() - student_logp.detach()) * mask
    weights = (adv.sign() * delta).exp() * mask
    clipped = weights.clamp(1 - clip, 1 + clip)
    reweight = ((1 - lam) + lam * clipped) * mask
    if negative_only:
        neg = (adv.sum(-1, keepdim=True) < 0).float()
        reweight = neg * reweight + (1 - neg) * mask
    if thought_mask is not None:
        reweight = thought_mask * reweight + (1 - thought_mask) * mask
    result = adv * reweight.detach()
    metrics = {"rlsd/delta_mean": masked_mean(delta, mask),
               "rlsd/weight_mean": masked_mean(weights.detach(), mask),
               "rlsd/clipped_weight_mean": masked_mean(clipped.detach(), mask),
               "rlsd/clip_low_ratio": masked_mean((weights < 1 - clip).float(), mask),
               "rlsd/clip_high_ratio": masked_mean((weights > 1 + clip).float(), mask)}
    return result, metrics


def add_tail(logp: torch.Tensor) -> torch.Tensor:
    """Append SDPO's residual probability bucket (core_algos.py:1112-1118)."""
    log_mass = logp.logsumexp(-1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([logp, (-log_mass.expm1()).log()], -1)


def sdpo_per_token(student: torch.Tensor, teacher: torch.Tensor, alpha: float) -> torch.Tensor:
    """KL(T||S), generalized JSD, or KL(S||T) at alpha=0, (0,1), 1 (SDPO)."""
    teacher = teacher.detach()
    if alpha == 0:
        return F.kl_div(student, teacher, reduction="none", log_target=True).sum(-1)
    if alpha == 1:
        return F.kl_div(teacher, student, reduction="none", log_target=True).sum(-1)
    mixture = torch.logsumexp(torch.stack([student + math.log(1 - alpha), teacher + math.log(alpha)]), 0)
    ks = F.kl_div(mixture, student, reduction="none", log_target=True)
    kt = F.kl_div(mixture, teacher, reduction="none", log_target=True)
    return torch.lerp(ks, kt, alpha).sum(-1)
