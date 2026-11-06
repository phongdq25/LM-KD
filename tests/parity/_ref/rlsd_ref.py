# Verbatim: Self-Distill/RLSD-main/verl/workers/actor/dp_opsd_actor.py:210-269.
from typing import Optional
import torch

class VF:
    @staticmethod
    def masked_mean(values, mask):
        return (values * mask).sum() / mask.sum().clamp_min(1)

def _build_stgca_advantages(
    *,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    lam: float,
    clip_range: float,
    negative_only: bool = False,
    thought_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    sign_adv = torch.sign(advantages)
    delta = (teacher_log_probs.detach() - student_log_probs.detach()) * response_mask
    weights = torch.exp(sign_adv * delta) * response_mask

    low_clip_mask = ((weights < (1.0 - clip_range)).to(response_mask.dtype)) * response_mask
    high_clip_mask = ((weights > (1.0 + clip_range)).to(response_mask.dtype)) * response_mask
    clipped_weights = torch.clamp(weights, min=1.0 - clip_range, max=1.0 + clip_range)
    reweight = ((1.0 - lam) + lam * clipped_weights) * response_mask

    if negative_only:
        seq_negative = (advantages.sum(dim=-1, keepdim=True) < 0).float()
        reweight = seq_negative * reweight + (1.0 - seq_negative) * response_mask

    if thought_mask is not None:
        # Only reweight tokens inside <thought>...</thought> spans.
        # Tokens outside (format tags, answer, etc.) keep reweight=1 (pure GRPO).
        reweight = thought_mask * reweight + (1.0 - thought_mask) * response_mask

    stgca_advantages = advantages * reweight.detach()

    neg_mask = response_mask * (advantages < 0).float()
    neg_token_count = neg_mask.sum().clamp(min=1.0)
    metrics = {
        "grpo_opsd/stgca_delta_mean": VF.masked_mean(delta.detach(), response_mask).item(),
        "grpo_opsd/stgca_weight_mean": VF.masked_mean(weights.detach(), response_mask).item(),
        "grpo_opsd/stgca_clipped_weight_mean": VF.masked_mean(clipped_weights.detach(), response_mask).item(),
        "grpo_opsd/stgca_clip_low_ratio": VF.masked_mean(low_clip_mask.detach(), response_mask).item(),
        "grpo_opsd/stgca_clip_high_ratio": VF.masked_mean(high_clip_mask.detach(), response_mask).item(),
        "grpo_opsd/stgca_adv_mean": VF.masked_mean(stgca_advantages.detach(), response_mask).item(),
        "grpo_opsd/stgca_adv_abs_mean": VF.masked_mean(
            stgca_advantages.detach().abs(), response_mask
        ).item(),
    }
    if negative_only:
        metrics["grpo_opsd/stgca_negative_only"] = 1.0
        metrics["grpo_opsd/stgca_neg_seq_ratio"] = (
            (neg_mask.sum(dim=-1) > 0).float().mean().item()
        )
        metrics["grpo_opsd/stgca_neg_delta_mean"] = (
            (delta.detach() * neg_mask).sum() / neg_token_count
        ).item()
    if thought_mask is not None:
        thought_token_count = thought_mask.sum().item()
        total_token_count = response_mask.sum().item()
        metrics["grpo_opsd/stgca_thought_only"] = 1.0
        metrics["grpo_opsd/stgca_thought_token_ratio"] = (
            thought_token_count / max(total_token_count, 1.0)
        )
    return stgca_advantages, metrics

