# Verbatim reference implementation from LUFFY/luffy/verl/verl/mix_src/mix_core_alg.py
from collections import defaultdict
import torch

class VerlF:
    @staticmethod
    def masked_mean(data, mask):
        s = (data * mask).sum()
        c = mask.sum()
        return s / c.clamp_min(1)


def compute_sft_pure_loss(log_prob, eos_mask):
    sft_losses = -log_prob
    sft_loss = VerlF.masked_mean(sft_losses, eos_mask)
    return sft_loss


def compute_grpo_outcome_advantage_split(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor | list,
    on_policy_mask: torch.Tensor,
    epsilon: float = 1e-6,
    use_std: bool = True,
):
    """Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = token_level_rewards != 0
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # only include on-policy samples for mean and std calculation
            idx_val = index[i].item() if isinstance(index[i], torch.Tensor) else index[i]
            if on_policy_mask[i].item() is True:
                id2score[idx_val].append(scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx], dtype=torch.float32))
                id2std[idx] = torch.std(torch.tensor(id2score[idx], dtype=torch.float32))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # process std
        for idx in id2std:
            if id2std[idx].item() == 0:
                id2std[idx] = torch.tensor(1.0)
        for i in range(bsz):
            idx_val = index[i].item() if isinstance(index[i], torch.Tensor) else index[i]
            mean_val = id2mean.get(idx_val, torch.tensor(0.0))
            std_val = id2std.get(idx_val, torch.tensor(1.0))
            if use_std:
                scores[i] = (scores[i] - mean_val) / (std_val + epsilon)
            else:
                scores[i] = scores[i] - mean_val
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_token_on_off_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange,
    clip_upper_bound,
    prefix_mask,
    off_cliprange,
    off_normalize=False,
    off_abs_cliprange=None,
    off_max_clip=None,
    off_min_clip=None,
    all_max_clip=None,
    off_policy_reshape="no_reshape",
    off_policy_reshape_weight=1.0,
    off_policy_reshape_pow_exp=0.5,
    on_policy_reshape="no_reshape",
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):
    """Adapted reference implementation of LUFFY core algorithm."""
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = VerlF.masked_mean(-negative_approx_kl, eos_mask)

    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl)
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.exp(negative_approx_kl)
        ratio = torch.sqrt(ratio)
    elif on_policy_reshape == "pow":
        ratio = torch.exp(negative_approx_kl)
        ratio = torch.pow(ratio, on_policy_reshape_pow_exp)
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.5)
        f_old_prob = old_prob / (old_prob + 0.5)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    on_pg_losses = -advantages * ratio
    upper_bound = max(clip_upper_bound, 1.0 + cliprange)

    if loss_remove_clip is False:
        on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
        on_pg_clipfrac = VerlF.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = VerlF.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
    else:
        on_pg_loss = VerlF.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0, device=log_prob.device)

    # compute off-policy loss
    if target_probs is None:
        off_ratio = torch.exp(log_prob)
        if off_policy_reshape == "no_reshape":
            pass
        elif off_policy_reshape == "logp":
            off_ratio = log_prob * off_policy_reshape_weight
        elif off_policy_reshape == "p_logp":
            off_ratio = log_prob * off_policy_reshape_weight + off_ratio
        elif off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif off_policy_reshape == "p_div_p_0.1":
            off_ratio = off_ratio / (off_ratio + 0.1)
        elif off_policy_reshape == "p_div_p_0.5":
            off_ratio = off_ratio / (off_ratio + 0.5)
        elif off_policy_reshape == "p_div_p_0.3":
            off_ratio = off_ratio / (off_ratio + 0.3)
        elif off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
        else:
            raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
    else:
        assert target_probs.shape == log_prob.shape
        off_ratio = torch.exp(log_prob) / (target_probs + 1e-6)
        off_ratio = off_ratio * prefix_mask

    # clip off-policy ratio
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = VerlF.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0, device=log_prob.device)

    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = VerlF.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0, device=log_prob.device)

    off_ratio_mean = VerlF.masked_mean(off_ratio, prefix_mask * eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0, device=log_prob.device)

    off_pg_losses = -advantages * off_ratio
    off_pg_loss = VerlF.masked_mean(off_pg_losses, prefix_mask * eos_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0, device=log_prob.device)
    off_pg_clipfrac = torch.tensor(0.0, device=log_prob.device)

    prefix_mask_f = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask_f + on_pg_losses * (1 - prefix_mask_f)

    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = VerlF.masked_mean(off_policy_probs, prefix_mask * eos_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0, device=log_prob.device)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = VerlF.masked_mean(on_policy_probs, (1.0 - prefix_mask_f) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0, device=log_prob.device)

    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask

    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
    else:
        pg_loss = VerlF.masked_mean(pg_losses, eos_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }

