"""LUFFY: Learning to Reason Under Off-Policy Guidance - Advantage Estimation.

Paper: LUFFY: Learning to Reason Under Off-Policy Guidance (NeurIPS 2025)

Key Method Characteristics:
1. Split-sample GRPO Advantage (`grpo_split`):
   - Computes baseline mean and variance solely across on-policy student rollouts, preventing
     high-performing off-policy demonstrations from distorting on-policy gradient baselines.
   - Centers and standardizes both on-policy and off-policy trajectories with respect to the on-policy baseline.
2. Prefix Advantage Reweighting (`luffy_prefix_weight`):
   - Modulates advantages with separate weights (alpha, beta) for off-policy demonstration tokens
     versus on-policy generated tokens.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from pydantic import Field

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import ADV_TRANSFORMS, ADVANTAGES


def compute_grpo_outcome_advantage_split(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor | None = None,
    index: torch.Tensor | list[Any] | None = None,
    on_policy_mask: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    use_std: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GRPO advantage operating only on Outcome reward with on-policy baseline estimation.

    """
    if token_level_rewards.ndim == 1:
        # Scalar rewards per sequence [B]
        scores = token_level_rewards.clone()
        bsz = scores.shape[0]
        if index is None:
            index = list(range(bsz))
        if on_policy_mask is None:
            on_policy_mask = torch.ones(bsz, dtype=torch.bool, device=scores.device)

        id2score = defaultdict(list)
        with torch.no_grad():
            for i in range(bsz):
                idx_val = index[i].item() if isinstance(index[i], torch.Tensor) else index[i]
                if on_policy_mask[i].item() is True:
                    val = scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i]
                    id2score[idx_val].append(val)

            id2mean = {}
            id2std = {}
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx], dtype=torch.float32))
                    id2std[idx] = torch.std(torch.tensor(id2score[idx], dtype=torch.float32))

            for idx in id2std:
                if id2std[idx].item() == 0:
                    id2std[idx] = torch.tensor(1.0)

            for i in range(bsz):
                idx_val = index[i].item() if isinstance(index[i], torch.Tensor) else index[i]
                mean_val = id2mean.get(idx_val, torch.tensor(0.0, device=scores.device))
                std_val = id2std.get(idx_val, torch.tensor(1.0, device=scores.device))
                if use_std:
                    scores[i] = (scores[i] - mean_val) / (std_val + epsilon)
                else:
                    scores[i] = scores[i] - mean_val

        return scores, scores

    # 2D Token-level rewards [B, T]
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = token_level_rewards != 0
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
    bsz = scores.shape[0]

    if index is None:
        index = list(range(bsz))
    if on_policy_mask is None:
        on_policy_mask = torch.ones(bsz, dtype=torch.bool, device=scores.device)
    if eos_mask is None:
        eos_mask = torch.ones_like(token_level_rewards)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        for i in range(bsz):
            idx_val = index[i].item() if isinstance(index[i], torch.Tensor) else index[i]
            if on_policy_mask[i].item() is True:
                val = scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i]
                id2score[idx_val].append(val)

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx], dtype=torch.float32))
                id2std[idx] = torch.std(torch.tensor(id2score[idx], dtype=torch.float32))

        for idx in id2std:
            if id2std[idx].item() == 0:
                id2std[idx] = torch.tensor(1.0)

        for i in range(bsz):
            idx_val = index[i].item() if isinstance(index[i], torch.Tensor) else index[i]
            mean_val = id2mean.get(idx_val, torch.tensor(0.0, device=scores.device))
            std_val = id2std.get(idx_val, torch.tensor(1.0, device=scores.device))
            if use_std:
                scores[i] = (scores[i] - mean_val) / (std_val + epsilon)
            else:
                scores[i] = scores[i] - mean_val

        scores_2d = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores_2d, scores_2d


@ADVANTAGES.register("grpo_split", force=True)
class GRPOSplit(Plugin):
    """GRPO outcome advantage estimator with on-policy baseline centering.

    Calculates the prompt group mean (and std) using only on-policy samples,
    then evaluates advantages for both on-policy and off-policy responses.
    """

    requires = ("rollout.reward", "group.uid")

    class Config(BaseConfig):
        norm_by_std: bool = False
        eps: float = Field(default=1e-6, gt=0)

    def __call__(self, batch: Any) -> torch.Tensor:
        rewards = batch["rollout.reward"]
        uids = batch.meta.get("group.uid", list(range(rewards.shape[0])))

        if "rollout.prefix_mask" in batch:
            pmask = batch["rollout.prefix_mask"]
            on_policy = ~pmask.any(-1) if pmask.ndim > 1 else ~pmask
        elif "prefix_mask" in batch:
            pmask = batch["prefix_mask"]
            on_policy = ~pmask.any(-1) if pmask.ndim > 1 else ~pmask
        elif "group.on_policy_mask" in batch:
            on_policy = batch["group.on_policy_mask"]
        else:
            on_policy = torch.ones_like(rewards, dtype=torch.bool)

        adv, _ = compute_grpo_outcome_advantage_split(
            token_level_rewards=rewards,
            index=uids,
            on_policy_mask=on_policy,
            epsilon=self.cfg.eps,
            use_std=self.cfg.norm_by_std,
        )
        return adv


@ADV_TRANSFORMS.register(
    "luffy_prefix_weight", force=True
)
class LUFFYPrefixWeight(Plugin):
    """Reweight advantages for off-policy demonstration tokens (alpha) and on-policy tokens (beta)."""

    class Config(BaseConfig):
        alpha: float = 1.0
        beta: float = 1.0

    def __call__(self, adv: torch.Tensor, batch: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        prefix_mask = batch.get("rollout.prefix_mask", batch.get("prefix_mask", None))
        if prefix_mask is None:
            return adv
        alpha_weight = prefix_mask.float() * self.cfg.alpha
        beta_weight = (~prefix_mask.bool()).float() * self.cfg.beta
        prefix_weight = alpha_weight + beta_weight
        return prefix_weight * adv


__all__ = [
    "compute_grpo_outcome_advantage_split",
    "GRPOSplit",
    "LUFFYPrefixWeight",
]
