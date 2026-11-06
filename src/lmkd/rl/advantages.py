"""Group-relative advantages (verl base)."""

from collections import defaultdict

import torch
from pydantic import Field

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import ADVANTAGES


def grpo_advantage(rewards: torch.Tensor, uids: list[str], norm_by_std: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """Center rewards within uid; singleton groups use mean=0, std=1 (verl convention)."""
    if rewards.ndim != 1 or len(uids) != rewards.numel():
        raise ValueError("GRPO needs one uid per scalar reward")
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    result = torch.empty_like(rewards)
    for indices in groups.values():
        values = rewards[indices]
        if len(indices) == 1:
            result[indices] = values
        else:
            centered = values - values.mean()
            result[indices] = centered / (values.std() + eps) if norm_by_std else centered
    return result


def effective_lambda(lam: float, step: int, warmup_steps: int = 0, decay_steps: int = 0) -> float:
    """Linear warmup followed by linear decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return lam * step / warmup_steps
    if decay_steps > 0 and step >= warmup_steps:
        return lam * max(1 - (step - warmup_steps) / decay_steps, 0)
    return lam


@ADVANTAGES.register("grpo", source="Self-Distill/RLSD-main/verl/trainer/core_algos.py:176-216")
class GRPO(Plugin):
    """Group-relative centered rewards, optionally divided by unbiased standard deviation."""

    requires = ("rollout.reward", "group.uid")

    class Config(BaseConfig):
        norm_by_std: bool = True
        eps: float = Field(default=1e-6, gt=0)

    def __call__(self, batch):
        return grpo_advantage(batch["rollout.reward"], batch.meta["group.uid"], self.cfg.norm_by_std, self.cfg.eps)


__all__ = [
    "grpo_advantage",
    "effective_lambda",
    "GRPO",
]
