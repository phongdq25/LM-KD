"""Group-relative advantages and teacher-guided RLSD token weighting."""
from collections import defaultdict

import torch
from pydantic import Field

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import ADV_TRANSFORMS, ADVANTAGES
from lmkd.data.labels import supervised_mask
from lmkd.losses.functional.rl import rlsd_reweight


def grpo_advantage(rewards: torch.Tensor, uids: list[str], norm_by_std=True, eps=1e-6) -> torch.Tensor:
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


def effective_lambda(lam: float, step: int, warmup_steps=0, decay_steps=0) -> float:
    """Linear warmup followed by linear decay (RLSD opsd_trainer.py:1952-1961)."""
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


class LambdaSchedule(BaseConfig):
    """RLSD warmup and decay durations measured in rollout batches."""
    warmup_steps: int = Field(default=0, ge=0)
    decay_steps: int = Field(default=0, ge=0)


@ADV_TRANSFORMS.register("rlsd_reweight", source="Self-Distill/RLSD-main/verl/workers/actor/dp_opsd_actor.py:210-269")
class RLSDReweight(Plugin):
    """A * ((1-lambda) + lambda*clip(exp(sign(A)*(logT-logS)))) with detached weights."""
    needs_teacher_logp = True

    class Config(BaseConfig):
        lam: float = Field(default=.5, ge=0, le=1)
        clip: float = Field(default=.2, ge=0, le=1)
        negative_only: bool = False
        thought_only: bool = False
        lam_schedule: LambdaSchedule = LambdaSchedule()

    def setup(self, rt):
        if self.cfg.thought_only:
            raise NotImplementedError("RLSD thought_only is optional and not implemented; use thought_only=False")

    def __call__(self, adv, batch, teacher_logp, student_logp, state):
        schedule = self.cfg.lam_schedule
        lam = effective_lambda(self.cfg.lam, state.step, schedule.warmup_steps, schedule.decay_steps)
        result, metrics = rlsd_reweight(adv, teacher_logp, student_logp,
                                       supervised_mask(batch["student.labels"]), lam, self.cfg.clip,
                                       self.cfg.negative_only)
        state.log("rlsd/effective_lambda", lam)
        for key, value in metrics.items():
            state.log(key, value)
        return result
