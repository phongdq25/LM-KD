"""Native PPO criteria with global signal-token normalization (verl base)."""

from pydantic import Field

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossOutput, LossTerm
from lmkd.losses.functional.rl import masked_mean, ppo_clip_per_token
from lmkd.models.forward import FeatureRequest, SideRequest


@CRITERIONS.register("ppo_clip")
class PPOClip(Criterion):
    """Clipped PG divided by global token count; matches TRL bnpo for one micro-batch."""

    requires = ("student.labels", "rollout.advantage")
    supports_packing = False

    class Config(BaseConfig):
        clip_low: float = Field(default=0.2, ge=0, lt=1)
        clip_high: float = Field(default=0.28, ge=0)
        clip_c: float | None = Field(default=None, gt=1)
        temperature: float = 1.0

    def features(self):
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx):
        s_view = ctx.student_logits()
        if self.cfg.temperature != 1.0:
            s_view = s_view.with_temperature(self.cfg.temperature)
        logp = s_view.target_logp(ctx.batch["student.labels"])
        old = ctx.batch.get("rollout.old_logp", logp.detach())
        loss, clip, kl = ppo_clip_per_token(
            logp, old, ctx.batch["rollout.advantage"], self.cfg.clip_low, self.cfg.clip_high, self.cfg.clip_c
        )
        mask = ctx.supervised_mask()
        loss = loss * ctx.batch.get("rollout.is_weight", 1.0)
        return LossOutput(
            {"pg": LossTerm((loss * mask).sum(), "token_sum")},
            {"clipfrac": masked_mean(clip, mask).detach(), "approx_kl": masked_mean(kl, mask).detach()},
        )

