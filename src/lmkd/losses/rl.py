"""Native PPO and SDPO criteria with global signal-token normalization."""
from pydantic import Field

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossOutput, LossTerm
from lmkd.losses.functional.rl import add_tail, masked_mean, ppo_clip_per_token, sdpo_per_token
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import SignalKind, TeacherSignalSpec


@CRITERIONS.register("ppo_clip", source="Self-Distill/RLSD-main/verl/workers/actor/dp_opsd_actor.py:426-579")
class PPOClip(Criterion):
    """Clipped PG divided by global token count; matches TRL bnpo for one micro-batch."""
    requires = ("student.labels", "rollout.advantage")
    supports_packing = False

    class Config(BaseConfig):
        clip_low: float = Field(default=.2, ge=0, lt=1)
        clip_high: float = Field(default=.28, ge=0)
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
        loss, clip, kl = ppo_clip_per_token(logp, old, ctx.batch["rollout.advantage"],
                                           self.cfg.clip_low, self.cfg.clip_high, self.cfg.clip_c)
        mask = ctx.supervised_mask()
        loss = loss * ctx.batch.get("rollout.is_weight", 1.)
        return LossOutput({"pg": LossTerm((loss * mask).sum(), "token_sum")},
                          {"clipfrac": masked_mean(clip, mask).detach(), "approx_kl": masked_mean(kl, mask).detach()})


@CRITERIONS.register("sdpo_distill", source="Self-Distill/SDPO-main/verl/trainer/ppo/core_algos.py:1085-1190")
class SDPODistill(Criterion):
    """SDPO on student top-k support with tail and importance correction.

    Numerator / global signal-token count; sdpo_reprompt supplies seq.loss_mask.
    Unlike verl's micro/mini-batch means, micro-batching does not change normalization.
    """
    requires = ("student.labels", "rollout.has_signal")
    needs_teacher = True
    supports_packing = False

    class Config(BaseConfig):
        alpha: float = Field(default=.5, ge=0, le=1)
        k: int | None = Field(default=100, ge=0)
        add_tail: bool = True
        is_clip: float | None = Field(default=2., gt=0)
        full_logit: bool = True
        temperature: float = 1.0

    @classmethod
    def required_keys(cls, cfg):
        return cls.requires + (("rollout.old_logp",) if cfg.is_clip is not None else ())

    def features(self):
        if not self.cfg.full_logit and self.cfg.alpha != 1:
            raise ValueError("sampled SDPO requires alpha=1")
        kind: SignalKind = "topk" if self.cfg.k else "full"
        if not self.cfg.full_logit:
            kind = "sampled_logp"
        return FeatureRequest(student=SideRequest(logits="view"), teacher=SideRequest(logits="view"),
                              teacher_signal=TeacherSignalSpec(kind=kind, k=self.cfg.k or 0,
                                                               support="student", tail="none"))

    def _distribution_loss(self, ctx):
        student, teacher = ctx.student_logits(), ctx.teacher
        if self.cfg.temperature != 1.0:
            student = student.with_temperature(self.cfg.temperature)
        if self.cfg.k:
            ids, s = student.topk(self.cfg.k)
            # Richer merged requests may omit topk; gather from the remapped view then.
            t = teacher.topk_logp
            if t is None or teacher.topk_ids is None or not teacher.topk_ids.equal(ids):
                t = teacher.logits.gather(ids)
            if self.cfg.add_tail:
                s, t = add_tail(s), add_tail(t)
            else:
                s, t = s - s.logsumexp(-1, keepdim=True), t - t.logsumexp(-1, keepdim=True)
        else:
            # Explicit full-vocabulary SDPO needs both distributions instead of sparse support.
            s, t = student.full().log_softmax(-1), teacher.logits.full().log_softmax(-1)
        return sdpo_per_token(s, t, self.cfg.alpha)

    def forward(self, ctx):
        labels = ctx.batch["student.labels"]
        s_view = ctx.student_logits()
        if self.cfg.temperature != 1.0:
            s_view = s_view.with_temperature(self.cfg.temperature)
        s_tok = s_view.target_logp(labels)
        if self.cfg.full_logit:
            per_token = self._distribution_loss(ctx)
        else:
            t_tok = ctx.teacher.token_logp
            if t_tok is None:
                t_tok = ctx.teacher.logits.target_logp(labels)
            per_token = (s_tok - t_tok).detach() * s_tok
        if self.cfg.is_clip is not None:
            old = ctx.batch.get("rollout.old_logp")
            if old is None:
                raise ValueError("SDPO importance correction requires rollout.old_logp")
            ratio = (s_tok.detach() - old.detach()).clamp(-20, 20).exp().clamp(max=self.cfg.is_clip)
            per_token = per_token * ratio
        per_token = per_token * ctx.batch.get("rollout.is_weight", 1.)
        mask = ctx.supervised_mask() * ctx.batch["rollout.has_signal"][:, None]
        return LossOutput({"distill": LossTerm((per_token * mask).sum(), "token_sum")})
