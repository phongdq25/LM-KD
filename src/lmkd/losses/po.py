"""Native preference optimization objectives missing from the initial DPO plugin."""
from typing import Literal

import torch
from pydantic import Field

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossOutput, LossTerm
from lmkd.losses.functional.po import preference_loss
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


@CRITERIONS.register("po", source="Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/preference/losses.py")
class PO(Criterion):
    """BPO, TDPO, beta/epsilon-DPO, Ra-DPO, ADPO and TBPO-A pair risks.

    Frozen reference probabilities; adaptive beta is saved through criterion state.
    Source: human_alignment/mechanisms/training/preference/losses.py.
    """
    needs_teacher = True
    requires = ("group.size", "group.valid")

    class Config(BaseConfig):
        method: Literal["bpo", "tdpo", "beta_dpo", "epsilon_dpo", "ra_dpo", "adpo", "tbpo_a"] = "bpo"
        beta: float = Field(default=0.1, gt=0)
        alpha: float = Field(default=0.5, ge=0)
        tdpo2: bool = True
        generator: Literal["sba", "ba", "lsif", "kliep", "logistic"] = "sba"
        bregman_lambda: float = Field(default=0.2, gt=0)
        bregman_scale: float = Field(default=4.0, gt=0)
        log_ratio_clip: float = Field(default=30.0, gt=0)
        label_smoothing: float = Field(default=0.0, ge=0, le=0.5)
        beta_dpo_a: float = 0.6
        beta_dpo_filter_ratio: float = Field(default=0.2, ge=0, lt=1)
        beta_dpo_momentum: float = Field(default=0.9, ge=0, lt=1)
        epsilon: float = Field(default=0.01, gt=0, lt=1)
        ra_dpo_variant: Literal[1, 2] = 1
        ra_dpo_alpha: float = Field(default=0.5, ge=0)
        ra_dpo_confidence: float = Field(default=0.95, gt=0, lt=1)
        ra_dpo_split_vocabulary: bool = True
        adpo_granularity: Literal["adaptive", "static", "pass"] = "adaptive"
        adpo_segments: int = Field(default=256, ge=1)
        adpo_static_k: int = Field(default=32, ge=1)
        adpo_segment_reduce: Literal["sum", "mean"] = "sum"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.adaptive = {}
        self.pending_steps = []

    def prepare(self, ctx):
        if self.cfg.method in {"beta_dpo", "epsilon_dpo"} and ctx.rt.engine.world_size != 1:
            raise ValueError("Adaptive PO statistics currently require local single-process training")
        if self.cfg.method in {"beta_dpo", "epsilon_dpo"}:
            step = ctx.rt.config.step
            name = step if isinstance(step, str) else step["type"]
            scaler = getattr(ctx.rt.engine, "scaler", None)
            if name != "standard" or (scaler is not None and scaler.is_enabled()):
                raise ValueError("Adaptive PO requires standard steps without fp16 loss scaling")

    def features(self):
        return FeatureRequest(student=SideRequest(logits="view"), teacher=SideRequest(logits="view"),
                              teacher_signal=TeacherSignalSpec(kind="full"))

    def forward(self, ctx):
        if int(ctx.batch["group.size"][0]) != 2 or not ctx.batch["group.valid"].all():
            raise ValueError("PO requires exactly one chosen/rejected pair per group")
        # Full vocabulary is required by KL, CVaR and perturbed-logit normalization;
        # these parity paths intentionally trade memory for faithful upstream objectives.
        s, t = ctx.student_logits().full(), ctx.teacher.logits.full()
        order = torch.cat([torch.arange(0, s.shape[0], 2), torch.arange(1, s.shape[0], 2)]).to(s.device)
        loss = preference_loss(self.cfg.method, s[order], t[order], ctx.batch["student.labels"][order],
                               self.cfg, state=self.adaptive)
        if self.cfg.method == "epsilon_dpo":
            self.pending_steps.extend(self.adaptive.pop("last_steps").detach().cpu().tolist())
        return LossOutput({self.cfg.method: LossTerm(loss.sum() * 2, "seq_sum")},
                          metrics={"loss": loss.mean().detach()})

    def on_step_end(self, state):
        if self.pending_steps:
            mean = sum(self.pending_steps) / len(self.pending_steps)
            self.adaptive["beta"] = self.adaptive.get("beta", self.cfg.beta) / (1 + self.cfg.epsilon * mean)
            self.pending_steps.clear()

    def extra_state(self):
        return {"adaptive": self.adaptive, "pending_steps": self.pending_steps}

    def load_extra_state(self, state):
        self.adaptive = state.get("adaptive", {})
        self.pending_steps = state.get("pending_steps", [])
