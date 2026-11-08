"""Test-only plugins shared by test_deepspeed.py and its torchrun worker."""

from __future__ import annotations

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.modules import LinearProjector


class _ProjFeature(Criterion):
    """sum_t ||P h_t - 1||^2 on the last student hidden state (exercises an aux projector)."""

    class Config(BaseConfig):
        pass

    def features(self):
        return FeatureRequest(student=SideRequest(hidden=(-1,)))

    def build_modules(self, info, modules):
        modules.add(
            f"{self.alias}__p",
            LinearProjector(LinearProjector.Config(in_dim=info.student_hidden, out_dim=4)),
            owner=self.alias,
        )

    def forward(self, ctx):
        h = ctx.modules.get(f"{self.alias}__p")(ctx.student.hidden(-1))
        m = ctx.batch["student.attention_mask"].unsqueeze(-1).to(h.dtype)
        return LossOutput({"sq": LossTerm(((h.float() - 1.0) ** 2 * m).sum(), "token_sum")})


if "_t05_proj" not in CRITERIONS:
    CRITERIONS.register("_t05_proj")(_ProjFeature)
