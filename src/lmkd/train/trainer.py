"""The ONE training loop (DESIGN §7.1). Recipe baselines extend it through plugins; scripts may
write their own loop from the same building blocks (``rt.begin / next_batch / train_step / end``,
see ``lmkd.api``), so both paths share every step semantic."""
from __future__ import annotations

from lmkd.core.config import RecipeConfig
from lmkd.train.builder import build_runtime
from lmkd.train.runtime import Runtime


class Trainer:
    """Thin wrapper over ``Runtime.fit``: ``begin``, ``train_step`` until done, ``end``."""
    def __init__(self, rt: Runtime):
        self.rt = rt

    @classmethod
    def from_config(cls, cfg: RecipeConfig) -> Trainer:
        return cls(build_runtime(cfg))

    def fit(self) -> Runtime:
        return self.rt.fit()


def train(cfg: RecipeConfig) -> Runtime:
    return Trainer.from_config(cfg).fit()
