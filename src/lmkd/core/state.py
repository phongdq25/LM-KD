"""Training state shared (read-only) with plugins through contexts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrainState:
    """Counters of the running job (step, micro step, epoch, tokens, ``progress`` in [0,1]) plus free-form
    ``extra``; checkpointed.
    """
    step: int = 0                 # optimizer steps done
    micro_step: int = 0           # forward/backward passes done
    epoch: float = 0.0            # fractional epoch of the data producer
    total_steps: int = 1
    tokens_seen: int = 0
    samples_seen: int = 0
    done: bool = False
    extra: dict = field(default_factory=dict)  # free-form, e.g. last eval metrics

    @property
    def progress(self) -> float:
        """Fraction of training done in [0, 1] (use it for schedules)."""
        return min(1.0, self.step / max(1, self.total_steps))

    def log(self, key: str, value: Any) -> None:
        """Record an auxiliary metric for the current step (producers, hooks, callbacks...).
        ConsoleLogger merges ``extra["metrics"]`` with the step metrics and clears it (K4)."""
        m = self.extra.setdefault("metrics", {})
        m[key] = float(value.detach().float().mean()) if hasattr(value, "detach") else float(value)

    def pop_metrics(self) -> dict:
        return self.extra.pop("metrics", {}) or {}

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)
