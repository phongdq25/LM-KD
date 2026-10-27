"""Evaluator & grader interfaces (DESIGN L5). Implementations: task T07 (generation, math),
T23 (embedding).

Convention (library mode): the metric itself is a PURE FUNCTION
``metric(model, tokenizer, records, **options) -> dict[str, float]`` (or on predictions only,
e.g. ``rouge_scores(predictions, references)``), usable without a Runtime from any script. The
Evaluator plugin is a thin wrapper: read records from its Config, call the function on
``rt.student_model`` / ``rt.student_tokenizer``, prefix the keys with ``<kind>/<name>/``.
Example: :mod:`lmkd.eval.embedding`. Only metrics that are defined by the training objective
(``dev_loss``) need the Runtime. ``rt.evaluate()`` runs every evaluator (plugins or callables
``fn(rt) -> dict``) and stores the result in ``rt.state.extra["eval"]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import GRADERS


class Evaluator(Plugin):
    """Evaluation plugin: ``evaluate(rt) -> {metric: value}``. A thin wrapper over a pure metric function (see
    module docstring).
    """

    name_prefix: str = ""

    def setup(self, rt: Any) -> None:
        self.rt = rt

    def evaluate(self, rt: Any) -> dict[str, float]:
        raise NotImplementedError

    def output_dir(self, rt: Any) -> Path:
        """Directory to store raw evaluation outputs: <train.output_dir>/eval/<name>/step-<rt.state.step>."""
        name = getattr(self.cfg, "name", None) or getattr(self, "alias", None) or self.name_prefix or self.__class__.__name__
        step = getattr(rt.state, "step", 0) if hasattr(rt, "state") else 0
        train_cfg = getattr(rt.config, "train", None) if hasattr(rt, "config") else None
        out_root = Path(getattr(train_cfg, "output_dir", ".")) if train_cfg else Path(".")
        p = out_root / "eval" / str(name) / f"step-{step}"
        p.mkdir(parents=True, exist_ok=True)
        return p


class Grader(Plugin):
    """Scores one prediction against a reference (0/1 or a float); used by generation evaluators."""

    def __call__(self, prediction: str, reference: Any) -> float:
        raise NotImplementedError

    def grade_many(self, predictions: list[str], reference: Any) -> list[int]:
        """Score multiple predictions against a single reference."""
        return [int(round(self(p, reference))) for p in predictions]


@GRADERS.register("exact_match")
class ExactMatch(Grader):
    """1.0 if prediction equals reference (optionally stripped)."""

    class Config(BaseConfig):
        strip: bool = True

    def __call__(self, prediction, reference):
        p, r = str(prediction), str(reference)
        if self.cfg.strip:
            p, r = p.strip(), r.strip()
        return float(p == r)
