"""Stage base class (DESIGN §10)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from lmkd.core.plugin import Plugin
from lmkd.pipeline.artifacts import Artifact


@dataclass
class Resources:
    """Resources a stage needs (GPUs, vLLM, separate interpreter)."""
    gpus: int = 0
    uses_vllm: bool = False
    python_env: str | None = None   # run in another interpreter (e.g. SSFT train vs eval envs)


@dataclass
class StageContext:
    """Arguments of ``Stage.run``: runtime, input artifacts, output location, shard, force flag."""
    rt: Any                                   # partial Runtime (models may be absent)
    inputs: dict[str, Artifact] = field(default_factory=dict)
    output: Artifact | None = None
    shard: tuple[int, int] = (0, 1)           # (index, count)
    node: str = ""
    force: bool = False


class Stage(Plugin):
    """Offline pipeline step with content-hashed, resumable, shardable output (axis A). Override ``run``;
    ``load_samples`` if it yields the training set.
    """
    inputs: ClassVar[dict[str, str]] = {}      # name -> artifact kind ("dataset" | "signals" | "checkpoint" | "file")
    outputs: ClassVar[dict[str, str]] = {}
    resources: ClassVar[Resources] = Resources()
    code_version: ClassVar[str] = "1"          # bump when output semantics change
    supports_shards: ClassVar[bool] = False    # True if run() respects ctx.shard

    def fingerprint(self) -> dict:
        return self.cfg.model_dump()

    def run(self, ctx: StageContext) -> dict:
        """Write outputs under ``ctx.output.path``; return info for the manifest."""
        raise NotImplementedError

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        """Merge shard outputs into ``ctx.output.path`` (only when supports_shards=True)."""
        raise NotImplementedError

    def load_samples(self, art: Artifact) -> list[dict]:
        """For stages producing the training dataset: read the samples back."""
        raise NotImplementedError

    def attach(self, art: Artifact, rt: Any) -> None:
        """Called after the stage is done OR found cached: expose outputs to the runtime
        (e.g. ``rt.data_ctx.vocab_map = ...``, calibration scores in ``rt.extra``) (K10)."""
        return None
