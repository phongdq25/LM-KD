"""Run a child training recipe as a content-addressed pipeline stage."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

from lmkd.core.config import load_recipe
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import STAGES
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.stage import Stage, StageContext


@STAGES.register("train_recipe", source="pipeline/stages/train.py")
class TrainRecipe(Stage):
    """Execute a child `lmkd train`, cache its recipe hash, and expose the final checkpoint."""

    outputs: ClassVar[dict[str, str]] = {"ckpt": "checkpoint"}

    class Config(BaseConfig):
        """Recipe path, CLI overrides and subprocess selection."""
        recipe: str
        overrides: list[str] = []
        subprocess: bool = True

    def fingerprint(self) -> dict:
        recipe = Path(self.cfg.recipe)
        return {**self.cfg.model_dump(), "recipe_hash": hashlib.sha256(recipe.read_bytes()).hexdigest()}

    def run(self, ctx: StageContext) -> dict:
        if ctx.output is None:
            raise ValueError("train_recipe needs an output artifact")
        cfg = load_recipe(self.cfg.recipe, self.cfg.overrides)
        if self.cfg.subprocess:
            command = [sys.executable, "-m", "lmkd.cli", "train", "-c", self.cfg.recipe]
            for override in self.cfg.overrides:
                command += ["--set", override]
            subprocess.run(command, check=True)
        else:
            from lmkd.train.trainer import train

            train(cfg)
        output_dir = Path(cfg.train.output_dir).resolve()
        checkpoint = output_dir / "checkpoint-final"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"child recipe did not save {checkpoint}")
        ctx.output.path.mkdir(parents=True, exist_ok=True)
        (ctx.output.path / "ckpt").symlink_to(checkpoint, target_is_directory=True)
        (ctx.output.path / "child.json").write_text(json.dumps({"output_dir": str(output_dir)}))
        return {"output_dir": str(output_dir), "checkpoint": str(checkpoint)}

    def attach(self, art: Artifact, rt: Any) -> None:
        path = json.loads((art.path / "child.json").read_text())["output_dir"]
        rt.extra[f"train_recipe:{art.node}"] = path
