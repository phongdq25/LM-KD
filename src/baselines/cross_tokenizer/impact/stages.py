"""Offline IMPACT BI calibration stage."""

from __future__ import annotations

import itertools
from typing import Any

import torch

from baselines.cross_tokenizer.impact.layer_mappers import MapperInputs, compute_bi_scores
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import SOURCES, STAGES
from lmkd.core.state import TrainState
from lmkd.models.loading import load_model
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.stage import Stage, StageContext
from lmkd.pipeline.stages.xtok import tokenizer_fingerprint


@STAGES.register(
    "impact_calibrate"
)
class ImpactCalibrate(Stage):
    """Offline mean teacher block-influence scores on raw training prompts."""

    code_version = "1"

    class Config(BaseConfig):
        method: str = "bi"
        samples: int = 256
        max_length: int = 256
        batch_size: int = 1
        prompt_field: str = "prompt"

    def setup(self, rt: Any) -> None:
        self._rt = rt

    def fingerprint(self) -> dict:
        rt = self._rt
        source = SOURCES.build(rt.config.data.train).fingerprint() if rt.config.data.train is not None else None
        return {
            **self.cfg.model_dump(),
            "source": source,
            "teacher_model": rt.config.models.teacher.model_dump() if rt.config.models.teacher else None,
            "teacher_tokenizer": tokenizer_fingerprint(rt.teacher_tokenizer),
        }

    def run(self, ctx: StageContext) -> dict:
        if self.cfg.method != "bi":
            raise ValueError("impact_calibrate currently supports method='bi' only")
        rt = ctx.rt
        if rt.config.data.train is None:
            raise ValueError("impact_calibrate needs data.train source")
        if rt.teacher_tokenizer is None or rt.config.models.teacher is None:
            raise ValueError("impact_calibrate needs a teacher model and tokenizer")
        records = itertools.islice(SOURCES.build(rt.config.data.train).records(), self.cfg.samples)
        prompts = [str(row[self.cfg.prompt_field]) for row in records]
        if not prompts:
            raise ValueError("impact_calibrate needs nonempty calibration prompts")
        model = rt.teacher_model or load_model(rt.config.models.teacher, trainable=False).to(rt.engine.device)
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        total: torch.Tensor | None = None
        count = 0
        with torch.no_grad():
            for start in range(0, len(prompts), self.cfg.batch_size):
                encoded = rt.teacher_tokenizer(
                    prompts[start : start + self.cfg.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=min(self.cfg.max_length, 256),
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                hidden = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states

                def get_hidden(layer: int, states: tuple = hidden) -> torch.Tensor:
                    return states[layer]

                inp = MapperInputs(0, len(hidden) - 1, TrainState(), teacher_hidden=get_hidden)
                scores = compute_bi_scores(inp).detach().float().cpu()
                total = scores if total is None else total + scores
                count += 1
        if was_training:
            model.train()
        assert total is not None and ctx.output is not None
        ctx.output.path.mkdir(parents=True, exist_ok=True)
        torch.save(total / count, ctx.output.path / "layer_scores.pt")
        return {"n_prompts": len(prompts), "n_batches": count, "layers": len(total)}

    def attach(self, art: Artifact, rt: Any) -> None:
        rt.extra["impact_layer_scores"] = torch.load(
            art.path / "layer_scores.pt", map_location="cpu", weights_only=True
        )

