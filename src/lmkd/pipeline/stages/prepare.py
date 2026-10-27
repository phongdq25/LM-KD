"""Prepare dataset offline: load records from source, apply transforms, and cache to Arrow/JSONL."""
from __future__ import annotations

import hashlib
import logging
from typing import Any, ClassVar

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import SOURCES, STAGES, TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import CorpusTransform, run_transforms
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import load_samples_dir, save_samples
from lmkd.pipeline.stage import Stage, StageContext

logger = logging.getLogger(__name__)


def _process_chunk(args: tuple[list[dict], list[Any], DataContext]) -> list[dict]:
    chunk_records, transforms, ctx = args
    return run_transforms(chunk_records, transforms, ctx)


@STAGES.register(
    "prepare_dataset",
    source="lmkd/pipeline/stages/prepare.py",
)
class PrepareDataset(Stage):
    """Offline dataset preparation pipeline stage.

    Runs raw source loading and offline transforms (e.g. tokenization, formatting)
    once, caching results to Arrow / JSONL.
    """

    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}

    class Config(BaseConfig):
        source: dict
        transforms: list[dict | str] = []
        num_proc: int = 1
        limit: int | None = None

    def setup(self, rt: Any) -> None:
        self._rt = rt

    def fingerprint(self) -> dict:
        cfg_dict = self.cfg.model_dump()
        source_obj = SOURCES.build(self.cfg.source)
        tok_info = {}
        rt = getattr(self, "_rt", None)
        if rt is not None:
            st_tok = getattr(rt, "student_tokenizer", None)
            if st_tok is not None:
                name = getattr(st_tok, "name_or_path", "")
                tmpl = getattr(st_tok, "chat_template", "") or ""
                tmpl_hash = hashlib.sha256(tmpl.encode()).hexdigest()[:8]
                tok_info["student"] = {"name": name, "len": len(st_tok), "chat_template": tmpl_hash}
            te_tok = getattr(rt, "teacher_tokenizer", None)
            if te_tok is not None:
                name = getattr(te_tok, "name_or_path", "")
                tmpl = getattr(te_tok, "chat_template", "") or ""
                tmpl_hash = hashlib.sha256(tmpl.encode()).hexdigest()[:8]
                tok_info["teacher"] = {"name": name, "len": len(te_tok), "chat_template": tmpl_hash}
        return {**cfg_dict, "source_fp": source_obj.fingerprint(), "tokenizers": tok_info}

    def run(self, ctx: StageContext) -> dict:
        source = SOURCES.build(self.cfg.source)
        if self.cfg.source.get("type") == "artifact_records":
            node = self.cfg.source["node"]
            art = ctx.inputs.get(node, ctx.inputs.get("prev"))
            if art is None:
                raise ValueError(f"artifact_records source needs pipeline input node {node!r}")
            source.bind(art)
        records = list(source.records())
        if self.cfg.limit is not None:
            records = records[: self.cfg.limit]

        data_ctx = getattr(ctx.rt, "data_ctx", None)
        if data_ctx is None:
            data_ctx = DataContext(
                student_tokenizer=getattr(ctx.rt, "student_tokenizer", None),
                teacher_tokenizer=getattr(ctx.rt, "teacher_tokenizer", None),
                artifacts=getattr(ctx.rt, "artifacts", None),
            )

        transforms = [TRANSFORMS.build(t) for t in self.cfg.transforms]
        for t in transforms:
            if hasattr(t, "setup"):
                t.setup(ctx.rt)

        samples: list[dict] | None = None
        if self.cfg.num_proc > 1 and any(isinstance(t, CorpusTransform) for t in transforms):
            logger.warning("num_proc ignored: CorpusTransform must be fitted on the whole corpus, not per chunk")
        elif self.cfg.num_proc > 1 and len(records) > 0:
            try:
                import multiprocessing as mp
                import pickle

                pickle.dumps((transforms, data_ctx))
                chunk_size = max(1, len(records) // (self.cfg.num_proc * 2))
                chunks = [
                    (records[i : i + chunk_size], transforms, data_ctx)
                    for i in range(0, len(records), chunk_size)
                ]
                with mp.Pool(self.cfg.num_proc) as pool:
                    chunk_results = pool.map(_process_chunk, chunks)
                samples = [s for cr in chunk_results for s in cr]
            except Exception as e:
                logger.warning(
                    "Multiprocessing failed for transforms (%s), falling back to 1 process", e
                )
                samples = None

        if samples is None:
            samples = run_transforms(records, transforms, data_ctx)

        assert ctx.output is not None
        save_samples(ctx.output.path, samples)
        return {"n": len(samples), "dropped": len(records) - len(samples)}

    def load_samples(self, art: Artifact) -> list[dict]:
        return load_samples_dir(art.path)

