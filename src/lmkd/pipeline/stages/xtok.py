"""Cross-tokenizer stages: vocab map and alignment."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from typing import Any

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import STAGES, TRANSFORMS
from lmkd.data.aligners.base import Aligner
from lmkd.data.context import DataContext
from lmkd.data.vocab_map import VocabMap, vocab_map_kwargs
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import load_samples_dir, save_samples
from lmkd.pipeline.stage import Stage, StageContext


def tokenizer_fingerprint(tok: Any) -> dict:
    """Name/path, size and chat-template hash of a tokenizer, for stage fingerprints."""
    if tok is None:
        return {}
    tmpl = getattr(tok, "chat_template", "") or ""
    return {
        "name": getattr(tok, "name_or_path", ""),
        "len": len(tok),
        "chat_template": hashlib.sha256(tmpl.encode()).hexdigest()[:8],
    }


@STAGES.register("vocab_map")
class VocabMapStage(Stage):
    """Student/teacher vocabulary overlap and id mappings (``VocabMap``), exposed as ``rt.data_ctx.vocab_map``.

    ``kind`` is only ``overlap``; ``lambda_reg`` is kept for compatibility and unused (the ridge W is computed in
    ``impact.prepare``). Put it BEFORE any ``align_precompute`` stage; ``attach`` loads the map also when the
    stage output was cached.
    """

    code_version = "1"

    class Config(BaseConfig):
        kind: str = "overlap"
        lambda_reg: float = 1e-3
        mapping_file: str = ""
        space_mask_mode: str = ""
        student_vocab_size: int = 0
        teacher_vocab_size: int = 0
        student_byteify: str = ""
        teacher_byteify: str = ""

    def setup(self, rt: Any) -> None:
        self._rt = rt

    def fingerprint(self) -> dict:
        rt = self._rt
        return {
            **self.cfg.model_dump(),
            "student": tokenizer_fingerprint(rt.student_tokenizer),
            "teacher": tokenizer_fingerprint(rt.teacher_tokenizer),
        }

    def run(self, ctx: StageContext) -> dict:
        rt = ctx.rt
        if rt.teacher_tokenizer is None:
            raise ValueError("vocab_map stage needs models.teacher (a teacher tokenizer)")
        if self.cfg.kind != "overlap":
            raise ValueError(f"vocab_map.kind must be 'overlap', got {self.cfg.kind!r}")
        assert ctx.output is not None
        vm = VocabMap.build(rt.student_tokenizer, rt.teacher_tokenizer, **vocab_map_kwargs(self.cfg))
        vm.save(ctx.output.path / "vocab_map.pt")
        return {"n_overlap": int(vm.overlap_s.numel()), "same_vocab": vm.same_vocab}

    def attach(self, art: Artifact, rt: Any) -> None:
        rt.data_ctx.vocab_map = VocabMap.load(art.path / "vocab_map.pt")


def _align_chunk(args: tuple[list[dict], Any, DataContext]) -> list[dict]:
    samples, transform, ctx = args
    out = (transform(s, ctx) for s in samples)
    return [s for s in out if s is not None]


@STAGES.register("align_precompute")
class AlignPrecompute(Stage):
    """Run an Aligner over the whole dataset offline and store the samples with the new ``align.*`` keys.

    Reads the samples of input slot ``data`` (or ``prev``), applies the ``align`` transform to each (optionally in
    ``num_workers`` processes) and writes them in the format of ``prepare_dataset``. The vocab map (if the aligner
    needs one) comes from a preceding ``vocab_map`` stage / ``load_vocab_map`` transform via ``rt.data_ctx``.
    """

    code_version = "1"

    class Config(BaseConfig):
        aligner: dict | str = {}
        on_fail: str = "drop"
        num_workers: int = 0

    def setup(self, rt: Any) -> None:
        self._rt = rt
        self._tf = TRANSFORMS.build({"type": "align", "aligner": self.cfg.aligner, "on_fail": self.cfg.on_fail})

    def fingerprint(self) -> dict:
        rt = self._rt
        aligner: Aligner = self._tf.aligner
        return {
            "aligner": aligner.fingerprint(),
            "on_fail": self.cfg.on_fail,
            "student": tokenizer_fingerprint(rt.student_tokenizer),
            "teacher": tokenizer_fingerprint(getattr(rt, "teacher_tokenizer", None)),
        }

    def _data_ctx(self, rt: Any) -> DataContext:
        ctx = getattr(rt, "data_ctx", None)
        if ctx is not None:
            return ctx
        return DataContext(
            student_tokenizer=rt.student_tokenizer, teacher_tokenizer=getattr(rt, "teacher_tokenizer", None)
        )

    def run(self, ctx: StageContext) -> dict:
        src = ctx.inputs.get("data") or ctx.inputs.get("prev")
        if src is None:
            raise ValueError("align_precompute needs an input stage (inputs: {data: <stage name>})")
        samples = load_samples_dir(src.path)
        dctx = self._data_ctx(ctx.rt)
        n_workers = self.cfg.num_workers
        if n_workers > 1 and len(samples) > 1:
            size = max(1, len(samples) // (n_workers * 2))
            chunks = [(samples[i : i + size], self._tf, dctx) for i in range(0, len(samples), size)]
            with mp.Pool(n_workers) as pool:
                out = [s for part in pool.map(_align_chunk, chunks) for s in part]
        else:
            out = _align_chunk((samples, self._tf, dctx))
        assert ctx.output is not None
        save_samples(ctx.output.path, out)
        return {"n": len(out), "dropped": len(samples) - len(out)}

    def load_samples(self, art: Artifact) -> list[dict]:
        return load_samples_dir(art.path)
