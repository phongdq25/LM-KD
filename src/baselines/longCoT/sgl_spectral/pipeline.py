"""SGL-Spectral pipeline stages: SpectralCapture, AnswerGain, and SGLSpectralStage."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, ClassVar

import torch
from pydantic import field_validator

from baselines.longCoT.common import (
    ensure_segmented,
    input_rows,
    load_frozen,
    merge_jsonl,
    model_path,
    pending,
    resolve_source_rows,
    selection_stats,
    write_signals_parquet,
)
from baselines.longCoT.sgl_spectral.spectral import (
    analytic_gradients_and_entropies,
    effective_rank,
    step_strengths,
)
from lmkd.core.batch import FIELDS, FieldSpec
from lmkd.core.config import ModelSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import STAGES, TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import CorpusTransform, Transform, run_transforms
from lmkd.eval.graders import extract_boxed
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.loading import load_tokenizer
from lmkd.models.logits_view import LogitsView
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import ResumableWriter, load_samples_dir, read_jsonl, save_samples, shard_of
from lmkd.pipeline.stage import Stage, StageContext
from lmkd.rollout.base import ScoreSpec
from lmkd.rollout.vllm_colocate import make_llm, score_with_llm

FIELDS.register(FieldSpec("signals.k_star", "meta"))
FIELDS.register(FieldSpec("meta.skip_signal_transforms", "meta"))


def _span(sample: dict) -> tuple[int, int]:
    start, end = map(int, sample["seq.response_span"])
    if start < 1 or end <= start or end > len(sample["student.input_ids"]):
        raise ValueError(f"invalid response span for {sample.get('meta.id')}: {(start, end)}")
    return start, end


@STAGES.register("spectral_capture")
class SpectralCapture(Stage):
    """Capture per-step leverage and entropy from closed-form CE gradients.

    G_t = Wᵀ(softmax(Wh[t-1]) − onehot(y_t)).
    """

    inputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    outputs: ClassVar[dict[str, str]] = {"signals": "signals"}
    supports_shards = True

    class Config(BaseConfig):
        model: str = ""
        dtype: str = "bf16"
        energy_cutoff: float = 0.95
        chunk: int = 1024
        device: str = "auto"
        verify: bool = False

    def _capture(
        self,
        sample: dict,
        model: Any,
        adapter: HFAutoAdapter,
        weight: torch.Tensor,
        verify: bool,
    ) -> dict:
        start, end = _span(sample)
        ids = torch.tensor(sample["student.input_ids"], dtype=torch.long, device=weight.device)
        with torch.no_grad():
            hidden = adapter.forward_hidden(model, ids[None], None).last_hidden_state[0]
            rows, targets = hidden[start - 1 : end - 1], ids[start:end]
            grads, entropies = analytic_gradients_and_entropies(rows, targets, weight, self.cfg.chunk)
        if verify:
            probe = rows[: min(3, len(rows))].detach().float().requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(probe @ weight.T, targets[: len(probe)], reduction="sum")
            ref = torch.autograd.grad(loss, probe)[0]
            if (grads[: len(probe)] - ref).abs().max() >= 1e-2:
                raise AssertionError("analytic gradients differ from autograd")
        u, singular, _ = torch.linalg.svd(grads.float(), full_matrices=False)
        rank = effective_rank(singular, self.cfg.energy_cutoff)
        relative = [(int(a) - start, int(b) - start) for a, b in sample["seq.steps"]]
        leverage = u[:, :rank].square().sum(-1)
        result = {
            "meta.id": str(sample["meta.id"]),
            "step_strength": step_strengths(leverage, relative),
            "step_entropy": step_strengths(entropies, relative),
            "k_star": rank,
        }
        del grads, entropies
        return result

    def capture_rows(
        self,
        samples: list[dict],
        resolved_model: str,
        writer: ResumableWriter | None = None,
    ) -> list[dict]:
        """Compute spectral signals for samples using frozen student model."""
        self._resolved_model = resolved_model
        model, _, adapter = load_frozen(resolved_model, self.cfg.dtype, self.cfg.device)
        head = adapter.lm_head(model)
        if head is None or not hasattr(head, "weight"):
            raise ValueError("spectral_capture needs an lm_head with weight")
        head_weight = head.weight
        assert isinstance(head_weight, torch.Tensor)
        weight = head_weight.float()  # Cast once for the whole corpus.
        results = []
        computed = 0
        try:
            for sample in samples:
                sid = str(sample["meta.id"])
                if writer is not None and writer.done(sid):
                    continue
                captured = self._capture(sample, model, adapter, weight, self.cfg.verify and computed < 3)
                if writer is not None:
                    writer.write(captured)
                results.append(captured)
                computed += 1
        finally:
            del model, weight
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results

    def run(self, ctx: StageContext) -> dict:
        source = input_rows(ctx, "data", "prev")
        path, samples = pending(ctx, "signals.jsonl", source)
        self._resolved_model = model_path(self.cfg.model, ctx)
        with ResumableWriter(path) as writer:
            computed = self.capture_rows(samples, self._resolved_model, writer=writer)
        write_signals_parquet(path)
        return {"n": len(samples), "computed": len(computed), "signals": ["step_strength", "step_entropy"]}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = merge_jsonl(ctx, shard_dirs, "signals.jsonl", input_rows(ctx, "data", "prev"))
        assert ctx.output is not None
        write_signals_parquet(ctx.output.path / "signals.jsonl")
        return {"n": len(rows), "signals": ["step_strength", "step_entropy"]}


@STAGES.register("answer_gain")
class AnswerGain(Stage):
    """Measure each step's change in final-answer log probability.

    gain_k = log p(answer|prefix_k) − log p(answer|prefix_{k−1}).
    """

    inputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    outputs: ClassVar[dict[str, str]] = {"signals": "signals"}
    supports_shards = True

    class Config(BaseConfig):
        model: str = ""
        probe: str = "\n\nTherefore, the final answer is \\boxed{"
        max_model_len: int = 16384
        gpu_memory_utilization: float = 0.4
        engine: str = "vllm"

    def _score_hf(
        self,
        seqs: list[list[int]],
        starts: list[int],
        model: Any,
        adapter: HFAutoAdapter,
    ) -> list[float]:
        head = adapter.lm_head(model)
        assert head is not None
        device = next(model.parameters()).device
        result = []
        with torch.no_grad():
            for seq, start in zip(seqs, starts):
                ids = torch.tensor(seq, dtype=torch.long, device=device)[None]
                hidden = adapter.forward_hidden(model, ids, None).last_hidden_state
                labels = torch.full_like(ids, -100)
                labels[:, start:] = ids[:, start:]
                view = LogitsView(hidden=hidden, lm_head=head)
                result.append(float(view.target_logp(labels, chunk_size=256).sum()))
        return result

    def _scores(self, seqs: list[list[int]], starts: list[int]) -> list[float]:
        if self.cfg.engine == "hf":
            return self._score_hf(seqs, starts, self._score_model, self._adapter)
        if self.cfg.engine != "vllm":
            raise ValueError(f"unknown answer_gain engine {self.cfg.engine}")
        distributions = score_with_llm(self._score_model, seqs, ScoreSpec(top_k=0))
        return [
            sum(dist[pos][seq[pos]] for pos in range(start, len(seq)))
            for seq, start, dist in zip(seqs, starts, distributions)
        ]

    def _sample_gains(self, sample: dict, tokenizer: Any) -> list[float] | None:
        response = str(sample.get("meta.response_text", ""))
        answer = extract_boxed(response)
        if not answer:
            return None
        ids = list(sample["student.input_ids"])
        probe_ids = tokenizer(self.cfg.probe, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer + "}", add_special_tokens=False)["input_ids"]
        cuts = [_span(sample)[0], *[int(end) for _, end in sample["seq.steps"]]]
        seqs, starts, positions = [], [], []
        for index, cut in enumerate(cuts):
            seq = ids[:cut] + list(probe_ids) + list(answer_ids)
            if len(seq) >= self.cfg.max_model_len:
                continue
            seqs.append(seq)
            starts.append(len(seq) - len(answer_ids))
            positions.append(index)
        if not seqs or positions[0] != 0:
            return None
        logp: list[float | None] = [None] * len(cuts)
        for index, value in zip(positions, self._scores(seqs, starts)):
            logp[index] = value
        assert logp[0] is not None
        values = [float(logp[0])]
        for item in logp[1:]:
            values.append(values[-1] if item is None else float(item))
        return [values[i] - values[i - 1] for i in range(1, len(values))]

    def run(self, ctx: StageContext) -> dict:
        source = input_rows(ctx, "data", "prev")
        path, rows = pending(ctx, "signals.jsonl", source)
        self._resolved_model = model_path(self.cfg.model, ctx)
        tokenizer = load_tokenizer(ModelSpec(path=self._resolved_model))
        if self.cfg.engine == "hf":
            self._score_model, _, self._adapter = load_frozen(self._resolved_model, "fp32")
        elif self.cfg.engine == "vllm":
            self._score_model = make_llm(
                self._resolved_model,
                max_model_len=self.cfg.max_model_len,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                enable_prefix_caching=True,
                enforce_eager=True,
            )
        else:
            raise ValueError(f"unknown answer_gain engine {self.cfg.engine}")
        skipped = 0
        try:
            with ResumableWriter(path) as writer:
                for row in rows:
                    sid = str(row["meta.id"])
                    if writer.done(sid):
                        continue
                    gains = self._sample_gains(row, tokenizer)
                    if gains is None:
                        n_steps = len(row.get("seq.steps", []))
                        if n_steps > 0:
                            gains = [0.0] * n_steps
                        else:
                            skipped += 1
                            continue
                    writer.write({"meta.id": sid, "step_gain": gains})
        finally:
            del self._score_model
            gc.collect()
        write_signals_parquet(path)
        return {"n": len(rows) - skipped, "skipped": skipped, "signals": ["step_gain"]}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = merge_jsonl(ctx, shard_dirs, "signals.jsonl", input_rows(ctx, "data", "prev"))
        assert ctx.output is not None
        write_signals_parquet(ctx.output.path / "signals.jsonl")
        return {"n": len(rows), "signals": ["step_gain"]}


def _signal_aliases(ctx: StageContext) -> dict[str, dict[str, dict]]:
    aliases = {}
    for alias, art in ctx.inputs.items():
        if alias.startswith("signals"):
            aliases[alias] = {str(row["meta.id"]): row for row in read_jsonl(art.path / "signals.jsonl")}
    return aliases


def _join_signals(
    rows: list[dict],
    aliases: dict[str, dict[str, dict]],
    mapping: dict[str, str],
    missing: str,
) -> tuple[list[dict], int]:
    joined, dropped = [], 0
    for row in rows:
        sid = str(row["meta.id"])
        absent = [alias for alias, values in aliases.items() if sid not in values]
        if absent:
            if missing == "error":
                raise ValueError(f"signals missing for sample {sid}: {absent}")
            if missing == "drop":
                dropped += 1
                continue
            if missing != "skip_transform":
                raise ValueError(f"unknown missing policy {missing}")
        sample = dict(row)
        for values in aliases.values():
            if sid not in values:
                continue
            for key, value in values[sid].items():
                if key == "meta.id":
                    continue
                signal = key.removeprefix("signals.")
                target = mapping.get(signal, f"signals.{signal}")
                if target not in FIELDS:
                    if not target.startswith("signals."):
                        raise ValueError(f"unregistered mapped signal field {target!r}")
                    FIELDS.register(FieldSpec(target, "meta"))
                sample[target] = value
        if absent:
            sample["meta.skip_signal_transforms"] = True
        joined.append(sample)
    return joined, dropped


def _run_with_missing(rows: list[dict], transforms: list[Transform], ctx: DataContext) -> list[dict]:
    current = rows
    for transform in transforms:
        required = transform.required_keys(transform.cfg)
        eligible = [row for row in current if all(key in row for key in required)]
        if isinstance(transform, CorpusTransform) and eligible:
            transform.fit(eligible, ctx)
        next_rows = []
        for row in current:
            if row.get("meta.skip_signal_transforms") and not all(key in row for key in required):
                next_rows.append(row)
                continue
            transformed = transform(row, ctx)
            if transformed is not None:
                next_rows.append(transformed)
        current = next_rows
    for row in current:
        row.pop("meta.skip_signal_transforms", None)
    return current


@STAGES.register("build_weights")
class BuildWeights(Stage):
    """Join offline signals by id and run selection/allocation transforms.

    """

    inputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    supports_shards = True

    class Config(BaseConfig):
        chain: list[dict] = []
        signals_map: dict[str, str] = {}
        check_mass: bool = True
        missing: str = "error"

        @field_validator("chain")
        @classmethod
        def validate_chain(cls, chain: list[dict]) -> list[dict]:
            """Reject invalid nested transform specs during stage validation."""
            try:
                for spec in chain:
                    TRANSFORMS.validate(spec)
            except (KeyError, ValueError, NotImplementedError) as exc:
                raise ValueError(str(exc)) from exc
            return chain

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        """Expose the outputs of the configured transform chain to recipe validation."""
        keys: set[str] = set()
        for spec in cfg.chain:
            meta, transform_cfg = TRANSFORMS.validate(spec)
            keys.update(meta.obj.provided_keys(transform_cfg))
        return tuple(sorted(keys))

    def _transform(self, rows: list[dict], rt: Any) -> list[dict]:
        data_ctx = getattr(rt, "data_ctx", None)
        if data_ctx is None:
            data_ctx = DataContext(student_tokenizer=getattr(rt, "student_tokenizer", None))
        transforms = [TRANSFORMS.build(spec) for spec in self.cfg.chain]
        for transform in transforms:
            transform.setup(rt)
        if any(r.get("meta.skip_signal_transforms") for r in rows):
            return _run_with_missing(rows, transforms, data_ctx)
        return run_transforms(rows, transforms, data_ctx)

    def build_rows(
        self,
        rows: list[dict],
        signals: list[dict] | dict[str, dict] | dict[str, dict[str, dict]],
        rt: Any = None,
    ) -> tuple[list[dict], dict]:
        """Join signals and apply transform chain to rows in memory."""
        if isinstance(signals, list):
            aliases = {"signals": {str(s["meta.id"]): s for s in signals}}
        elif isinstance(signals, dict):
            if signals and isinstance(next(iter(signals.values())), dict):
                first_val = next(iter(signals.values()))
                if any(k.startswith("step_") or k in ("k_star", "meta.id") for k in first_val):
                    aliases = {"signals": {str(k): v for k, v in signals.items()}}
                else:
                    aliases = signals
            else:
                aliases = {"signals": {}}
        else:
            aliases = {"signals": {}}

        joined, _ = _join_signals(rows, aliases, self.cfg.signals_map, self.cfg.missing)
        transformed = self._transform(joined, rt)
        stats = selection_stats(transformed)
        return transformed, stats

    def run(self, ctx: StageContext) -> dict:
        source = input_rows(ctx, "data")
        aliases = _signal_aliases(ctx)
        transformed, stats = self.build_rows(source, aliases, ctx.rt)
        path, rows = pending(ctx, "samples.jsonl", transformed)
        with ResumableWriter(path) as writer:
            for row in rows:
                if not writer.done(str(row["meta.id"])):
                    writer.write(row)
        assert ctx.output is not None
        save_samples(ctx.output.path, read_jsonl(path))
        return {
            "n": len(rows),
            "dropped": len(source) - len(transformed),
            "selection_stats": stats,
        }

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = merge_jsonl(ctx, shard_dirs, "samples.jsonl", input_rows(ctx, "data"))
        assert ctx.output is not None
        save_samples(ctx.output.path, rows)
        return {"n": len(rows), "selection_stats": selection_stats(rows)}

    def load_samples(self, art: Artifact) -> list[dict]:
        return load_samples_dir(art.path)


@STAGES.register("sgl_spectral")
class SGLSpectralStage(Stage):
    """Self-contained end-to-end Spectral Guided Learning (SGL-Spectral) stage.

    Composes data segmentation, SpectralCapture signal computation, and BuildWeights
    top-p energy selection into a single closed composite stage.
    """

    inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    supports_shards = True

    class Config(SpectralCapture.Config):
        source: dict | str | None = None
        top_p: float = 0.95
        segmenter: str = "sentence"
        chat_template: bool = True
        tokenize_verify: bool = True

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        return ("student.input_ids", "seq.loss_mask", "seq.steps", "seq.response_span", "meta.selected_steps")

    def run(self, ctx: StageContext) -> dict:
        source_rows = resolve_source_rows(ctx, self.cfg.source)
        self._resolved_model = model_path(self.cfg.model, ctx)
        prepared_rows = ensure_segmented(
            source_rows,
            ctx,
            segmenter=self.cfg.segmenter,
            store_texts=False,
            chat_template=self.cfg.chat_template,
            model_path=self._resolved_model,
            tokenize_verify=self.cfg.tokenize_verify,
        )
        assert ctx.output is not None
        samples_path = ctx.output.path / "samples.jsonl"
        signals_path = ctx.output.path / "signals.jsonl"
        shard_rows = shard_of(prepared_rows, *ctx.shard)

        # Flow step 1: Capture spectral signals (delegate to SpectralCapture)
        capturer = SpectralCapture(
            SpectralCapture.Config(
                model=self.cfg.model,
                dtype=self.cfg.dtype,
                energy_cutoff=self.cfg.energy_cutoff,
                chunk=self.cfg.chunk,
                device=self.cfg.device,
                verify=self.cfg.verify,
            )
        )
        with ResumableWriter(signals_path) as sig_writer:
            capturer.capture_rows(shard_rows, self._resolved_model, writer=sig_writer)
        write_signals_parquet(signals_path)
        all_signals = read_jsonl(signals_path)

        # Flow step 2: Build weights and top-p energy selection (delegate to BuildWeights)
        builder = BuildWeights(
            BuildWeights.Config(
                chain=[{"type": "select_energy_top_p", "p": self.cfg.top_p}],
            )
        )
        selected_rows, stats = builder.build_rows(shard_rows, all_signals, ctx.rt)

        with ResumableWriter(samples_path) as s_writer:
            for row in selected_rows:
                if not s_writer.done(str(row["meta.id"])):
                    s_writer.write(row)

        all_samples = read_jsonl(samples_path)
        save_samples(ctx.output.path, all_samples)
        return {"n": len(all_samples), "computed": len(all_signals), "selection_stats": stats}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        source_rows = resolve_source_rows(ctx, self.cfg.source)
        self._resolved_model = model_path(self.cfg.model, ctx)
        prepared = ensure_segmented(
            source_rows,
            ctx,
            segmenter=self.cfg.segmenter,
            store_texts=False,
            chat_template=self.cfg.chat_template,
            model_path=self._resolved_model,
            tokenize_verify=self.cfg.tokenize_verify,
        )
        merged_samples = merge_jsonl(ctx, shard_dirs, "samples.jsonl", prepared)
        merge_jsonl(ctx, shard_dirs, "signals.jsonl", prepared)
        assert ctx.output is not None
        save_samples(ctx.output.path, merged_samples)
        write_signals_parquet(ctx.output.path / "signals.jsonl")
        return {"n": len(merged_samples), "selection_stats": selection_stats(merged_samples)}

    def load_samples(self, art: Artifact) -> list[dict]:
        return load_samples_dir(art.path)
