"""SSFT offline attribution stage and self-contained SSFTStage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import torch

from baselines.longCoT.common import (
    chat_ids,
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
from baselines.longCoT.sgl_spectral.pipeline import BuildWeights
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import STAGES
from lmkd.eval.graders import extract_boxed
from lmkd.models.adapters import HFAutoAdapter
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import ResumableWriter, load_samples_dir, read_jsonl, save_samples, shard_of
from lmkd.pipeline.stage import Stage, StageContext

USER_TEMPLATE = "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
ANSWER_TEMPLATE = "</think> So, the final answer is \\boxed{{{answer}}}"


def compact_ig(scores: list[list[float]]) -> list[list[float]]:
    """Compact token IG into [count, sum(abs), sum(signed)] per segment.

    Source: SGL/src/sgl/ssft/attribution.py:163-171.
    """
    return [[len(s), round(sum(abs(x) for x in s), 8), round(sum(s), 8)] for s in scores]


def _segment_spans(tokenizer: Any, segments: list[str], offset: int) -> tuple[list[int], list[tuple[int, int]]]:
    joined = "".join(segments)
    encoded = tokenizer(joined, add_special_tokens=False, return_offsets_mapping=True)
    starts, cursor = [], 0
    for segment in segments:
        starts.append(cursor)
        cursor += len(segment)
    token_ranges = [[0, 0] for _ in segments]
    index = 0
    for position, (start, end) in enumerate(encoded["offset_mapping"]):
        if end <= start or not segments:
            continue
        while index + 1 < len(starts) and start >= starts[index + 1]:
            index += 1
        if token_ranges[index] == [0, 0]:
            token_ranges[index][0] = position + offset
        token_ranges[index][1] = position + offset + 1
    return list(encoded["input_ids"]), [(a, b) for a, b in token_ranges]


def _ig_example(
    tokenizer: Any,
    question: str,
    answer: str,
    segments: list[str],
    user_template: str,
    answer_template: str,
) -> tuple[list[int], list[tuple[int, int]], tuple[int, int]]:
    user_ids = chat_ids(tokenizer, user_template.format(input=question), enable_thinking=None)
    response_ids, spans = _segment_spans(tokenizer, segments, len(user_ids))
    answer_text = answer_template.format(answer=answer)
    answer_encoded = tokenizer(answer_text, add_special_tokens=False, return_offsets_mapping=True)
    brace = answer_text.rfind("\\boxed{")
    if brace < 0:
        raise ValueError("answer_template must include \\boxed{")
    first, last = brace + len("\\boxed{"), brace + len("\\boxed{") + len(answer)
    matching = [i for i, (a, b) in enumerate(answer_encoded["offset_mapping"]) if b > first and a < last]
    if not matching:
        raise ValueError("answer has no tokens")
    offset = len(user_ids) + len(response_ids)
    return (
        user_ids + response_ids + list(answer_encoded["input_ids"]),
        spans,
        (offset + matching[0], offset + matching[-1] + 1),
    )


def _integrated_gradients(
    model: Any,
    adapter: HFAutoAdapter,
    ids: list[int] | torch.Tensor,
    answer_span: tuple[int, int],
    pad_id: int,
    steps: int = 20,
    batch_size: int = 4,
) -> torch.Tensor:
    device = next(model.parameters()).device
    if isinstance(ids, torch.Tensor):
        tokens = ids.to(device)
        if tokens.dim() == 1:
            tokens = tokens[None]
    else:
        tokens = torch.tensor(ids, dtype=torch.long, device=device)[None]
    embed = adapter.embed_tokens(model)
    with torch.no_grad():
        source = embed(tokens)
        baseline = embed(torch.full_like(tokens, pad_id))
    total = torch.zeros_like(source)
    alphas = torch.linspace(0, 1, steps, device=device, dtype=source.dtype)
    start, end = answer_span
    head = adapter.lm_head(model)
    assert head is not None
    for first in range(0, steps, batch_size):
        alpha = alphas[first : first + batch_size, None, None]
        path = (baseline + alpha * (source - baseline).detach()).requires_grad_(True)
        hidden = adapter.forward_hidden(
            model, tokens.expand(len(alpha), -1), None, inputs_embeds=path
        ).last_hidden_state
        logits = head(hidden[:, start - 1 : end - 1])
        target = tokens[:, start:end].expand(len(alpha), -1).unsqueeze(-1)
        picked = logits.float().log_softmax(-1).gather(-1, target).sum()
        total += torch.autograd.grad(picked, path)[0].sum(0, keepdim=True)
    attribution = ((source - baseline) * total / steps).sum(-1)[0].float()
    return attribution / attribution.norm().clamp_min(1e-12)


@STAGES.register("ig_attribution", source="SGL/src/sgl/ssft/attribution.py:36-171")
class IGAttribution(Stage):
    """Integrated gradients from segment tokens to boxed-answer log probability.

    IG = (x−pad) mean_j ∇log p(answer|x_j); source: SGL ssft/attribution.py:36-171.
    """

    inputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    outputs: ClassVar[dict[str, str]] = {"signals": "signals"}
    supports_shards = True

    class Config(BaseConfig):
        model: str = ""
        dtype: str = "bf16"
        steps: int = 20
        batch_size: int = 4
        max_input_tokens: int = 16384
        user_template: str = USER_TEMPLATE
        answer_template: str = ANSWER_TEMPLATE
        gradient_checkpointing: bool = True

    def _prepare_model(self) -> tuple[Any, Any, HFAutoAdapter]:
        model, tokenizer, adapter = load_frozen(self._resolved_model, self.cfg.dtype)
        if self.cfg.gradient_checkpointing:
            if any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()):
                raise ValueError("IG gradient checkpointing needs zero dropout")
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
        return model, tokenizer, adapter

    def _one(self, row: dict, model: Any, tokenizer: Any, adapter: HFAutoAdapter) -> tuple[list[list[float]], bool]:
        segments = list(row["meta.segments"])
        question = row.get("meta.question", row.get("question", row.get("problem", "")))
        answer = row.get("meta.answer", row.get("answer"))
        if answer is None:
            answer = extract_boxed(str(row.get("meta.response_text", "")))
        if answer is None:
            raise ValueError(f"IG sample {row.get('meta.id')} has no answer")
        ids, spans, answer_span = _ig_example(
            tokenizer, str(question), str(answer), segments, self.cfg.user_template, self.cfg.answer_template
        )
        valid_spans = [(a, b) for a, b in spans if b > a]
        if len(ids) > self.cfg.max_input_tokens:
            return [[b - a, 0.0, 0.0] for a, b in valid_spans], True
        try:
            score = _integrated_gradients(
                model, adapter, ids, answer_span, int(tokenizer.pad_token_id), self.cfg.steps, self.cfg.batch_size
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return [[b - a, 0.0, 0.0] for a, b in valid_spans], True
        return compact_ig([score[a:b].cpu().tolist() for a, b in valid_spans]), False

    def attribute_rows(
        self,
        rows: list[dict],
        resolved_model: str,
        writer: ResumableWriter | None = None,
    ) -> tuple[list[dict], int]:
        """Compute integrated gradients attribution for rows using frozen student model."""
        self._resolved_model = resolved_model
        model, tokenizer, adapter = self._prepare_model()
        results = []
        oom = 0
        try:
            for row in rows:
                sid = str(row["meta.id"])
                if writer is not None and writer.done(sid):
                    continue
                scores, failed = self._one(row, model, tokenizer, adapter)
                sig_record = {"meta.id": sid, "segment_ig": scores}
                if writer is not None:
                    writer.write(sig_record)
                results.append(sig_record)
                oom += failed
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results, oom

    def run(self, ctx: StageContext) -> dict:
        source = input_rows(ctx, "data", "prev")
        path, rows = pending(ctx, "signals.jsonl", source)
        self._resolved_model = model_path(self.cfg.model, ctx)
        with ResumableWriter(path) as writer:
            computed, oom = self.attribute_rows(rows, self._resolved_model, writer=writer)
        write_signals_parquet(path)
        return {"n": len(rows), "oom": oom, "signals": ["segment_ig"]}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = merge_jsonl(ctx, shard_dirs, "signals.jsonl", input_rows(ctx, "data", "prev"))
        assert ctx.output is not None
        write_signals_parquet(ctx.output.path / "signals.jsonl")
        return {"n": len(rows), "signals": ["segment_ig"]}


@STAGES.register("ssft", source="SegmentSelectiveSFT-main/Attribution/get_important_segments.py")
class SSFTStage(Stage):
    """Self-contained end-to-end Segment-Selective SFT (SSFT) stage.

    Composes data segmentation, IGAttribution attribution computation, and BuildWeights
    cumulative attribution selection into a single closed composite stage.
    """

    inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    supports_shards = True

    class Config(IGAttribution.Config):
        source: dict | str | None = None
        segmenter: str = "paragraph"
        cumulative_ratio: float = 0.7
        coherence_max: float = 0.8
        always_keep: list[int] = [0, -2, -1]
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
            store_texts=True,
            chat_template=self.cfg.chat_template,
            model_path_str=self._resolved_model,
            tokenize_verify=self.cfg.tokenize_verify,
        )
        assert ctx.output is not None
        samples_path = ctx.output.path / "samples.jsonl"
        signals_path = ctx.output.path / "signals.jsonl"
        shard_rows = shard_of(prepared_rows, *ctx.shard)

        # Flow step 1: Attribution computation (delegate to IGAttribution)
        attributor = IGAttribution(
            IGAttribution.Config(
                model=self.cfg.model,
                dtype=self.cfg.dtype,
                steps=self.cfg.steps,
                batch_size=self.cfg.batch_size,
                max_input_tokens=self.cfg.max_input_tokens,
                user_template=self.cfg.user_template,
                answer_template=self.cfg.answer_template,
                gradient_checkpointing=self.cfg.gradient_checkpointing,
            )
        )
        with ResumableWriter(signals_path) as sig_writer:
            _, oom = attributor.attribute_rows(shard_rows, self._resolved_model, writer=sig_writer)
        write_signals_parquet(signals_path)
        all_signals = read_jsonl(signals_path)

        # Flow step 2: Build weights and cumulative selection (delegate to BuildWeights)
        builder = BuildWeights(
            BuildWeights.Config(
                chain=[
                    {
                        "type": "select_cumulative_strength",
                        "cumulative_ratio": self.cfg.cumulative_ratio,
                        "coherence_max": self.cfg.coherence_max,
                        "always_keep": self.cfg.always_keep,
                    }
                ],
            )
        )
        selected_rows, stats = builder.build_rows(shard_rows, all_signals, ctx.rt)

        with ResumableWriter(samples_path) as s_writer:
            for row in selected_rows:
                if not s_writer.done(str(row["meta.id"])):
                    s_writer.write(row)

        all_samples = read_jsonl(samples_path)
        save_samples(ctx.output.path, all_samples)
        return {"n": len(all_samples), "oom": oom, "selection_stats": stats}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        source_rows = resolve_source_rows(ctx, self.cfg.source)
        self._resolved_model = model_path(self.cfg.model, ctx)
        prepared = ensure_segmented(
            source_rows,
            ctx,
            segmenter=self.cfg.segmenter,
            store_texts=True,
            chat_template=self.cfg.chat_template,
            model_path_str=self._resolved_model,
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
