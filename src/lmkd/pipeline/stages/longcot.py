"""Offline Long-CoT signal capture, attribution, prefix search, and weight building."""
from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from pydantic import field_validator

from lmkd.compat import vllm_api
from lmkd.core.batch import FIELDS, FieldSpec
from lmkd.core.config import ModelSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import SEGMENTERS, STAGES, TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import CorpusTransform, Transform, run_transforms
from lmkd.eval.graders import extract_boxed
from lmkd.losses.functional.spectral import analytic_gradients_and_entropies, effective_rank, step_strengths
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.loading import load_model, load_tokenizer
from lmkd.models.logits_view import LogitsView
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import ResumableWriter, load_samples_dir, read_jsonl, save_samples, shard_of, write_jsonl
from lmkd.pipeline.stage import Stage, StageContext
from lmkd.rollout.base import ScoreSpec
from lmkd.rollout.vllm_colocate import make_llm, score_with_llm

JUDGE_TEMPLATE = """
You are a reasoning evaluator.

You are given a partial reasoning prefix extracted from a longer chain-of-thought.
Your task is to judge whether this prefix already contains the essential logical structure and key transformations needed to complete the solution.

- Reply "[ENOUGH]" if the prefix establishes the core reasoning steps such that the remaining reasoning is straightforward or routine.
- Reply "[NOT_ENOUGH]" if any crucial reasoning step is still missing, making it difficult to reliably complete the solution.

Reply with exactly one token: [ENOUGH] or [NOT_ENOUGH].

Question:
{question}

Partial reasoning:
{reasoning_part}
"""

USER_TEMPLATE = "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
ANSWER_TEMPLATE = "</think> So, the final answer is \\boxed{{{answer}}}"

FIELDS.register(FieldSpec("signals.k_star", "meta"))
FIELDS.register(FieldSpec("meta.skip_signal_transforms", "meta"))


def _input_art(ctx: StageContext, *names: str) -> Artifact:
    for name in names:
        if name in ctx.inputs:
            return ctx.inputs[name]
    raise ValueError(f"missing input artifact: expected one of {names}")


def _input_rows(ctx: StageContext, *names: str) -> list[dict]:
    art = _input_art(ctx, *names)
    for filename in ("samples.jsonl", "records.jsonl"):
        if (art.path / filename).exists():
            return read_jsonl(art.path / filename)
    raise FileNotFoundError(f"no samples.jsonl or records.jsonl in {art.path}")


def _pending(ctx: StageContext, filename: str, rows: list[dict]) -> tuple[Path, list[dict]]:
    assert ctx.output is not None
    path = ctx.output.path / filename
    return path, shard_of(rows, *ctx.shard)


def _merge_jsonl(ctx: StageContext, shard_dirs: list[Path], filename: str,
                 source_rows: list[dict]) -> list[dict]:
    by_id = {}
    for folder in shard_dirs:
        for row in read_jsonl(folder / filename):
            sid = str(row["meta.id"])
            if sid in by_id:
                raise ValueError(f"duplicate shard id {sid}")
            by_id[sid] = row
    ordered = [by_id[str(row["meta.id"])] for row in source_rows if str(row["meta.id"]) in by_id]
    assert ctx.output is not None
    write_jsonl(ctx.output.path / filename, ordered)
    return ordered


def _write_signals_parquet(path: Path) -> None:
    try:
        import pandas as pd

        pd.DataFrame(read_jsonl(path)).to_parquet(path.with_suffix(".parquet"), index=False)
    except ImportError:
        pass


def _load_frozen(path: str, dtype: str = "bf16", device: str = "auto") -> tuple[Any, Any, HFAutoAdapter]:
    spec = ModelSpec(path=path, dtype=dtype)
    model = load_model(spec, trainable=False)
    target = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model.to(target)
    return model, load_tokenizer(spec), HFAutoAdapter(HFAutoAdapter.Config())


def _model_path(configured: str, ctx: StageContext) -> str:
    if configured:
        return configured
    spec = getattr(getattr(ctx.rt, "config", None), "models", None)
    student = getattr(spec, "student", None)
    if student is None:
        raise ValueError("stage model is empty and runtime has no models.student")
    return str(student.path)


def _span(sample: dict) -> tuple[int, int]:
    start, end = map(int, sample["seq.response_span"])
    if start < 1 or end <= start or end > len(sample["student.input_ids"]):
        raise ValueError(f"invalid response span for {sample.get('meta.id')}: {(start, end)}")
    return start, end


@STAGES.register("spectral_capture", source="SGL/src/sgl/signals/capture.py:80-224")
class SpectralCapture(Stage):
    """Capture per-step leverage and entropy from closed-form CE gradients.

    G_t = Wᵀ(softmax(Wh[t-1]) − onehot(y_t)); source: SGL signals/capture.py:80-224.
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

    def _capture(self, sample: dict, model: Any, adapter: HFAutoAdapter,
                 weight: torch.Tensor, verify: bool) -> dict:
        start, end = _span(sample)
        ids = torch.tensor(sample["student.input_ids"], dtype=torch.long, device=weight.device)
        with torch.no_grad():
            hidden = adapter.forward_hidden(model, ids[None], None).last_hidden_state[0]
            rows, targets = hidden[start - 1:end - 1], ids[start:end]
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
        result = {"meta.id": str(sample["meta.id"]),
                  "step_strength": step_strengths(leverage, relative),
                  "step_entropy": step_strengths(entropies, relative), "k_star": rank}
        del grads, entropies
        return result

    def run(self, ctx: StageContext) -> dict:
        source = _input_rows(ctx, "data", "prev")
        path, samples = _pending(ctx, "signals.jsonl", source)
        model, _, adapter = _load_frozen(_model_path(self.cfg.model, ctx), self.cfg.dtype, self.cfg.device)
        head = adapter.lm_head(model)
        if head is None or not hasattr(head, "weight"):
            raise ValueError("spectral_capture needs an lm_head with weight")
        head_weight = head.weight
        assert isinstance(head_weight, torch.Tensor)
        weight = head_weight.float()  # Cast once for the whole corpus.
        computed = 0
        with ResumableWriter(path) as writer:
            for sample in samples:
                if writer.done(str(sample["meta.id"])):
                    continue
                writer.write(self._capture(sample, model, adapter, weight, self.cfg.verify and computed < 3))
                computed += 1
        _write_signals_parquet(path)
        del model, weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"n": len(samples), "computed": computed, "signals": ["step_strength", "step_entropy"]}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = _merge_jsonl(ctx, shard_dirs, "signals.jsonl", _input_rows(ctx, "data", "prev"))
        assert ctx.output is not None
        _write_signals_parquet(ctx.output.path / "signals.jsonl")
        return {"n": len(rows), "signals": ["step_strength", "step_entropy"]}


def compact_ig(scores: list[list[float]]) -> list[list[float]]:
    """Compact token IG into [count, sum(abs), sum(signed)] per segment.

    Source: SGL/src/sgl/ssft/attribution.py:163-171.
    """
    return [[len(s), round(sum(abs(x) for x in s), 8), round(sum(s), 8)] for s in scores]


@dataclass
class PrefixSearch:
    """Binary search for the shortest judged-sufficient sentence prefix.

    Source: SGL/src/sgl/palign/truncate.py:53-91.
    """

    question: str
    sentences: list[str]
    left: int = 1
    right: int = 0
    best: int | None = None
    rounds: int = 0

    def __post_init__(self) -> None:
        self.right = len(self.sentences)

    @property
    def done(self) -> bool:
        return self.left > self.right

    @property
    def mid(self) -> int:
        return (self.left + self.right) // 2

    def prompt(self, template: str = JUDGE_TEMPLATE) -> str:
        return template.format(question=self.question, reasoning_part=" ".join(self.sentences[:self.mid]))

    def update(self, reply: str) -> None:
        midpoint = self.mid
        if "[ENOUGH]" in reply or reply.strip() == "ENOUGH":
            self.best, self.right = midpoint, midpoint - 1
        else:
            self.left = midpoint + 1
        self.rounds += 1

    def result(self) -> tuple[str, int, bool]:
        count = len(self.sentences) if self.best is None else self.best
        return " ".join(self.sentences[:count]), count, self.best is not None


def run_searches(searches: list[PrefixSearch], judge: Callable[[list[str]], list[str]],
                 batch_size: int = 512, max_rounds: int = 20, template: str = JUDGE_TEMPLATE) -> None:
    """Advance open searches in lockstep, issuing one batch per round when it fits.

    Source: SGL/src/sgl/palign/truncate.py:94-107.
    """
    for _ in range(max_rounds):
        active = [s for s in searches if not s.done]
        if not active:
            break
        for start in range(0, len(active), batch_size):
            group = active[start:start + batch_size]
            replies = judge([search.prompt(template) for search in group])
            if len(replies) != len(group):
                raise ValueError("judge returned a different number of replies")
            for search, reply in zip(group, replies):
                search.update(reply)


def _chat_ids(tokenizer: Any, prompt: str, enable_thinking: bool | None = False) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            encoded = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            encoded = tokenizer.apply_chat_template(messages, **kwargs)
        return list(encoded["input_ids"] if isinstance(encoded, dict) else encoded)
    return list(tokenizer(prompt, add_special_tokens=False)["input_ids"])


@STAGES.register("palign_search", source="SGL/src/sgl/palign/truncate.py:40-162")
class PAlignSearch(Stage):
    """Find the shortest useful CoT prefix with a lockstep student judge.

    Source: SGL palign/truncate.py:40-162; P-ALIGN/src/binary_search.py:46-154.
    """

    supports_shards = True
    outputs: ClassVar[dict[str, str]] = {"records": "text"}

    class Config(BaseConfig):
        model: str = ""
        judge_template: str = JUDGE_TEMPLATE
        question_field: str = "question"
        cot_field: str = "deepseek_thinking_trajectory"
        batch_size: int = 512
        max_new_tokens: int = 256
        engine: str = "vllm"
        max_rounds: int = 20

    def _judge(self, prompts: list[str], model: Any, tokenizer: Any) -> list[str]:
        ids = [_chat_ids(tokenizer, prompt) for prompt in prompts]
        if self.cfg.engine == "vllm":
            params = vllm_api.sampling_params(max_tokens=self.cfg.max_new_tokens, temperature=0.0)
            outputs = model.generate([vllm_api.tokens_prompt(p) for p in ids], params, use_tqdm=False)
            return [out.outputs[0].text for out in outputs]
        if self.cfg.engine != "hf":
            raise ValueError(f"unknown judge engine {self.cfg.engine}")
        device = next(model.parameters()).device
        width = max(map(len, ids))
        padded = torch.full((len(ids), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(padded)
        for index, prompt in enumerate(ids):
            padded[index, -len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
            mask[index, -len(prompt):] = 1
        with torch.no_grad():
            generated = model.generate(padded, attention_mask=mask, max_new_tokens=self.cfg.max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id)
        return [tokenizer.decode(row[width:], skip_special_tokens=True) for row in generated]

    def run(self, ctx: StageContext) -> dict:
        source = _input_rows(ctx, "data", "prev")
        path, rows = _pending(ctx, "records.jsonl", source)
        self._resolved_model = _model_path(self.cfg.model, ctx)
        with ResumableWriter(path) as writer:
            pending = [row for row in rows if not writer.done(str(row["meta.id"]))]
            if pending:
                tokenizer = load_tokenizer(ModelSpec(path=self._resolved_model))
                segmenter = SEGMENTERS.build("palign_sentences")
                searches = [PrefixSearch(str(row[self.cfg.question_field]),
                                         segmenter(str(row[self.cfg.cot_field]))) for row in pending]
                model = (make_llm(self._resolved_model, enable_prefix_caching=True, enforce_eager=True)
                         if self.cfg.engine == "vllm" else load_model(ModelSpec(path=self._resolved_model),
                                                                        trainable=False))
                if self.cfg.engine == "hf" and torch.cuda.is_available():
                    model = model.to("cuda")
                run_searches(searches, lambda p: self._judge(p, model, tokenizer),
                             self.cfg.batch_size, self.cfg.max_rounds, self.cfg.judge_template)
                for row, search in zip(pending, searches):
                    prefix, n_prefix, sufficient = search.result()
                    writer.write({"meta.id": str(row["meta.id"]), "question": search.question,
                                  "answer": row.get("answer", row.get("meta.answer", "")),
                                  "prefix": prefix, "n_prefix": n_prefix,
                                  "n_sentences": len(search.sentences), "sufficient": sufficient,
                                  "rounds": search.rounds})
                del model
                gc.collect()
        return {"n": len(rows)}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = _merge_jsonl(ctx, shard_dirs, "records.jsonl", _input_rows(ctx, "data", "prev"))
        return {"n": len(rows)}


@STAGES.register("answer_gain", source="SGL/src/sgl/signals/answer_gain.py:33-90")
class AnswerGain(Stage):
    """Measure each step's change in final-answer log probability.

    gain_k = log p(answer|prefix_k) − log p(answer|prefix_{k−1}); source: SGL answer_gain.py:33-90.
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

    def _score_hf(self, seqs: list[list[int]], starts: list[int], model: Any,
                  adapter: HFAutoAdapter) -> list[float]:
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
        return [sum(dist[pos][seq[pos]] for pos in range(start, len(seq)))
                for seq, start, dist in zip(seqs, starts, distributions)]

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
        source = _input_rows(ctx, "data", "prev")
        path, rows = _pending(ctx, "signals.jsonl", source)
        self._resolved_model = _model_path(self.cfg.model, ctx)
        tokenizer = load_tokenizer(ModelSpec(path=self._resolved_model))
        if self.cfg.engine == "hf":
            self._score_model, _, self._adapter = _load_frozen(self._resolved_model, "fp32")
        elif self.cfg.engine == "vllm":
            self._score_model = make_llm(self._resolved_model, max_model_len=self.cfg.max_model_len,
                                         gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                                         enable_prefix_caching=True, enforce_eager=True)
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
        _write_signals_parquet(path)
        return {"n": len(rows) - skipped, "skipped": skipped, "signals": ["step_gain"]}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = _merge_jsonl(ctx, shard_dirs, "signals.jsonl", _input_rows(ctx, "data", "prev"))
        assert ctx.output is not None
        _write_signals_parquet(ctx.output.path / "signals.jsonl")
        return {"n": len(rows), "signals": ["step_gain"]}


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


def _ig_example(tokenizer: Any, question: str, answer: str,
                segments: list[str], user_template: str, answer_template: str
                ) -> tuple[list[int], list[tuple[int, int]], tuple[int, int]]:
    user_ids = _chat_ids(tokenizer, user_template.format(input=question), enable_thinking=None)
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
    return (user_ids + response_ids + list(answer_encoded["input_ids"]), spans,
            (offset + matching[0], offset + matching[-1] + 1))


def _integrated_gradients(model: Any, adapter: HFAutoAdapter, ids: list[int],
                          answer_span: tuple[int, int], pad_id: int,
                          steps: int, batch_size: int) -> torch.Tensor:
    device = next(model.parameters()).device
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
        alpha = alphas[first:first + batch_size, None, None]
        path = (baseline + alpha * (source - baseline).detach()).requires_grad_(True)
        hidden = adapter.forward_hidden(model, tokens.expand(len(alpha), -1), None,
                                        inputs_embeds=path).last_hidden_state
        logits = head(hidden[:, start - 1:end - 1])
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
        model, tokenizer, adapter = _load_frozen(self._resolved_model, self.cfg.dtype)
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
        ids, spans, answer_span = _ig_example(tokenizer, str(question), str(answer), segments,
                                               self.cfg.user_template, self.cfg.answer_template)
        valid_spans = [(a, b) for a, b in spans if b > a]
        if len(ids) > self.cfg.max_input_tokens:
            return [[b - a, 0.0, 0.0] for a, b in valid_spans], True
        try:
            score = _integrated_gradients(model, adapter, ids, answer_span, int(tokenizer.pad_token_id),
                                          self.cfg.steps, self.cfg.batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return [[b - a, 0.0, 0.0] for a, b in valid_spans], True
        return compact_ig([score[a:b].cpu().tolist() for a, b in valid_spans]), False

    def run(self, ctx: StageContext) -> dict:
        source = _input_rows(ctx, "data", "prev")
        path, rows = _pending(ctx, "signals.jsonl", source)
        self._resolved_model = _model_path(self.cfg.model, ctx)
        model, tokenizer, adapter = self._prepare_model()
        oom = 0
        with ResumableWriter(path) as writer:
            for row in rows:
                sid = str(row["meta.id"])
                if writer.done(sid):
                    continue
                scores, failed = self._one(row, model, tokenizer, adapter)
                writer.write({"meta.id": sid, "segment_ig": scores})
                oom += failed
        _write_signals_parquet(path)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"n": len(rows), "oom": oom, "signals": ["segment_ig"]}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = _merge_jsonl(ctx, shard_dirs, "signals.jsonl", _input_rows(ctx, "data", "prev"))
        assert ctx.output is not None
        _write_signals_parquet(ctx.output.path / "signals.jsonl")
        return {"n": len(rows), "signals": ["segment_ig"]}


def _signal_aliases(ctx: StageContext) -> dict[str, dict[str, dict]]:
    aliases = {}
    for alias, art in ctx.inputs.items():
        if alias.startswith("signals"):
            aliases[alias] = {str(row["meta.id"]): row for row in read_jsonl(art.path / "signals.jsonl")}
    return aliases


def _join_signals(rows: list[dict], aliases: dict[str, dict[str, dict]],
                  mapping: dict[str, str], missing: str) -> tuple[list[dict], int]:
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


def _sample_selection_stats(row: dict) -> dict[str, float]:
    spans = row.get("seq.steps", [])
    selected = row.get("meta.selected_steps", [])
    total_steps = len(spans)
    total_tokens = sum(end - start for start, end in spans)
    kept_tokens = sum(spans[i][1] - spans[i][0] for i in selected)
    step_retention = len(selected) / total_steps if total_steps else 0.0
    token_retention = kept_tokens / total_tokens if total_tokens else 0.0
    mask, weights = row.get("seq.loss_mask"), row.get("seq.token_weights")
    mass_ratio = (sum(float(m) * float(w) for m, w in zip(mask, weights)) / sum(mask)
                  if mask is not None and weights is not None and sum(mask) else 1.0)
    return {"n_steps": float(total_steps), "n_steps_kept": float(len(selected)),
            "n_steps_dropped": float(total_steps - len(selected)),
            "step_retention": step_retention, "step_drop": 1.0 - step_retention if total_steps else 0.0,
            "n_tokens": float(total_tokens), "n_tokens_kept": float(kept_tokens),
            "n_tokens_dropped": float(total_tokens - kept_tokens),
            "token_retention": token_retention, "token_drop": 1.0 - token_retention if total_tokens else 0.0,
            "keeps_final_step": float(total_steps - 1 in selected) if total_steps else 0.0,
            "weight_mass_ratio_mean": mass_ratio}


def _selection_stats(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    stats = [_sample_selection_stats(row) for row in rows]
    return {key: sum(item[key] for item in stats) / len(stats) for key in stats[0]}


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


@STAGES.register("build_weights", source="SGL/src/sgl/allocation/build.py:144-355")
class BuildWeights(Stage):
    """Join offline signals by id and run selection/allocation transforms.

    Source: SGL allocation/build.py:144-355 and selection/build_masks.py:125-194.
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

    def run(self, ctx: StageContext) -> dict:
        source = _input_rows(ctx, "data")
        aliases = _signal_aliases(ctx)
        joined, missing_dropped = _join_signals(source, aliases, self.cfg.signals_map, self.cfg.missing)
        # Fit corpus transforms on all rows even for a shard; then emit only its assigned ids.
        transformed = self._transform(joined, ctx.rt)
        path, rows = _pending(ctx, "samples.jsonl", transformed)
        with ResumableWriter(path) as writer:
            for row in rows:
                if not writer.done(str(row["meta.id"])):
                    writer.write(row)
        assert ctx.output is not None
        save_samples(ctx.output.path, read_jsonl(path))
        stats = _selection_stats(transformed)
        if (self.cfg.check_mass and any(c.get("type") == "allocate_iwc" and
                                        c.get("variant", "stable") == "stable" for c in self.cfg.chain)
                and abs(stats.get("weight_mass_ratio_mean", 1.0) - 1.0) > 1e-5):
            raise ValueError("stable IWC did not preserve selected token mass")
        return {"n": len(rows), "dropped": missing_dropped + len(joined) - len(transformed),
                "selection_stats": stats}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = _merge_jsonl(ctx, shard_dirs, "samples.jsonl", _input_rows(ctx, "data"))
        assert ctx.output is not None
        save_samples(ctx.output.path, rows)
        return {"n": len(rows), "selection_stats": _selection_stats(rows)}

    def load_samples(self, art: Artifact) -> list[dict]:
        return load_samples_dir(art.path)
