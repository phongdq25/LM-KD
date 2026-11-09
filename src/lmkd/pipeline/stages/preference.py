"""Offline preference teacher caches and response scoring."""

from __future__ import annotations

import hashlib
import logging
import math
from itertools import combinations
from pathlib import Path
from string import Template
from typing import Any, ClassVar, cast

import torch
from jinja2.exceptions import TemplateError
from transformers import AutoModelForSequenceClassification

from lmkd.core.config import ModelSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import STAGES
from lmkd.data.transforms.listwise import _normalize, _tokenize_response, tokenize_fingerprint
from lmkd.losses.functional.sequence import sequence_logps
from lmkd.models.loading import load_model, load_tokenizer
from lmkd.models.logits_view import LogitsView
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import ResumableWriter, read_jsonl, shard_of, write_jsonl
from lmkd.pipeline.stage import Stage, StageContext

logger = logging.getLogger(__name__)
_PAD_MCQ_SYSTEM = (
    "You are a helpful assistant, that evaluates the following responses to the given query "
    "and choose the best one based on relevance, accuracy, completeness, coherence, and "
    "clarity.\nOutput only the identifier of the best response $identifier_list_str. If none "
    "of the responses meet these criteria, or if the responses are too similar to "
    'distinguish, Output "None".'
)
_PAD_MCQ_USER = (
    "### Query\n$instruction\n\n### Candidate Responses\n$candidate_response_list_str\n\n"
    "### Output\nThe identifier of the best response:"
)


def _input_rows(ctx: StageContext) -> list[dict]:
    art = ctx.inputs.get("data", ctx.inputs.get("prev"))
    if art is None:
        raise ValueError("preference stage needs a data input artifact")
    for name in ("samples.jsonl", "records.jsonl"):
        path = art.path / name
        if path.exists():
            return read_jsonl(path)
    raise FileNotFoundError(f"no samples.jsonl or records.jsonl in {art.path}")


def _merge(ctx: StageContext, shards: list[Path], name: str) -> dict:
    by_id = {str(row["meta.id"]): row for folder in shards for row in read_jsonl(folder / name)}
    ordered = [by_id[str(row["meta.id"])] for row in _input_rows(ctx)]
    assert ctx.output is not None
    write_jsonl(ctx.output.path / name, ordered)
    return {"n": len(ordered)}


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _forward_view(model: Any, rows: list[dict], pad_id: int) -> tuple[LogitsView, torch.Tensor]:
    device = _model_device(model)
    length = max(len(row["student.input_ids"]) for row in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        n = len(row["student.input_ids"])
        ids[i, :n] = torch.tensor(row["student.input_ids"], device=device)
        mask[i, :n] = 1
        labels[i, :n] = torch.tensor(row["student.labels"], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    return LogitsView(logits=logits), labels


def compress_target_logits(
    view: LogitsView, labels: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-k probabilities and residual at targets."""
    if k < 1:
        raise ValueError("top-k must be positive")
    shape = (*labels.shape, k)
    ids = torch.zeros(shape, dtype=torch.long, device=labels.device)
    values = torch.zeros(shape, dtype=torch.float32, device=labels.device)
    tail = torch.zeros_like(labels, dtype=torch.float32)
    valid = labels.ne(-100).clone()
    valid[:, 0] = False
    for chunk in view.chunks():
        probs = chunk.logits.float().softmax(-1)
        count = min(k, probs.shape[-1])
        top_p, top_id = probs.topk(count, dim=-1)
        mask = valid[:, chunk.start : chunk.end]
        ids[:, chunk.start : chunk.end, :count] = top_id.masked_fill(~mask[..., None], 0)
        values[:, chunk.start : chunk.end, :count] = top_p * mask[..., None]
        tail[:, chunk.start : chunk.end] = (1 - top_p.sum(-1)).clamp_min(0) * mask
    return ids, values, tail


def sparse_advantages(
    teacher: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    max_k: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sparse log(p_teacher/p_ref) over top-k union."""
    ids = torch.full((*labels.shape, max_k), -1, dtype=torch.long, device=labels.device)
    values = torch.zeros((*labels.shape, max_k), dtype=torch.float32, device=labels.device)
    truncated = 0
    for row, target in labels.ne(-100).nonzero().tolist():
        if target == 0:
            continue
        t_map = dict(zip(teacher[0][row, target].tolist(), teacher[1][row, target].tolist()))
        r_map = dict(zip(reference[0][row, target].tolist(), reference[1][row, target].tolist()))
        union = sorted(set(t_map) | set(r_map))
        margins = [math.log(max(t_map.get(i, 1e-8), 1e-8)) - math.log(max(r_map.get(i, 1e-8), 1e-8)) for i in union]
        order = sorted(range(len(union)), key=margins.__getitem__, reverse=True)
        truncated += max(0, len(order) - max_k)
        for j, index in enumerate(order[:max_k]):
            ids[row, target, j] = union[index]
            values[row, target, j] = margins[index]
    return ids, values, truncated


def _empty_cache(row: dict, kind: str, k: int) -> None:
    n = len(row["student.input_ids"])
    if kind == "topk":
        row.update(
            {
                "cache.topk_ids": [[0] * k for _ in range(n)],
                "cache.topk_p": [[0.0] * k for _ in range(n)],
                "cache.tail_p": [0.0] * n,
            }
        )
    elif kind == "adv_sparse":
        row.update({"cache.adv_ids": [[-1] * k for _ in range(n)], "cache.adv_val": [[0.0] * k for _ in range(n)]})
    elif kind == "token_logp":
        row["cache.token_logp"] = [0.0] * n
    elif kind == "seq_logp":
        row["cache.seq_logp"] = 0.0
    elif kind == "ctpd":
        row.update({"pref.ctpd_groups": [[-1]], "pref.ctpd_w": [0.0]})
    else:
        raise ValueError(f"unknown teacher_cache kind {kind!r}")
    row["meta.cache_tokenize_fp"] = tokenize_fingerprint(row["student.input_ids"], row["student.labels"])


def _dataset_tokenize_fp(samples: list[dict]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        for row in sample["group.members"]:
            digest.update(str(row["meta.id"]).encode())
            digest.update(tokenize_fingerprint(row["student.input_ids"], row["student.labels"]).encode())
    return digest.hexdigest()[:16]


def _cache_batch(
    rows: list[dict], model: Any, reference: Any | None, kind: str, k: int, adv_k: int, pad_id: int
) -> int:
    view, labels = _forward_view(model, rows, pad_id)
    truncated = 0
    if kind == "topk":
        ids, probs, tail = compress_target_logits(view, labels, k)
    elif kind == "adv_sparse":
        assert reference is not None
        ref_view, _ = _forward_view(reference, rows, pad_id)
        ids, probs, truncated = sparse_advantages(
            compress_target_logits(view, labels, k),
            compress_target_logits(ref_view, labels, k),
            labels,
            adv_k,
        )
    elif kind == "ctpd":
        assert reference is not None
        ref_view, _ = _forward_view(reference, rows, pad_id)
        _, t_tok, _ = sequence_logps(view, labels, average=False)
        _, r_tok, _ = sequence_logps(ref_view, labels, average=False)
    else:
        seq, token, _ = sequence_logps(view, labels, average=False)
    for i, row in enumerate(rows):
        n = len(row["student.input_ids"])
        if kind == "topk":
            row["cache.topk_ids"] = ids[i, :n].cpu().tolist()
            row["cache.topk_p"] = probs[i, :n].cpu().tolist()
            row["cache.tail_p"] = tail[i, :n].cpu().tolist()
        elif kind == "adv_sparse":
            row["cache.adv_ids"] = ids[i, :n].cpu().tolist()
            row["cache.adv_val"] = probs[i, :n].cpu().tolist()
        elif kind == "ctpd":
            pos = [t for t, lbl in enumerate(row["student.labels"][:n]) if t > 0 and lbl != -100]
            if not pos:
                row["pref.ctpd_groups"] = [[-1]]
                row["pref.ctpd_w"] = [0.0]
            else:
                positive = int(row.get("group.index", i % 2)) == 0
                sign = 1.0 if positive else -1.0
                deltas = (t_tok[i, pos] - r_tok[i, pos]).float() * sign
                weights = torch.round(torch.exp(deltas.clamp(-0.5, 1.5)) * 100.0) / 100.0
                if weights.numel() > 0:
                    weights[-1] = 0.0
                row["pref.ctpd_groups"] = [[int(t)] for t in pos]
                row["pref.ctpd_w"] = weights.cpu().tolist()
        elif kind == "token_logp":
            row["cache.token_logp"] = token[i, :n].cpu().tolist()
        else:
            row["cache.seq_logp"] = float(seq[i].cpu())
    return truncated


@STAGES.register("teacher_cache")
class TeacherCache(Stage):
    """Cache batched teacher top-k, sampled log-probs, or sparse ADPA advantages.
    """

    inputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    supports_shards = True

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        return {
            "topk": ("cache.topk_ids", "cache.topk_p", "cache.tail_p"),
            "adv_sparse": ("cache.adv_ids", "cache.adv_val"),
            "token_logp": ("cache.token_logp",),
            "seq_logp": ("cache.seq_logp",),
            "ctpd": ("pref.ctpd_groups", "pref.ctpd_w"),
        }[cfg.kind]

    class Config(BaseConfig):
        model: str
        tokenizer: str | None = None
        kind: str = "topk"
        k: int = 50
        reference_model: str | None = None
        adv_k_max: int = 100
        fields: list[int] = [0, 1]
        batch_size: int = 8
        dtype: str = "bf16"

    def _process(self, samples: list[dict], model: Any, reference: Any | None, pad_id: int) -> int:
        selected = []
        width = self.cfg.adv_k_max if self.cfg.kind == "adv_sparse" else self.cfg.k
        for sample in samples:
            for i, row in enumerate(sample["group.members"]):
                _empty_cache(row, self.cfg.kind, width)
                if i in self.cfg.fields:
                    selected.append(row)
        truncated = 0
        for start in range(0, len(selected), self.cfg.batch_size):
            truncated += _cache_batch(
                selected[start : start + self.cfg.batch_size],
                model,
                reference,
                self.cfg.kind,
                self.cfg.k,
                self.cfg.adv_k_max,
                pad_id,
            )
        return truncated

    def run(self, ctx: StageContext) -> dict:
        if self.cfg.batch_size < 1 or self.cfg.k < 1 or self.cfg.adv_k_max < 1:
            raise ValueError("batch_size, k, and adv_k_max must be positive")
        if self.cfg.kind in ("adv_sparse", "ctpd") and not self.cfg.reference_model:
            raise ValueError(f"{self.cfg.kind} needs reference_model")
        assert ctx.output is not None
        path = ctx.output.path / "samples.jsonl"
        samples = shard_of(_input_rows(ctx), *ctx.shard)
        with ResumableWriter(path) as writer:
            pending = [sample for sample in samples if not writer.done(str(sample["meta.id"]))]
            if pending:
                model = load_model(ModelSpec(path=self.cfg.model, dtype=self.cfg.dtype), trainable=False)
                ref = (
                    load_model(ModelSpec(path=self.cfg.reference_model, dtype=self.cfg.dtype), trainable=False)
                    if self.cfg.reference_model
                    else None
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                if ref is not None:
                    ref.to(device)
                tok = (
                    load_tokenizer(ModelSpec(path=self.cfg.tokenizer))
                    if self.cfg.tokenizer
                    else getattr(ctx.rt, "student_tokenizer", None)
                )
                if tok is None:
                    raise ValueError("teacher_cache needs the student tokenizer or tokenizer path")
                student_vocab = len(getattr(ctx.rt, "student_tokenizer", tok))
                teacher_vocab = int(cast(Any, model).config.vocab_size)
                if teacher_vocab != student_vocab:
                    logger.warning(
                        "teacher vocab %d differs from student vocab %d; out-of-range top-k IDs remain",
                        teacher_vocab,
                        student_vocab,
                    )
                pad_id = tok.pad_token_id or tok.eos_token_id or 0
                truncated = 0
                for start in range(0, len(pending), self.cfg.batch_size):
                    group = pending[start : start + self.cfg.batch_size]
                    truncated += self._process(group, model, ref, pad_id)
                    for sample in group:
                        writer.write(sample)
                if truncated:
                    logger.warning(
                        "truncated %d sparse advantage entries above adv_k_max=%d", truncated, self.cfg.adv_k_max
                    )
        return {
            "n": len(samples),
            "computed": len(pending),
            "kind": self.cfg.kind,
            "tokenize_fingerprint": _dataset_tokenize_fp(samples),
        }

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        return _merge(ctx, shard_dirs, "samples.jsonl")

    def load_samples(self, art: Artifact) -> list[dict]:
        return read_jsonl(art.path / "samples.jsonl")


def _last_logits(
    model: Any, tokenizer: Any, texts: list[str], batch_size: int, *, add_special_tokens: bool = True
) -> torch.Tensor:
    out = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[start : start + batch_size], padding=True, return_tensors="pt", add_special_tokens=add_special_tokens
        )
        device = _model_device(model)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        index = mask.sum(-1) - 1
        out.append(logits[torch.arange(len(index), device=device), index].cpu())
    return torch.cat(out)


def pairwise_scores(model: Any, tokenizer: Any, prompt: str, responses: list[str], batch_size: int = 8) -> list[float]:
    """Order-balanced A/B scores."""
    if len(responses) < 2:
        raise ValueError("pairwise scoring needs at least two responses")
    option_ids = [tokenizer(x, add_special_tokens=False)["input_ids"][-1] for x in (" A", " B")]
    if option_ids[0] == option_ids[1]:
        raise ValueError("tokenizer maps pairwise options A and B to the same token")
    queries, pairs = [], []
    for left, right in combinations(range(len(responses)), 2):
        for first, second in ((left, right), (right, left)):
            queries.append(
                "Choose the response that better follows the instruction. Reply with A or B.\n\n"
                f"Instruction:\n{prompt}\n\nA:\n{responses[first]}\n\n"
                f"B:\n{responses[second]}\n\nAnswer:"
            )
            pairs.append((first, second))
    logits = _last_logits(model, tokenizer, queries, batch_size)
    probs = logits[:, option_ids].softmax(-1).tolist()
    totals, counts = [0.0] * len(responses), [0] * len(responses)
    for (first, second), (a, b) in zip(pairs, probs):
        totals[first] += a
        totals[second] += b
        counts[first] += 1
        counts[second] += 1
    return [math.log(max(total / count, 1e-8)) for total, count in zip(totals, counts)]


def pad_mcq_template(tokenizer: Any) -> Template:
    """Render the PAD MCQ template."""
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": _PAD_MCQ_SYSTEM}, {"role": "user", "content": _PAD_MCQ_USER}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except TemplateError:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{_PAD_MCQ_SYSTEM}\n\n{_PAD_MCQ_USER}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return Template(text)


def _pad_truncate(text: str, tokenizer: Any) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:2500]) + "..." if len(ids) > 2500 else text


def pad_mcq_scores(
    model: Any, tokenizer: Any, instruction: str, responses: list[str], batch_size: int = 8
) -> list[float]:
    """PAD top-20 option probabilities."""
    if len(responses) < 2:
        raise ValueError("PAD MCQ scoring needs at least two responses")
    template = pad_mcq_template(tokenizer)
    candidates = [_pad_truncate(r, tokenizer) for r in responses]
    pairs, queries = [], []
    for left, right in combinations(range(len(candidates)), 2):
        options = "\n".join(f'{letter}) """\n{candidates[index]}\n"""' for letter, index in (("A", left), ("B", right)))
        query = template.safe_substitute(
            identifier_list_str="(A, or B)", instruction=instruction, candidate_response_list_str=options
        )
        bos = getattr(tokenizer, "bos_token", None)
        queries.append(bos + query if bos and not query.startswith(bos) else query)
        pairs.append((left, right))
    logits = _last_logits(model, tokenizer, queries, batch_size, add_special_tokens=False)
    totals = [1e-8] * len(candidates)
    for (left, right), row in zip(pairs, logits):
        top_lp, top_ids = row.log_softmax(-1).topk(min(20, row.shape[-1]))
        selected = {"A": 1e-8, "B": 1e-8}
        for lp, token_id in zip(top_lp.tolist(), top_ids.tolist()):
            token = tokenizer.decode(int(token_id))
            for letter in selected:
                if letter in token and len(token) < 3:
                    selected[letter] += math.exp(lp)
        totals[left] += selected["A"]
        totals[right] += selected["B"]
    return [math.log(total / (len(candidates) - 1)) for total in totals]


def _teacher_prompt(tok: Any, prompt: Any) -> str:
    if isinstance(prompt, list):
        return str(tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
    if getattr(tok, "chat_template", None):
        return str(
            tok.apply_chat_template(
                [{"role": "user", "content": str(prompt)}], tokenize=False, add_generation_prompt=True
            )
        )
    return str(prompt)


def _instruction_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        turns = [message for message in prompt if message.get("role") == "user"]
        if not turns:
            raise ValueError("PAD MCQ needs a user turn")
        return str(turns[-1]["content"])
    return str(prompt)


def _sequence_scores(
    model: Any, tok: Any, prompt: str, responses: list[str], batch_size: int, max_length: int, max_prompt_length: int
) -> list[float]:
    rows = []
    for response in responses:
        result = _tokenize_response(tok, prompt, response, max_length, max_prompt_length)
        if result is None:
            raise ValueError("response became empty while scoring")
        ids, plen = result
        rows.append({"student.input_ids": ids, "student.labels": [-100] * plen + ids[plen:]})
    scores = []
    pad_id = tok.pad_token_id or tok.eos_token_id or 0
    for start in range(0, len(rows), batch_size):
        view, labels = _forward_view(model, rows[start : start + batch_size], pad_id)
        seq, _, _ = sequence_logps(view, labels, average=True)
        scores.extend(seq.cpu().tolist())
    return scores


def _reward_scores(model: Any, tok: Any, prompt: str, responses: list[str], batch_size: int) -> list[float]:
    if getattr(tok, "chat_template", None):
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], tokenize=False
            )
            for response in responses
        ]
    else:
        texts = [prompt + response for response in responses]
    scores = []
    for start in range(0, len(texts), batch_size):
        enc = tok(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
            add_special_tokens=not bool(tok.chat_template),
        )
        device = _model_device(model)
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)
            ).logits
        if logits.shape[-1] != 1:
            raise ValueError("reward model needs one output logit")
        scores.extend(logits[:, 0].float().cpu().tolist())
    return scores


def score_record(
    row: dict,
    model: Any,
    tok: Any,
    mode: str,
    batch_size: int,
    out_field: str,
    max_length: int = 1024,
    max_prompt_length: int = 512,
) -> dict:
    """Score every response in one raw or listwise record."""
    if "group.members" in row:
        prompt = row["group.members"][0]["meta.prompt_text"]
        responses = [member["meta.response_text"] for member in row["group.members"]]
    else:
        prompt, responses = _normalize(row, tok, True)
    raw_prompt = row.get(
        "prompt", row["group.members"][0].get("meta.raw_prompt", prompt) if "group.members" in row else prompt
    )
    teacher_prompt = _teacher_prompt(tok, raw_prompt)
    if mode == "sequence_logprob":
        result = {
            out_field: _sequence_scores(
                model, tok, teacher_prompt, responses, batch_size, max_length, max_prompt_length
            )
        }
    elif mode == "pairwise":
        result = {out_field: pairwise_scores(model, tok, prompt, responses, batch_size)}
    elif mode == "pad_mcq":
        instruction = _instruction_text(raw_prompt)
        result = {
            "scores_mcq": pad_mcq_scores(model, tok, instruction, responses, batch_size),
            "scores_avglogp": _sequence_scores(
                model, tok, teacher_prompt, responses, batch_size, max_length, max_prompt_length
            ),
        }
        if out_field:
            result[out_field] = result["scores_mcq"]
    elif mode == "reward_model":
        result = {"rewards": _reward_scores(model, tok, _instruction_text(raw_prompt), responses, batch_size)}
    else:
        raise ValueError(f"unknown score mode {mode!r}")
    if "group.members" in row:
        members = [dict(member) for member in row["group.members"]]
        for i, member in enumerate(members):
            if mode == "reward_model":
                member["pref.reward"] = result["rewards"][i]
            else:
                scores = result[out_field] if out_field in result else result["scores_mcq"]
                member["pref.teacher_score"] = scores[i]
        return {**row, **result, "group.members": members}
    return {**row, **result}


@STAGES.register("score", source="integrations/distillation_data.py:102-278")
class Score(Stage):
    """Score preference candidates by sequence, pairwise, PAD MCQ, or reward model.
    """

    supports_shards = True
    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        if cfg.mode == "reward_model":
            return ("pref.reward",)
        return ("pref.teacher_score",)

    class Config(BaseConfig):
        mode: str = "sequence_logprob"
        model: str
        out_field: str = "teacher_scores"
        batch_size: int = 8
        max_length: int = 1024
        max_prompt_length: int = 512

    def run(self, ctx: StageContext) -> dict:
        if self.cfg.batch_size < 1:
            raise ValueError("batch_size must be positive")
        assert ctx.output is not None
        rows = shard_of(_input_rows(ctx), *ctx.shard)
        name = "samples.jsonl" if rows and "group.members" in rows[0] else "records.jsonl"
        with ResumableWriter(ctx.output.path / name) as writer:
            pending = [row for row in rows if not writer.done(str(row["meta.id"]))]
            if pending:
                tok = load_tokenizer(ModelSpec(path=self.cfg.model))
                if self.cfg.mode == "reward_model":
                    model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model)
                    if model.config.pad_token_id is None:
                        model.config.pad_token_id = tok.pad_token_id
                else:
                    dtype = (
                        "bf16"
                        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                        else ("fp16" if torch.cuda.is_available() else "fp32")
                    )
                    model = load_model(ModelSpec(path=self.cfg.model, dtype=dtype), trainable=False)
                model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
                for row in pending:
                    writer.write(
                        score_record(
                            row,
                            model,
                            tok,
                            self.cfg.mode,
                            self.cfg.batch_size,
                            self.cfg.out_field,
                            self.cfg.max_length,
                            self.cfg.max_prompt_length,
                        )
                    )
        return {"n": len(rows), "computed": len(pending), "mode": self.cfg.mode}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = _input_rows(ctx)
        name = "samples.jsonl" if rows and "group.members" in rows[0] else "records.jsonl"
        return _merge(ctx, shard_dirs, name)

    def load_samples(self, art: Artifact) -> list[dict]:
        path = art.path / "samples.jsonl"
        if not path.exists():
            raise NotImplementedError
        return read_jsonl(path)


def _extract_prompt_text(row: dict) -> Any:
    """Extract raw prompt value from various dataset formats."""
    raw = row.get("prompt")
    if raw is None and "question" in row:
        raw = row["question"]
    if raw is None and "problem" in row:
        raw = row["problem"]
    if raw is None and "chosen" in row:
        chosen = row["chosen"]
        if isinstance(chosen, list) and len(chosen) > 1:
            raw = chosen[:-1]
        elif isinstance(chosen, str):
            raw = chosen
    return raw if raw is not None else ""


def sample_candidates_record(
    row: dict,
    model: Any,
    tok: Any,
    num_responses: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    prompt_field: str = "prompt",
    out_field: str = "responses",
) -> dict:
    raw_prompt = row.get(prompt_field) or _extract_prompt_text(row)
    if isinstance(raw_prompt, list):
        if getattr(tok, "chat_template", None):
            prompt = str(tok.apply_chat_template(raw_prompt, tokenize=False, add_generation_prompt=True))
        else:
            prompt = str(raw_prompt[-1].get("content", "")) if raw_prompt else ""
    elif getattr(tok, "chat_template", None):
        prompt = str(
            tok.apply_chat_template(
                [{"role": "user", "content": str(raw_prompt)}], tokenize=False, add_generation_prompt=True
            )
        )
    else:
        prompt = str(raw_prompt)

    device = _model_device(model)
    enc = tok(prompt, return_tensors="pt")
    if device:
        enc = {k: v.to(device) for k, v in enc.items()}

    pad_id = tok.pad_token_id or tok.eos_token_id or 0
    eos_id = tok.eos_token_id or pad_id
    with torch.no_grad():
        output = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_responses,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    plen = enc["input_ids"].shape[1]
    responses = [tok.decode(ids[plen:], skip_special_tokens=True) for ids in output]
    res = {**row, "prompt": raw_prompt, out_field: responses}
    if "meta.id" not in res:
        res["meta.id"] = str(row.get("id", id(row)))
    return res


@STAGES.register("sample_candidates")
class SampleCandidates(Stage):
    """Generate candidate responses using student/sampler model for preference distillation.
    """

    supports_shards = True
    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}

    class Config(BaseConfig):
        model: str
        num_responses: int = 4
        temperature: float = 1.0
        top_p: float = 0.95
        max_new_tokens: int = 512
        prompt_field: str = "prompt"
        out_field: str = "responses"

    def run(self, ctx: StageContext) -> dict:
        assert ctx.output is not None
        rows = shard_of(_input_rows(ctx), *ctx.shard)
        name = "records.jsonl"
        with ResumableWriter(ctx.output.path / name) as writer:
            pending = [row for row in rows if not writer.done(str(row.get("meta.id", row.get("id", ""))))]
            if pending:
                tok = load_tokenizer(ModelSpec(path=self.cfg.model))
                if tok.pad_token_id is None:
                    tok.pad_token_id = tok.eos_token_id
                dtype = (
                    "bf16"
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else ("fp16" if torch.cuda.is_available() else "fp32")
                )
                model = load_model(ModelSpec(path=self.cfg.model, dtype=dtype), trainable=False)
                model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
                for row in pending:
                    sampled = sample_candidates_record(
                        row,
                        model,
                        tok,
                        num_responses=self.cfg.num_responses,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        max_new_tokens=self.cfg.max_new_tokens,
                        prompt_field=self.cfg.prompt_field,
                        out_field=self.cfg.out_field,
                    )
                    writer.write(sampled)
        return {"n": len(rows), "computed": len(pending)}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        return _merge(ctx, shard_dirs, "records.jsonl")

    def load_samples(self, art: Artifact) -> list[dict]:
        path = art.path / "records.jsonl"
        if not path.exists():
            raise NotImplementedError
        return read_jsonl(path)


@STAGES.register("sppo_calibrate")
class SPPOCalibrate(Stage):
    """Calibrate preference candidates for SPPO and S-SPPO:
    Computes soft marginal win/lose probabilities P_win, P_lose, and optional semantic similarity sim_score.
    """

    supports_shards = True
    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}

    class Config(BaseConfig):
        sim_model: str | None = None  # e.g. "sentence-transformers/all-MiniLM-L6-v2"
        score_field: str = "teacher_scores"
        reward_field: str = "rewards"

    def run(self, ctx: StageContext) -> dict:
        assert ctx.output is not None
        rows = shard_of(_input_rows(ctx), *ctx.shard)
        name = "records.jsonl"
        with ResumableWriter(ctx.output.path / name) as writer:
            pending = [row for row in rows if not writer.done(str(row.get("meta.id", row.get("id", ""))))]
            if pending:
                encoder = None
                enc_tok = None
                if self.cfg.sim_model:
                    from transformers import AutoModel, AutoTokenizer

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    enc_tok = AutoTokenizer.from_pretrained(self.cfg.sim_model)
                    encoder = AutoModel.from_pretrained(self.cfg.sim_model).to(device).eval()

                for row in pending:
                    calibrated = self._calibrate_row(row, encoder, enc_tok)
                    writer.write(calibrated)
        return {"n": len(rows), "computed": len(pending)}

    def _calibrate_row(self, row: dict, encoder: Any, enc_tok: Any) -> dict:
        responses = row.get("responses", [])
        if not responses and "group.members" in row:
            responses = [m["meta.response_text"] for m in row["group.members"]]
        prompt = row.get("prompt", row.get("meta.raw_prompt", ""))
        if not prompt and "group.members" in row:
            prompt = row["group.members"][0].get(
                "meta.raw_prompt", row["group.members"][0].get("meta.prompt_text", "")
            )

        scores = row.get(self.cfg.score_field, row.get(self.cfg.reward_field))
        if scores is None and "group.members" in row:
            scores = [m.get("pref.teacher_score", m.get("pref.reward", 0.0)) for m in row["group.members"]]

        if not responses or scores is None or len(responses) != len(scores):
            return row

        scores = [float(s) for s in scores]
        k = len(scores)
        # Borda/Bradley-Terry win probabilities: P(y_i > y_j) = sigmoid(s_i - s_j)
        p_hat = []
        for i in range(k):
            p_i = sum(1.0 / (1.0 + math.exp(max(-50.0, min(50.0, scores[j] - scores[i])))) for j in range(k)) / max(1, k)
            p_hat.append(p_i)

        w = int(max(range(k), key=lambda idx: scores[idx]))
        l = int(min(range(k), key=lambda idx: scores[idx]))
        if w == l and k > 1:
            l = (w + 1) % k

        chosen = responses[w]
        rejected = responses[l]
        p_win = p_hat[w]
        p_lose = p_hat[l]
        p_chosen = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, scores[l] - scores[w]))))

        sim_score = 0.0
        if encoder is not None and enc_tok is not None:
            device = next(encoder.parameters()).device
            enc = enc_tok([chosen, rejected], padding=True, truncation=True, max_length=512, return_tensors="pt").to(
                device
            )
            with torch.no_grad():
                out = encoder(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                sum_emb = (out.last_hidden_state * mask).sum(dim=1)
                norm_emb = torch.nn.functional.normalize(sum_emb / mask.sum(dim=1).clamp_min(1e-9), p=2, dim=1)
                sim_score = float(torch.clamp((norm_emb[0:1] * norm_emb[1:2]).sum(), min=0.0).item())

        out_row = dict(row)
        out_row["prompt"] = prompt
        out_row["chosen"] = chosen
        out_row["rejected"] = rejected
        out_row["chosen_probs_win"] = p_win
        out_row["chosen_probs_lose"] = p_lose
        out_row["chosen_probs"] = p_chosen
        out_row["sim_score"] = sim_score
        return out_row

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        return _merge(ctx, shard_dirs, "records.jsonl")

    def load_samples(self, art: Artifact) -> list[dict]:
        path = art.path / "records.jsonl"
        if not path.exists():
            raise NotImplementedError
        return read_jsonl(path)


@STAGES.register("merge_proto")
class MergeProto(Stage):
    """Merge SFT baseline proto responses (y_0) into self-play pairs (y, y') for T-SPIN."""

    supports_shards = False
    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}

    class Config(BaseConfig):
        proto_field: str = "rejected"

    def run(self, ctx: StageContext) -> dict:
        assert ctx.output is not None
        gen_art = ctx.inputs.get("generated", ctx.inputs.get("data"))
        proto_art = ctx.inputs.get("proto")
        if gen_art is None or proto_art is None:
            raise ValueError("merge_proto stage requires 'generated' and 'proto' input artifacts")

        gen_path = gen_art.path / "records.jsonl" if (gen_art.path / "records.jsonl").exists() else gen_art.path / "samples.jsonl"
        proto_path = proto_art.path / "records.jsonl" if (proto_art.path / "records.jsonl").exists() else proto_art.path / "samples.jsonl"

        gen_rows = read_jsonl(gen_path)
        proto_rows = read_jsonl(proto_path)

        merged = []
        for g, p in zip(gen_rows, proto_rows):
            row = dict(g)
            row["proto"] = p.get(self.cfg.proto_field, p.get("proto", p.get("rejected", "")))
            merged.append(row)

        write_jsonl(ctx.output.path / "records.jsonl", merged)
        return {"n": len(merged)}

    def load_samples(self, art: Artifact) -> list[dict]:
        return read_jsonl(art.path / "records.jsonl")

