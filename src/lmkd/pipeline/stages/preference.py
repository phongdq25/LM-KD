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
    "distinguish, Output \"None\"."
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


def compress_target_logits(view: LogitsView, labels: torch.Tensor, k: int
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-k probabilities and residual at targets; source: distillation_losses.py:20-42."""
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
        mask = valid[:, chunk.start:chunk.end]
        ids[:, chunk.start:chunk.end, :count] = top_id.masked_fill(~mask[..., None], 0)
        values[:, chunk.start:chunk.end, :count] = top_p * mask[..., None]
        tail[:, chunk.start:chunk.end] = (1 - top_p.sum(-1)).clamp_min(0) * mask
    return ids, values, tail


def sparse_advantages(teacher: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      labels: torch.Tensor, max_k: int
                      ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sparse log(p_teacher/p_ref) over top-k union; source: _preference_math.py:9-40."""
    ids = torch.full((*labels.shape, max_k), -1, dtype=torch.long, device=labels.device)
    values = torch.zeros((*labels.shape, max_k), dtype=torch.float32, device=labels.device)
    truncated = 0
    for row, target in labels.ne(-100).nonzero().tolist():
        if target == 0:
            continue
        t_map = dict(zip(teacher[0][row, target].tolist(), teacher[1][row, target].tolist()))
        r_map = dict(zip(reference[0][row, target].tolist(), reference[1][row, target].tolist()))
        union = sorted(set(t_map) | set(r_map))
        margins = [math.log(max(t_map.get(i, 1e-8), 1e-8)) -
                   math.log(max(r_map.get(i, 1e-8), 1e-8)) for i in union]
        order = sorted(range(len(union)), key=margins.__getitem__, reverse=True)
        truncated += max(0, len(order) - max_k)
        for j, index in enumerate(order[:max_k]):
            ids[row, target, j] = union[index]
            values[row, target, j] = margins[index]
    return ids, values, truncated


def _empty_cache(row: dict, kind: str, k: int) -> None:
    n = len(row["student.input_ids"])
    if kind == "topk":
        row.update({"cache.topk_ids": [[0] * k for _ in range(n)],
                    "cache.topk_p": [[0.0] * k for _ in range(n)], "cache.tail_p": [0.0] * n})
    elif kind == "adv_sparse":
        row.update({"cache.adv_ids": [[-1] * k for _ in range(n)],
                    "cache.adv_val": [[0.0] * k for _ in range(n)]})
    elif kind == "token_logp":
        row["cache.token_logp"] = [0.0] * n
    elif kind == "seq_logp":
        row["cache.seq_logp"] = 0.0
    elif kind == "ctpd":
        row.update({"pref.ctpd_groups": [], "pref.ctpd_w": []})
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


def _cache_batch(rows: list[dict], model: Any, reference: Any | None, kind: str,
                 k: int, adv_k: int, pad_id: int) -> int:
    view, labels = _forward_view(model, rows, pad_id)
    truncated = 0
    if kind == "topk":
        ids, probs, tail = compress_target_logits(view, labels, k)
    elif kind == "adv_sparse":
        assert reference is not None
        ref_view, _ = _forward_view(reference, rows, pad_id)
        ids, probs, truncated = sparse_advantages(
            compress_target_logits(view, labels, k), compress_target_logits(ref_view, labels, k), labels, adv_k,
        )
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
        elif kind == "token_logp":
            row["cache.token_logp"] = token[i, :n].cpu().tolist()
        else:
            row["cache.seq_logp"] = float(seq[i].cpu())
    return truncated


@STAGES.register("teacher_cache", source="integrations/distillation_data.py:77-99,587-631")
class TeacherCache(Stage):
    """Cache batched teacher top-k, sampled log-probs, or sparse ADPA advantages.

    Source: human_alignment/integrations/distillation_data.py:77-99,587-631.
    """

    inputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    supports_shards = True

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        return {"topk": ("cache.topk_ids", "cache.topk_p", "cache.tail_p"),
                "adv_sparse": ("cache.adv_ids", "cache.adv_val"),
                "token_logp": ("cache.token_logp",), "seq_logp": ("cache.seq_logp",),
                "ctpd": ("pref.ctpd_groups", "pref.ctpd_w")}[cfg.kind]

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
            truncated += _cache_batch(selected[start:start + self.cfg.batch_size], model, reference,
                                      self.cfg.kind, self.cfg.k, self.cfg.adv_k_max, pad_id)
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
                ref = (load_model(ModelSpec(path=self.cfg.reference_model, dtype=self.cfg.dtype), trainable=False)
                       if self.cfg.reference_model else None)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                if ref is not None:
                    ref.to(device)
                tok = (load_tokenizer(ModelSpec(path=self.cfg.tokenizer)) if self.cfg.tokenizer
                       else getattr(ctx.rt, "student_tokenizer", None))
                if tok is None:
                    raise ValueError("teacher_cache needs the student tokenizer or tokenizer path")
                student_vocab = len(getattr(ctx.rt, "student_tokenizer", tok))
                teacher_vocab = int(cast(Any, model).config.vocab_size)
                if teacher_vocab != student_vocab:
                    logger.warning("teacher vocab %d differs from student vocab %d; out-of-range top-k IDs remain",
                                   teacher_vocab, student_vocab)
                pad_id = tok.pad_token_id or tok.eos_token_id or 0
                truncated = 0
                for start in range(0, len(pending), self.cfg.batch_size):
                    group = pending[start:start + self.cfg.batch_size]
                    truncated += self._process(group, model, ref, pad_id)
                    for sample in group:
                        writer.write(sample)
                if truncated:
                    logger.warning("truncated %d sparse advantage entries above adv_k_max=%d",
                                   truncated, self.cfg.adv_k_max)
        return {"n": len(samples), "computed": len(pending), "kind": self.cfg.kind,
                "tokenize_fingerprint": _dataset_tokenize_fp(samples)}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        return _merge(ctx, shard_dirs, "samples.jsonl")

    def load_samples(self, art: Artifact) -> list[dict]:
        return read_jsonl(art.path / "samples.jsonl")


def _last_logits(model: Any, tokenizer: Any, texts: list[str], batch_size: int,
                 *, add_special_tokens: bool = True) -> torch.Tensor:
    out = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], padding=True, return_tensors="pt",
                        add_special_tokens=add_special_tokens)
        device = _model_device(model)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        index = mask.sum(-1) - 1
        out.append(logits[torch.arange(len(index), device=device), index].cpu())
    return torch.cat(out)


def pairwise_scores(model: Any, tokenizer: Any, prompt: str, responses: list[str],
                    batch_size: int = 8) -> list[float]:
    """Order-balanced A/B scores; source: distillation_data.py:125-185."""
    if len(responses) < 2:
        raise ValueError("pairwise scoring needs at least two responses")
    option_ids = [tokenizer(x, add_special_tokens=False)["input_ids"][-1] for x in (" A", " B")]
    if option_ids[0] == option_ids[1]:
        raise ValueError("tokenizer maps pairwise options A and B to the same token")
    queries, pairs = [], []
    for left, right in combinations(range(len(responses)), 2):
        for first, second in ((left, right), (right, left)):
            queries.append("Choose the response that better follows the instruction. Reply with A or B.\n\n"
                           f"Instruction:\n{prompt}\n\nA:\n{responses[first]}\n\n"
                           f"B:\n{responses[second]}\n\nAnswer:")
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
    """Render the verbatim PAD MCQ template; source: distillation_data.py:188-224."""
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": _PAD_MCQ_SYSTEM}, {"role": "user", "content": _PAD_MCQ_USER}],
            tokenize=False, add_generation_prompt=True,
        )
    except TemplateError:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{_PAD_MCQ_SYSTEM}\n\n{_PAD_MCQ_USER}"}],
            tokenize=False, add_generation_prompt=True,
        )
    return Template(text)


def _pad_truncate(text: str, tokenizer: Any) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:2500]) + "..." if len(ids) > 2500 else text


def pad_mcq_scores(model: Any, tokenizer: Any, instruction: str, responses: list[str],
                   batch_size: int = 8) -> list[float]:
    """PAD top-20 option probabilities; source: distillation_data.py:188-278."""
    if len(responses) < 2:
        raise ValueError("PAD MCQ scoring needs at least two responses")
    template = pad_mcq_template(tokenizer)
    candidates = [_pad_truncate(r, tokenizer) for r in responses]
    pairs, queries = [], []
    for left, right in combinations(range(len(candidates)), 2):
        options = "\n".join(f'{letter}) """\n{candidates[index]}\n"""'
                            for letter, index in (("A", left), ("B", right)))
        query = template.safe_substitute(identifier_list_str="(A, or B)", instruction=instruction,
                                         candidate_response_list_str=options)
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
        return str(tok.apply_chat_template([{"role": "user", "content": str(prompt)}],
                                           tokenize=False, add_generation_prompt=True))
    return str(prompt)


def _instruction_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        turns = [message for message in prompt if message.get("role") == "user"]
        if not turns:
            raise ValueError("PAD MCQ needs a user turn")
        return str(turns[-1]["content"])
    return str(prompt)


def _sequence_scores(model: Any, tok: Any, prompt: str, responses: list[str], batch_size: int,
                     max_length: int, max_prompt_length: int) -> list[float]:
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
        view, labels = _forward_view(model, rows[start:start + batch_size], pad_id)
        seq, _, _ = sequence_logps(view, labels, average=True)
        scores.extend(seq.cpu().tolist())
    return scores


def _reward_scores(model: Any, tok: Any, prompt: str, responses: list[str], batch_size: int) -> list[float]:
    if getattr(tok, "chat_template", None):
        texts = [tok.apply_chat_template([{"role": "user", "content": prompt},
                                          {"role": "assistant", "content": response}], tokenize=False)
                 for response in responses]
    else:
        texts = [prompt + response for response in responses]
    scores = []
    for start in range(0, len(texts), batch_size):
        enc = tok(texts[start:start + batch_size], padding=True, truncation=True,
                  max_length=4096, return_tensors="pt", add_special_tokens=not bool(tok.chat_template))
        device = _model_device(model)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
        if logits.shape[-1] != 1:
            raise ValueError("reward model needs one output logit")
        scores.extend(logits[:, 0].float().cpu().tolist())
    return scores


def score_record(row: dict, model: Any, tok: Any, mode: str, batch_size: int,
                 out_field: str, max_length: int = 1024, max_prompt_length: int = 512) -> dict:
    """Score every response in one raw or listwise record."""
    if "group.members" in row:
        prompt = row["group.members"][0]["meta.prompt_text"]
        responses = [member["meta.response_text"] for member in row["group.members"]]
    else:
        prompt, responses = _normalize(row, tok, True)
    raw_prompt = row.get("prompt", row["group.members"][0].get("meta.raw_prompt", prompt)
                         if "group.members" in row else prompt)
    teacher_prompt = _teacher_prompt(tok, raw_prompt)
    if mode == "sequence_logprob":
        result = {out_field: _sequence_scores(model, tok, teacher_prompt, responses, batch_size,
                                              max_length, max_prompt_length)}
    elif mode == "pairwise":
        result = {out_field: pairwise_scores(model, tok, prompt, responses, batch_size)}
    elif mode == "pad_mcq":
        instruction = _instruction_text(raw_prompt)
        result = {"scores_mcq": pad_mcq_scores(model, tok, instruction, responses, batch_size),
                  "scores_avglogp": _sequence_scores(model, tok, teacher_prompt, responses, batch_size,
                                                      max_length, max_prompt_length)}
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

    Source: human_alignment/integrations/distillation_data.py:102-278 and reward_model.py:27-138.
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
                    model = load_model(ModelSpec(path=self.cfg.model, dtype="fp32"), trainable=False)
                model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
                for row in pending:
                    writer.write(score_record(row, model, tok, self.cfg.mode, self.cfg.batch_size,
                                              self.cfg.out_field, self.cfg.max_length,
                                              self.cfg.max_prompt_length))
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
