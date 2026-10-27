"""Embedding evaluation metrics (TALAS: STS, Linear Probe, Pair Classification).

Pure functions ``sts_spearman / linear_probe / pair_classification(model, tokenizer, records, ...)``
work on any model (scripts, notebooks); the ``embedding_*`` Evaluator plugins only read records
from their Config, call them on ``rt.student_model`` and prefix the metric keys.

Source: embedding-distill/TALAS-M-main/eval.py
Paper: TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import EVALUATORS
from lmkd.eval.base import Evaluator


def _load_records(data_spec: Any) -> list[dict]:
    if isinstance(data_spec, list):
        return data_spec
    if isinstance(data_spec, dict):
        if "records" in data_spec:
            return data_spec["records"]
        if "type" in data_spec:
            from lmkd.core.registry import SOURCES

            source = SOURCES.build(data_spec)
            return list(source.records())
        if "path" in data_spec:
            p = Path(data_spec["path"])
            if not p.exists():
                return []
            if p.suffix == ".jsonl":
                from lmkd.pipeline.io import read_jsonl

                return read_jsonl(p)
            elif p.suffix == ".csv":
                with open(p, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    return list(reader)
    return []


def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    pooler: str = "cls_attention",
    *,
    adapter: Any = None,
    device: Any = None,
    batch_size: int = 128,
    max_len: int = 256,
) -> torch.Tensor:
    """[N, D] fp32 sentence embeddings on CPU (no grad). ``adapter`` = ModelAdapter used to run the
    backbone (``None`` = ``model(...)`` must return ``last_hidden_state``)."""
    from lmkd.core.registry import POOLERS

    device = device or next(model.parameters()).device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = POOLERS.build(pooler)
    use_amp = torch.cuda.is_available() and "cuda" in str(device)
    out_chunks = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i : i + batch_size], padding=True, truncation=True, max_length=max_len,
                            return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            if adapter is not None and hasattr(adapter, "forward_hidden"):
                out = adapter.forward_hidden(model, input_ids, attn_mask)
            else:
                out = model(input_ids=input_ids, attention_mask=attn_mask)
            out_chunks.append(pool(out.last_hidden_state, attn_mask).cpu().float())
    return torch.cat(out_chunks, dim=0) if out_chunks else torch.empty((0, 0), dtype=torch.float32)


def _field(r: dict, primary: str, secondary: str) -> Any:
    val = r.get(primary)
    return r.get(secondary) if val is None else val


def sts_spearman(model: Any, tokenizer: Any, records: list[dict], *, pooler: str = "cls_attention",
                 text1_field: str = "sentence1", text2_field: str = "sentence2", score_field: str = "score",
                 **encode_kw: Any) -> dict[str, float]:
    """STS: Spearman between gold scores and ``(cos + 1) * 2.5``. Source: TALAS-M eval.py:18-69."""
    from scipy.stats import spearmanr

    if not records:
        return {"spearman": 0.0}
    t1 = [str(_field(r, text1_field, "text1")) for r in records]
    t2 = [str(_field(r, text2_field, "text2")) for r in records]
    labels = [float(_field(r, score_field, "label")) for r in records]
    e1 = encode_texts(model, tokenizer, t1, pooler, **encode_kw)
    e2 = encode_texts(model, tokenizer, t2, pooler, **encode_kw)
    scores = ((F.cosine_similarity(e1, e2, dim=-1) + 1.0) * 2.5).tolist()
    corr, _ = spearmanr(scores, labels)
    return {"spearman": 0.0 if (corr is None or torch.isnan(torch.tensor(corr))) else float(corr)}


def linear_probe(model: Any, tokenizer: Any, train_records: list[dict], test_records: list[dict], *,
                 pooler: str = "cls_attention", text_field: str = "text", label_field: str = "label",
                 max_iter: int = 200, **encode_kw: Any) -> dict[str, float]:
    """Logistic regression on frozen embeddings -> macro-F1 / accuracy. Source: TALAS-M eval.py:71-137."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    if not train_records or not test_records:
        return {"macro_f1": 0.0, "accuracy": 0.0}

    def texts(rs: list[dict]) -> list[str]:
        return [str(r.get(text_field, r.get("sentence1", ""))) for r in rs]

    y_train = np.array([int(r[label_field]) for r in train_records])
    y_test = np.array([int(r[label_field]) for r in test_records])
    x_train = encode_texts(model, tokenizer, texts(train_records), pooler, **encode_kw).numpy()
    x_test = encode_texts(model, tokenizer, texts(test_records), pooler, **encode_kw).numpy()
    clf = LogisticRegression(random_state=42, max_iter=max_iter, verbose=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return {"macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_test, y_pred))}


def pair_classification(model: Any, tokenizer: Any, records: list[dict], *, pooler: str = "cls_attention",
                        text1_field: str = "sentence1", text2_field: str = "sentence2",
                        label_field: str = "label", thresholds: int = 200, **encode_kw: Any) -> dict[str, float]:
    """Pair classification: AP of ``(cos + 1) / 2`` + best-threshold accuracy / macro-F1.
    Source: TALAS-M eval.py:139-206."""
    import numpy as np
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score

    if not records:
        return {"ap": 0.0}
    t1 = [str(_field(r, text1_field, "text1")) for r in records]
    t2 = [str(_field(r, text2_field, "text2")) for r in records]
    labels = np.array([int(_field(r, label_field, "score")) for r in records])
    e1 = encode_texts(model, tokenizer, t1, pooler, **encode_kw)
    e2 = encode_texts(model, tokenizer, t2, pooler, **encode_kw)
    scores = ((F.cosine_similarity(e1, e2, dim=-1) + 1.0) / 2.0).numpy()
    ap = average_precision_score(labels, scores)
    best_acc, best_thr = 0.0, 0.0
    for thr in np.linspace(0, 1, thresholds):
        acc = accuracy_score(labels, (scores >= thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    macro_f1 = f1_score(labels, (scores >= best_thr).astype(int), average="macro", zero_division=0)
    return {"ap": float(ap), "best_threshold": float(best_thr), "accuracy": float(best_acc), "f1": float(macro_f1)}


def _encode_kw(rt: Any, cfg: Any) -> dict[str, Any]:
    return {"adapter": getattr(rt, "student_adapter", None),
                "device": getattr(getattr(rt, "engine", None), "device", None),
                "batch_size": cfg.batch_size, "max_len": cfg.max_len}


@EVALUATORS.register(
    "embedding_sts",
    source="embedding-distill/TALAS-M-main/eval.py:18-69",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
    extras=("eval-emb",),
)
class EmbeddingSTS(Evaluator):
    """Semantic Textual Similarity (STS) evaluation with Spearman correlation."""

    class Config(BaseConfig):
        data: dict
        name: str = "sts"
        pooler: str = "cls_attention"
        batch_size: int = 128
        max_len: int = 256
        text1_field: str = "sentence1"
        text2_field: str = "sentence2"
        score_field: str = "score"

    def evaluate(self, rt: Any) -> dict[str, float]:
        c = self.cfg
        m = sts_spearman(rt.student_model, rt.student_tokenizer, _load_records(c.data), pooler=c.pooler,
                         text1_field=c.text1_field, text2_field=c.text2_field, score_field=c.score_field,
                         **_encode_kw(rt, c))
        return {f"sts/{c.name}/{k}": v for k, v in m.items()}


@EVALUATORS.register(
    "embedding_linear_probe",
    source="embedding-distill/TALAS-M-main/eval.py:71-137",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
    extras=("eval-emb",),
)
class EmbeddingLinearProbe(Evaluator):
    """Linear probe classification on frozen sentence embeddings (macro-F1)."""

    class Config(BaseConfig):
        train: dict
        test: dict
        name: str = "probe"
        pooler: str = "cls_attention"
        batch_size: int = 128
        max_len: int = 256
        text_field: str = "text"
        label_field: str = "label"
        max_iter: int = 200

    def evaluate(self, rt: Any) -> dict[str, float]:
        c = self.cfg
        m = linear_probe(rt.student_model, rt.student_tokenizer, _load_records(c.train), _load_records(c.test),
                         pooler=c.pooler, text_field=c.text_field, label_field=c.label_field,
                         max_iter=c.max_iter, **_encode_kw(rt, c))
        return {f"cls/{c.name}/{k}": v for k, v in m.items()}


@EVALUATORS.register(
    "embedding_pair",
    source="embedding-distill/TALAS-M-main/eval.py:139-206",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
    extras=("eval-emb",),
)
class EmbeddingPair(Evaluator):
    """Pair classification evaluation with Average Precision (AP) and optimal threshold search."""

    class Config(BaseConfig):
        data: dict
        name: str = "pair"
        pooler: str = "cls_attention"
        batch_size: int = 128
        max_len: int = 256
        text1_field: str = "sentence1"
        text2_field: str = "sentence2"
        label_field: str = "label"
        thresholds: int = 200

    def evaluate(self, rt: Any) -> dict[str, float]:
        c = self.cfg
        m = pair_classification(rt.student_model, rt.student_tokenizer, _load_records(c.data), pooler=c.pooler,
                                text1_field=c.text1_field, text2_field=c.text2_field, label_field=c.label_field,
                                thresholds=c.thresholds, **_encode_kw(rt, c))
        return {f"pair/{c.name}/{k}": v for k, v in m.items()}
