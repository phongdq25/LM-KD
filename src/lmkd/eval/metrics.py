"""Metric helpers (pure python)."""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k = 1 - C(n-c, k) / C(n, k) (Chen et al. 2021)."""
    if k > n:
        raise ValueError("k > n")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def mean(xs: Any) -> float:
    """Arithmetic mean of a sequence; returns nan for empty sequences."""
    items = list(xs)
    return sum(items) / len(items) if items else float("nan")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for binomial distribution."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _compute_pass_at_k_metrics(rows: list[dict], ks: list[int], summary: dict[str, float]) -> None:
    for k in ks:
        if k == 1:
            continue
        vals = []
        for r in rows:
            gens = r.get("generations", [])
            labels = r.get("labels", [])
            n_i, c_i = len(gens), sum(labels)
            if k <= n_i:
                vals.append(pass_at_k(n_i, c_i, k))
            else:
                logger.warning("Problem has fewer generations (%d) than k=%d; skipping for this problem.", n_i, k)
        if vals:
            summary[f"pass@{k}"] = sum(vals) / len(vals)


def aggregate_passk(rows: list[dict], ks: list[int] | None = None) -> dict[str, float]:
    """Aggregate per-problem generation records into benchmark metrics.

    Pass@1 is the mean accuracy over all generations across all problems (fixing the SSFT
    single-column bias bug). Pass@k is the unbiased estimator averaged over problems.
    any_correct is the fraction of problems with >= 1 correct sample.
    """
    ks = ks or [1]
    total = sum(len(r.get("generations", [])) for r in rows)
    correct = sum(sum(r.get("labels", [])) for r in rows)
    solved = sum(1 for r in rows if sum(r.get("labels", [])) > 0)

    truncated = sum(sum(1 for g in r.get("generations", []) if g.get("finish_reason") == "length") for r in rows)
    unboxed = sum(
        sum(1 for g in r.get("generations", []) if len(g.get("text", "")) > 0 and "\\boxed" not in g.get("text", ""))
        for r in rows
    )
    token_sum = sum(sum(g.get("n_tokens", 0) for g in r.get("generations", [])) for r in rows)

    summary: dict[str, float] = {
        "pass@1": correct / total if total else 0.0,
        "any_correct": solved / len(rows) if rows else 0.0,
        "truncation_rate": truncated / total if total else 0.0,
        "no_boxed_rate": unboxed / total if total else 0.0,
        "mean_len": token_sum / total if total else 0.0,
    }
    _compute_pass_at_k_metrics(rows, ks, summary)
    return summary


def score_generations(rows: list[dict], grader: Any, ks: list[int] | None = None) -> dict[str, float]:
    """Grade each generation in rows using grader, assign row['labels'], and aggregate pass@k metrics."""
    for row in rows:
        texts = [g["text"] for g in row.get("generations", [])]
        gold = row.get("gold", row.get("answer", ""))
        if hasattr(grader, "grade_many"):
            labels = grader.grade_many(texts, gold)
        else:
            labels = [int(round(grader(t, gold))) for t in texts]
        row["labels"] = labels
    return aggregate_passk(rows, ks=ks)
