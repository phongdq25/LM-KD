"""Standard reasoning and math evaluation benchmarks (T07).

Ported from SGL (SpectralGuidedLearning/src/sgl/eval/benchmarks.py:24-100).
Supported benchmark sets: aime24, aime25, math500, olympiadbench, amc12, amc23, minerva, gpqa.
"""
from __future__ import annotations

import os
import random
from collections.abc import Iterator
from typing import Any

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import SOURCES
from lmkd.data.sources import Source

MATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{{}}.{question}"
MCQ_PROMPT = (
    "{question}\n\n"
    "A) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
    "Please reason step by step, and put the letter of your final answer within \\boxed{{}}.\n\n"
)

BENCH_DATA_ROOT = os.environ.get("BENCH_DATA_ROOT")


def _dataset_path(repo_id: str, local_dir_name: str) -> str:
    return f"{BENCH_DATA_ROOT}/{local_dir_name}" if BENCH_DATA_ROOT else repo_id


def _math_records(rows: Any, question_key: str, answer_key: str) -> list[dict]:
    return [
        {
            "id": index,
            "question": row[question_key],
            "prompt": MATH_PROMPT.format(question=row[question_key]),
            "gold": str(row[answer_key]),
            "answer": str(row[answer_key]),
            "task_type": "math",
        }
        for index, row in enumerate(rows)
    ]


def load_aime24() -> list[dict]:
    """Maxwell-Jia/AIME_2024: plain integer gold in Answer."""
    from datasets import load_dataset

    rows = load_dataset(_dataset_path("Maxwell-Jia/AIME_2024", "AIME_2024"), split="train")
    return _math_records(rows, "Problem", "Answer")


def load_aime25() -> list[dict]:
    """yentinglin/aime_2025."""
    from datasets import load_dataset

    rows = load_dataset(_dataset_path("yentinglin/aime_2025", "aime_2025"), split="train")
    return _math_records(rows, "problem", "answer")


def load_math500() -> list[dict]:
    """HuggingFaceH4/MATH-500 test split."""
    from datasets import load_dataset

    rows = load_dataset(_dataset_path("HuggingFaceH4/MATH-500", "MATH-500"), split="test")
    return _math_records(rows, "problem", "answer")


def load_olympiadbench() -> list[dict]:
    """English open-ended text-only math competition subset."""
    from datasets import load_dataset

    rows = load_dataset(_dataset_path("Hothan/OlympiadBench", "OlympiadBench"), "OE_TO_maths_en_COMP", split="train")
    records = []
    for index, row in enumerate(rows):
        answers = row.get("final_answer") or []
        if not answers:
            continue
        gold = str(answers[0]).strip("$")
        records.append(
            {
                "id": index,
                "question": row["question"],
                "prompt": MATH_PROMPT.format(question=row["question"]),
                "gold": gold,
                "answer": gold,
                "task_type": "math",
            }
        )
    return records


def load_amc12() -> list[dict]:
    """AI-MO/aimo-validation-amc: 83 AMC 12 problems (2022-2023)."""
    from datasets import load_dataset

    rows = load_dataset(_dataset_path("AI-MO/aimo-validation-amc", "aimo-validation-amc"), split="train")
    return _math_records(rows, "problem", "answer")


def load_amc23() -> list[dict]:
    """math-ai/amc23: 40 problems (2023 AMC 12A/12B)."""
    from datasets import load_dataset

    rows = load_dataset(_dataset_path("math-ai/amc23", "amc23"), split="test")
    return _math_records(rows, "question", "answer")


def load_minerva() -> list[dict]:
    """math-ai/minervamath: 272 undergraduate quantitative-reasoning problems."""
    from datasets import load_dataset

    rows = load_dataset(_dataset_path("math-ai/minervamath", "minervamath"), split="test")
    return _math_records(rows, "question", "answer")


def load_gpqa_diamond() -> list[dict]:
    """Idavidrein/gpqa gpqa_diamond subset."""
    from datasets import load_dataset

    rows = load_dataset(_dataset_path("Idavidrein/gpqa", "gpqa"), "gpqa_diamond", split="train")
    records = []
    for index, row in enumerate(rows):
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        rng = random.Random(index)
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        shuffled = [options[pos] for pos in order]
        gold = "ABCD"[order.index(0)]
        records.append(
            {
                "id": index,
                "question": row["Question"],
                "prompt": MCQ_PROMPT.format(
                    question=row["Question"],
                    a=shuffled[0],
                    b=shuffled[1],
                    c=shuffled[2],
                    d=shuffled[3],
                ),
                "gold": gold,
                "answer": gold,
                "task_type": "mcq",
            }
        )
    return records


BENCHMARK_LOADERS = {
    "aime24": load_aime24,
    "aime25": load_aime25,
    "math500": load_math500,
    "olympiadbench": load_olympiadbench,
    "gpqa": load_gpqa_diamond,
    "gpqa_diamond": load_gpqa_diamond,
    "amc12": load_amc12,
    "amc23": load_amc23,
    "minerva": load_minerva,
}


@SOURCES.register("benchmark", source="Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/eval/benchmarks.py:24-100")
class BenchmarkSource(Source):
    """Reasoning and math evaluation benchmarks source."""

    class Config(BaseConfig):
        name: str
        limit: int | None = None

    def records(self) -> Iterator[dict]:
        name = self.cfg.name.lower()
        loader = BENCHMARK_LOADERS.get(name)
        if loader is None:
            supported = ", ".join(sorted(BENCHMARK_LOADERS.keys()))
            raise ValueError(f"Unknown benchmark {self.cfg.name!r}. Supported benchmarks: {supported}")
        try:
            records = loader()
        except Exception as e:
            raise RuntimeError(f"Failed to load benchmark {self.cfg.name!r} from datasets: {e}") from e

        for i, r in enumerate(records):
            if self.cfg.limit is not None and i >= self.cfg.limit:
                break
            rec = dict(r)
            rec.setdefault("meta.id", str(rec.get("id", i)))
            yield rec

