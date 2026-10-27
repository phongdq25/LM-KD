"""Answer graders for generation and reasoning benchmarks (T07).

Graders score a generated response against a reference answer (0/1 or float).
All implementations follow their original baseline implementations in DistiLLM / SGL / P-ALIGN.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import os
import re
import signal
import string
import threading
from collections.abc import Sequence
from typing import Any

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import GRADERS
from lmkd.eval.base import Grader

logger = logging.getLogger(__name__)

_BOXED = re.compile(r"\\boxed\s*{")
_MCQ_PATTERNS = [
    re.compile(r"\\boxed\s*{\s*\(?([A-D])\)?\s*}"),
    re.compile(r"(?:answer|Answer|option|Option|choice|Choice)\s*(?:is)?\s*[:\-]?\s*\(?([A-D])\)?\b"),
    re.compile(r"\b([A-D])\s*[\.]?\s*$"),
]


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace.

    Source: white-box-distillation/same-tokenizer/rouge_metric.py:12-26 and Squad v1.1.
    """
    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Exact match comparison after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn: Any, prediction: str, ground_truths: Sequence[str]) -> float:
    """Compute maximum score of prediction against any ground truth reference."""
    if not ground_truths:
        return 0.0
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores)


def rouge_metrics(
    predictions: Sequence[str],
    references: Sequence[Any],
    xlingual: bool = False,
) -> dict[str, float]:
    """Pure function: ROUGE-L and exact match (x100, 4 decimal places).

    Source: white-box-distillation/same-tokenizer/rouge_metric.py:46-66.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError("rouge_score is required for ROUGE evaluation. Install with `pip install rouge-score`.") from e

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    min_length = min(len(predictions), len(references))
    preds = predictions[:min_length]
    refs = references[:min_length]

    em_total, rouge_total = 0.0, 0.0
    for pred, gold in zip(preds, refs):
        golds: list[str] = [str(g) for g in gold] if isinstance(gold, (list, tuple)) else [str(gold)]
        em_total += metric_max_over_ground_truths(exact_match, pred, golds)

        def _rouge_score(p: str, g: str) -> float:
            return scorer.score(prediction=p, target=g)["rougeL"].fmeasure

        rouge_total += metric_max_over_ground_truths(_rouge_score, pred, golds)

    n = len(refs)
    if n == 0:
        return {"exact_match": 0.0, "rougeL": 0.0}
    em_score = round(100.0 * em_total / n, 4)
    rouge_score_val = round(100.0 * rouge_total / n, 4)
    return {"exact_match": em_score, "rougeL": rouge_score_val}


@GRADERS.register("exact_match_norm", source="white-box-distillation/same-tokenizer/rouge_metric.py:28-29")
class ExactMatchNormGrader(Grader):
    """1.0 if normalized prediction equals normalized reference."""

    class Config(BaseConfig):
        pass

    def __call__(self, prediction: str, reference: Any) -> float:
        return float(normalize_answer(str(prediction)) == normalize_answer(str(reference)))


@GRADERS.register("rouge_l", source="white-box-distillation/same-tokenizer/rouge_metric.py:32-35", extras=("eval-gen",))
class RougeLGrader(Grader):
    """ROUGE-L F1 score in [0, 1] using stemmed rouge-score."""

    class Config(BaseConfig):
        pass

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        from rouge_score import rouge_scorer

        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def __call__(self, prediction: str, reference: Any) -> float:
        if not hasattr(self, "scorer"):
            from rouge_score import rouge_scorer

            self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        gold = reference[0] if isinstance(reference, list) else str(reference)
        res = self.scorer.score(prediction=str(prediction), target=gold)
        return float(res["rougeL"].fmeasure)


def extract_boxed(text: str) -> str | None:
    """Extract content of the last \\boxed{...}, properly matching nested braces.

    Source: Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/eval/graders/answer_scoring.py:17-31.
    """
    matches = list(_BOXED.finditer(text))
    if not matches:
        return None

    start = matches[-1].end()
    depth, index = 1, start
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    return text[start : index - 1].strip() if depth == 0 else text[start:].strip()


def normalize_math(answer: str) -> str:
    """Lowercase and strip LaTeX formatting noise that never changes mathematical meaning.

    Source: Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/eval/graders/answer_scoring.py:34-42.
    """
    answer = answer.strip().rstrip(".").replace(" ", "")
    answer = re.sub(r"\\(?:left|right|text|mathrm|displaystyle)", "", answer)
    answer = re.sub(r"\\[!,;:]", "", answer)
    answer = re.sub(r"\\%|%|\\\$|\$", "", answer)
    answer = re.sub(r"^\{(.*)\}$", r"\1", answer)
    answer = re.sub(r"\.0+$", "", answer)
    return answer.lower()


def math_answers_equal(predicted: str | None, gold: str) -> bool:
    """Symbolic comparison via math-verify when available, normalized-string fallback otherwise.

    Source: Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/eval/graders/answer_scoring.py:45-60.
    """
    if predicted is None:
        return False
    if normalize_math(predicted) == normalize_math(gold):
        return True

    try:
        from math_verify import parse, verify
    except ImportError:
        return False

    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${predicted}$")))
    except Exception:
        return False


def extract_choice(text: str) -> str | None:
    """Extract first multiple-choice option (A-D), searching tail first.

    Source: Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/eval/graders/answer_scoring.py:63-70.
    """
    tail = text[-500:] if len(text) > 500 else text
    for pattern in _MCQ_PATTERNS:
        match = pattern.search(tail) or pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


@GRADERS.register("boxed_builtin", source="Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/eval/graders/answer_scoring.py:17-80")
class BoxedBuiltinGrader(Grader):
    """Answer extraction and equivalence checking with brace matching and math normalization."""

    class Config(BaseConfig):
        task_type: str = "math"

    def __call__(self, prediction: str, reference: Any) -> float:
        ref_str = str(reference)
        if self.cfg.task_type == "mcq":
            choice = extract_choice(str(prediction))
            return float(choice == ref_str.strip().upper())
        boxed = extract_boxed(str(prediction))
        return float(math_answers_equal(boxed, ref_str))


@contextlib.contextmanager
def _time_limit(seconds: float):
    """Abort hanging verification. POSIX SIGALRM in main thread, no-op fallback."""
    if os.name != "posix" or threading.current_thread() != threading.main_thread():
        yield
        return

    def handler(signum: int, frame: Any) -> None:
        raise TimeoutError("verification timed out")

    prev = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(math_ceil(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def math_ceil(x: float) -> int:
    import math
    return max(1, math.ceil(x))


@GRADERS.register("math_verify", source="Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/eval/graders/palign.py:17-44,68-73", extras=("eval-math",))
class MathVerifyGrader(Grader):
    """Math answer equivalence via the ``math_verify`` library (P-ALIGN / SGL)."""

    class Config(BaseConfig):
        timeout_s: float = 10.0

    def _verify_one(self, parse: Any, verify: Any, parsed_gold: Any, pred: str) -> int:
        try:
            return int(bool(verify(parsed_gold, parse(pred))))
        except Exception:
            return 0

    def grade_many(self, predictions: list[str], reference: Any) -> list[int]:
        try:
            from math_verify import parse, verify
        except ImportError as e:
            raise ImportError(
                "math_verify grading needs the math-verify package installed. Run `pip install lmkd[eval-math]`."
            ) from e

        gold_str = str(reference)
        try:
            parsed_gold = parse(f"${gold_str}$")
        except Exception:
            return [0] * len(predictions)

        # Thread check for timeout strategy
        if os.name == "posix" and threading.current_thread() == threading.main_thread():
            try:
                with _time_limit(self.cfg.timeout_s):
                    return [self._verify_one(parse, verify, parsed_gold, p) for p in predictions]
            except Exception:
                return [0] * len(predictions)

        # Off-main-thread fallback: ThreadPoolExecutor
        def _task():
            return [self._verify_one(parse, verify, parsed_gold, p) for p in predictions]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_task)
            try:
                return fut.result(timeout=self.cfg.timeout_s)
            except Exception:
                return [0] * len(predictions)

    def __call__(self, prediction: str, reference: Any) -> float:
        return float(self.grade_many([str(prediction)], reference)[0])


@GRADERS.register("oat", source="Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/eval/graders/oat.py")
class OatGrader(Grader):
    """Math answer grader ported from Garena Online / OAT math grader."""

    class Config(BaseConfig):
        fast: bool = False

    def __call__(self, prediction: str, reference: Any) -> float:
        from lmkd.eval._vendor import oat_grader

        try:
            fn = getattr(oat_grader, "boxed_reward_fn", None)
            if fn is None:
                return 0.0
            _, score = fn(str(prediction), str(reference), fast=self.cfg.fast)
            return float(score == 1.0)
        except Exception:
            return 0.0


@GRADERS.register("palign", source="Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/eval/graders/palign.py:75")
class PAlignGrader(Grader):
    """P-ALIGN math grader: accepts prediction if either math_verify OR oat accepts it."""

    class Config(BaseConfig):
        timeout_s: float = 10.0

    def __init__(self, cfg: Any = None) -> None:
        super().__init__(cfg)
        self._math_verify = MathVerifyGrader(MathVerifyGrader.Config(timeout_s=self.cfg.timeout_s))
        self._oat = OatGrader(OatGrader.Config())

    def grade_many(self, predictions: list[str], reference: Any) -> list[int]:
        try:
            mv_labels = self._math_verify.grade_many(predictions, reference)
        except ImportError:
            mv_labels = [0] * len(predictions)
        if hasattr(self._oat, "grade_many"):
            oat_labels = self._oat.grade_many(predictions, reference)
        else:
            oat_labels = [int(round(self._oat(p, reference))) for p in predictions]
        return [int(m or o) for m, o in zip(mv_labels, oat_labels)]

    def __call__(self, prediction: str, reference: Any) -> float:
        return float(self.grade_many([str(prediction)], reference)[0])


@GRADERS.register("mcq", source="Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/eval/graders/answer_scoring.py:63-72")
class MCQGrader(Grader):
    """Multiple choice answer grader: extracts option letter (A-D) and checks equality."""

    class Config(BaseConfig):
        pass

    def __call__(self, prediction: str, reference: Any) -> float:
        choice = extract_choice(str(prediction))
        gold = str(reference).strip().upper()
        return float(choice == gold)
