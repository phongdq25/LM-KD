"""Unit tests for graders (T07)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lmkd.eval.graders import (
    BoxedBuiltinGrader,
    ExactMatchNormGrader,
    MathVerifyGrader,
    MCQGrader,
    PAlignGrader,
    RougeLGrader,
    extract_boxed,
    normalize_answer,
    rouge_metrics,
)


def test_rouge_matches_original():
    pytest.importorskip("rouge_score")

    preds = [
        "the capital of france is paris",
        "machine learning is great",
        "python programming language",
        "deep neural networks",
        "distillation of language models",
    ]
    refs = [
        ["paris", "the capital is paris"],
        ["machine learning is very great"],
        ["python language"],
        ["deep neural networks and models"],
        ["distillation of large language models"],
    ]

    metrics = rouge_metrics(preds, refs)
    assert "exact_match" in metrics and "rougeL" in metrics
    assert metrics["rougeL"] > 0


def test_normalize_answer():
    assert normalize_answer("The  Cat!") == "cat"
    assert normalize_answer("An Apple, A Pear.") == "apple pear"
    assert normalize_answer("  hello   world  ") == "hello world"


def test_boxed_builtin():
    grader = BoxedBuiltinGrader(BoxedBuiltinGrader.Config())
    # \frac{1}{2} vs 1/2
    assert grader(r"The answer is \boxed{\frac{1}{2}}", "1/2") == 1.0
    assert grader(r"The answer is \boxed{\frac12}", "1/2") == 1.0
    # Nested braces
    assert extract_boxed(r"\boxed{a{b}}") == "a{b}"
    assert extract_boxed(r"No boxed here") is None
    assert extract_boxed(r"First \boxed{1} second \boxed{2}") == "2"


def test_math_verify_grader():
    pytest.importorskip("math_verify")
    grader = MathVerifyGrader(MathVerifyGrader.Config(timeout_s=5.0))
    assert grader(r"The answer is \boxed{2}", "2") == 1.0
    assert grader(r"The answer is \boxed{3}", "2") == 0.0


def test_palign_or():
    palign = PAlignGrader(PAlignGrader.Config())
    mock_mv = MagicMock()
    mock_oat = MagicMock()
    palign._math_verify = mock_mv
    palign._oat = mock_oat

    # True OR False -> 1
    mock_mv.grade_many.return_value = [1, 0, 0]
    mock_oat.grade_many.return_value = [0, 1, 0]
    labels = palign.grade_many(["p1", "p2", "p3"], "gold")
    assert labels == [1, 1, 0]


def test_mcq_grader():
    grader = MCQGrader(MCQGrader.Config())
    assert grader(r"The answer is \boxed{B}", "B") == 1.0
    assert grader("Therefore the correct option is (C).", "C") == 1.0
    assert grader("The answer is A", "B") == 0.0


def test_exact_match_norm_and_rouge_l_graders():
    pytest.importorskip("rouge_score")
    em_norm = ExactMatchNormGrader(ExactMatchNormGrader.Config())
    assert em_norm("The  Cat!", "cat") == 1.0
    assert em_norm("Dog", "Cat") == 0.0

    rouge_l = RougeLGrader(RougeLGrader.Config())
    score = rouge_l("hello world", "hello beautiful world")
    assert 0.0 < score <= 1.0
