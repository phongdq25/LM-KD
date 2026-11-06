"""Unit tests for evaluation metrics and pure evaluation functions (T07)."""

from __future__ import annotations

from lmkd.eval.graders import BoxedBuiltinGrader, rouge_metrics
from lmkd.eval.metrics import aggregate_passk, score_generations


def test_aggregate_passk():
    # 2 problems, n=4
    # Problem 1: labels = [1, 0, 0, 0] (c=1, n=4)
    # Problem 2: labels = [1, 1, 1, 1] (c=4, n=4)
    # Total samples = 8, correct = 5 -> pass@1 = 5/8 = 0.625
    # any_correct: prob 1 has 1>0, prob 2 has 4>0 -> 2/2 = 1.0
    # pass@2:
    #   prob 1: pass_at_k(4, 1, 2) = 1 - (3/4)*(2/3) = 1 - 0.5 = 0.5
    #   prob 2: pass_at_k(4, 4, 2) = 1.0
    #   mean(0.5, 1.0) = 0.75
    rows = [
        {
            "id": 1,
            "gold": "1",
            "task_type": "math",
            "generations": [
                {"text": r"\boxed{1}", "n_tokens": 10},
                {"text": r"\boxed{0}", "n_tokens": 10},
                {"text": r"\boxed{0}", "n_tokens": 10},
                {"text": r"\boxed{0}", "n_tokens": 10},
            ],
            "labels": [1, 0, 0, 0],
        },
        {
            "id": 2,
            "gold": "2",
            "task_type": "math",
            "generations": [
                {"text": r"\boxed{2}", "n_tokens": 10},
                {"text": r"\boxed{2}", "n_tokens": 10},
                {"text": r"\boxed{2}", "n_tokens": 10},
                {"text": r"\boxed{2}", "n_tokens": 10},
            ],
            "labels": [1, 1, 1, 1],
        },
    ]
    summary = aggregate_passk(rows, ks=[1, 2])
    assert summary["pass@1"] == 5 / 8
    assert summary["any_correct"] == 1.0
    assert summary["pass@2"] == 0.75


def test_not_first_column_bug():
    # In SSFT evaluate.py, pass@1 was incorrectly computed as mean_score[0],
    # which only checks the first sample!
    # If the first sample is wrong [0, 1, 1, 1], SSFT gave 0, but true pass@1 is 0.75.
    rows = [
        {
            "id": 1,
            "gold": "42",
            "task_type": "math",
            "generations": [
                {"text": "wrong"},
                {"text": r"\boxed{42}"},
                {"text": r"\boxed{42}"},
                {"text": r"\boxed{42}"},
            ],
            "labels": [0, 1, 1, 1],
        }
    ]
    summary = aggregate_passk(rows, ks=[1])
    assert summary["pass@1"] == 0.75


def test_truncation_and_boxed_rates():
    rows = [
        {
            "id": 1,
            "gold": "10",
            "task_type": "math",
            "generations": [
                {"text": "long thinking...", "finish_reason": "length", "n_tokens": 50},
                {"text": r"answer is \boxed{10}", "finish_reason": "stop", "n_tokens": 20},
                {"text": "no boxed here", "finish_reason": "stop", "n_tokens": 15},
                {"text": r"\boxed{5}", "finish_reason": "stop", "n_tokens": 15},
            ],
            "labels": [0, 1, 0, 0],
        }
    ]
    summary = aggregate_passk(rows, ks=[1])
    # 1 of 4 is truncated
    assert summary["truncation_rate"] == 0.25
    # 2 of 4 have no boxed ("long thinking...", "no boxed here")
    assert summary["no_boxed_rate"] == 0.5
    # mean_len = (50 + 20 + 15 + 15) / 4 = 25.0
    assert summary["mean_len"] == 25.0


def test_pure_functions_no_runtime():
    # Both rouge_metrics and score_generations can be invoked directly as pure functions
    # without creating an Evaluator or Runtime
    preds = ["hello world", "test answer"]
    refs = [["hello world"], ["test answer"]]
    r_metrics = rouge_metrics(preds, refs)
    assert r_metrics["exact_match"] == 100.0
    assert r_metrics["rougeL"] == 100.0

    grader = BoxedBuiltinGrader(BoxedBuiltinGrader.Config())
    rows = [
        {
            "id": 1,
            "gold": "5",
            "task_type": "math",
            "generations": [
                {"text": r"\boxed{5}", "finish_reason": "stop", "n_tokens": 10},
                {"text": r"\boxed{4}", "finish_reason": "stop", "n_tokens": 10},
            ],
        }
    ]
    scored = score_generations(rows, grader=grader, ks=[1])
    assert scored["pass@1"] == 0.5
    assert scored["any_correct"] == 1.0
