"""Unit tests for evaluation metrics and pure evaluation functions (T07)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lmkd.core.registry import EVALUATORS
from lmkd.data.transforms.tokenize import render_prompt
from lmkd.eval.generation import generate_for_records
from lmkd.eval.graders import BoxedBuiltinGrader, rouge_metrics
from lmkd.eval.metrics import aggregate_passk, score_generations
from lmkd.rollout.base import Rollout, SamplingSpec


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


def test_render_prompt_system_and_chat_template():
    tok = MagicMock()
    tok.chat_template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"

    def mock_apply_chat(messages, tokenize=False, add_generation_prompt=True, **kwargs):
        res = ""
        for m in messages:
            res += f"<{m['role']}>{m['content']}</{m['role']}>"
        if kwargs.get("enable_thinking") is False:
            res += "<no_think>"
        if add_generation_prompt:
            res += "<gen>"
        return res

    tok.apply_chat_template.side_effect = mock_apply_chat

    # 1. Plain string without chat template
    res = render_prompt(tok, "hello", use_chat_template=False)
    assert res == "hello"

    # 2. Plain string with system prompt, no chat template
    res = render_prompt(tok, "hello", use_chat_template=False, system_prompt="be concise")
    assert res == "be concise\n\nhello"

    # 3. Chat template with system prompt and chat_template_kwargs
    res = render_prompt(
        tok,
        "hello",
        use_chat_template=True,
        system_prompt="be concise",
        enable_thinking=False,
    )
    assert res == "<system>be concise</system><user>hello</user><no_think><gen>"

    # 4. Chat messages list with system prompt prepended
    msgs = [{"role": "user", "content": "hi"}]
    res = render_prompt(
        tok,
        msgs,
        use_chat_template=True,
        system_prompt="sys",
    )
    assert res == "<system>sys</system><user>hi</user><gen>"

    # 5. Chat messages list with existing system prompt (does not duplicate)
    msgs_with_sys = [{"role": "system", "content": "orig"}, {"role": "user", "content": "hi"}]
    res = render_prompt(
        tok,
        msgs_with_sys,
        use_chat_template=True,
        system_prompt="sys",
    )
    assert res == "<system>orig</system><user>hi</user><gen>"


def test_generate_for_records_chat_template():
    tok = MagicMock()
    tok.chat_template = "template"
    tok.bos_token_id = 1
    tok.apply_chat_template.side_effect = lambda msgs, **kwargs: (
        f"<sys>{msgs[0]['content']}</sys><user>{msgs[1]['content']}</user>"
        if len(msgs) > 1 and msgs[0]["role"] == "system"
        else f"<user>{msgs[0]['content']}</user>"
    ) + ("<no_think>" if kwargs.get("enable_thinking") is False else "")

    def fake_tokenize(text, add_special_tokens=False):
        return {"input_ids": [100, 101]}

    tok.side_effect = fake_tokenize

    rt = MagicMock()
    rt.student_tokenizer = tok

    engine = MagicMock()
    engine.generate.side_effect = lambda ids, spec: [
        [Rollout(prompt_ids=p, response_ids=[200], text="response")] for p in ids
    ]
    rt.rollout = engine

    records = [{"question": "What is 2+2?"}]
    sampling = SamplingSpec(max_new_tokens=10)

    rollouts = generate_for_records(
        rt,
        records,
        sampling,
        engine_spec=None,
        use_chat_template=True,
        chat_template_kwargs={"enable_thinking": False},
        system_prompt="Math expert",
    )

    assert len(rollouts) == 1
    assert rollouts[0][0].text == "response"
    assert engine.generate.called
    prompt_ids_passed = engine.generate.call_args[0][0]
    # bos_token_id (1) prepended
    assert prompt_ids_passed == [[1, 100, 101]]
    tok.apply_chat_template.assert_called_with(
        [{"role": "system", "content": "Math expert"}, {"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def test_lm_eval_evaluator_plugin(tmp_path):
    evaluator = EVALUATORS.build(
        {
            "type": "lm_eval",
            "name": "benchmarks",
            "tasks": ["hellaswag", "gsm8k"],
            "batch_size": 2,
        }
    )

    rt = MagicMock()
    rt.engine.is_main = True
    rt.engine.device = "cpu"
    rt.state.step = 10
    rt.config.train.output_dir = str(tmp_path)
    rt.unwrap_model.return_value = MagicMock()
    rt.student_tokenizer = MagicMock()

    mock_res = {
        "results": {
            "hellaswag": {
                "acc,none": 0.65,
                "acc_norm,none": 0.70,
                "acc_norm_stderr,none": 0.01,
            },
            "gsm8k": {
                "exact_match,strict-match": 0.50,
            },
        }
    }

    with (
        patch("lm_eval.models.huggingface.HFLM"),
        patch("lm_eval.simple_evaluate", return_value=mock_res),
    ):
        metrics = evaluator.evaluate(rt)

    assert metrics["benchmarks/hellaswag"] == 0.70
    assert metrics["benchmarks/gsm8k"] == 0.50
    assert metrics["benchmarks/avg"] == 0.60
    assert metrics["benchmarks/hellaswag/acc_norm"] == 0.70
    assert (tmp_path / "eval" / "benchmarks" / "step-10" / "lm_eval_summary.json").exists()


def test_lm_eval_evaluator_vllm_backend(tmp_path):
    evaluator = EVALUATORS.build(
        {
            "type": "lm_eval",
            "backend": "vllm",
            "tasks": ["hellaswag"],
            "batch_size": "auto",
        }
    )

    rt = MagicMock()
    rt.engine.is_main = True
    rt.engine.device = "cuda:0"
    rt.state.step = 5
    rt.config.train.output_dir = str(tmp_path)
    rt.config.models.student.path = str(tmp_path)
    rt.unwrap_model.return_value = MagicMock()
    rt.student_tokenizer = MagicMock()

    mock_res = {
        "results": {
            "hellaswag": {
                "acc_norm,none": 0.75,
            },
        }
    }

    with (
        patch("lm_eval.models.vllm_causallms.VLLM") as mock_vllm,
        patch("lm_eval.simple_evaluate", return_value=mock_res),
    ):
        metrics = evaluator.evaluate(rt)

    assert metrics["lm_eval/hellaswag"] == 0.75
    assert mock_vllm.called



