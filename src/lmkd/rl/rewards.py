"""Pure batch verifiers shared by native rollouts and TRL reward functions."""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from math_verify import parse, verify
from pydantic import Field

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import REWARDS


def last_boxed(text: str) -> str | None:
    """Extract the last boxed expression using balanced braces, including nested fractions."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end]
    return None


def _alarm_handler(signum, frame):
    raise TimeoutError("math verification timeout")


def _math_equivalent(gold: str, pred: str, timeout_s: float) -> bool:
    if threading.current_thread() is not threading.main_thread():
        # math_verify uses signal alarms internally: run it in a subprocess main thread.
        script = ("import json,sys; from math_verify import parse,verify; "
                  "a,b=json.loads(sys.stdin.read()); print(int(verify(parse('$'+a+'$'),parse('$'+b+'$'))))")
        result = subprocess.run([sys.executable, "-I", "-c", script], input=json.dumps([gold, pred]),
                                text=True, capture_output=True, timeout=timeout_s, check=True)
        return result.stdout.strip() == "1"
    previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        # Its nested alarm decorators otherwise replace and cancel our total wall-clock deadline.
        return bool(verify(parse(f"${gold}$", parsing_timeout=0), parse(f"${pred}$", parsing_timeout=0), timeout_seconds=0))
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _result(score: float, feedback="", acc: float | None = None) -> dict:
    return {"score": float(score), "acc": float(score if acc is None else acc), "feedback": feedback}


class Verifier(Plugin):
    """One reward/accuracy/feedback dictionary per (prompt, completion, metadata) row."""
    class Config(BaseConfig):
        pass

    def __call__(self, prompts: list[str], responses: list[str], meta: list[dict]) -> list[dict]:
        raise NotImplementedError

    @staticmethod
    def validate_rows(prompts, responses, meta):
        """Require one prompt and metadata dictionary per response."""
        if len(prompts) != len(responses) or len(meta) != len(responses):
            raise ValueError("verifier requires equally sized prompts, responses and metadata")


@REWARDS.register("math_verify", source="Self-Distill/OPSD-main/grpo_train.py:87-114")
class MathVerify(Verifier):
    """Boxed answer correctness with mathematical equivalence and whitespace string fallback."""
    class Config(BaseConfig):
        format_weight: float = 0.
        answer_field: str = "answer"
        timeout_s: float = Field(default=10., gt=0)

    def __call__(self, prompts, responses, meta):
        self.validate_rows(prompts, responses, meta)
        results = []
        for text, row in zip(responses, meta):
            pred = last_boxed(text)
            gold = row.get(self.cfg.answer_field, row.get("meta." + self.cfg.answer_field))
            if gold is None:
                gold = last_boxed(str(row.get("solution", row.get("meta.solution", ""))))
            gold = "" if gold is None else str(gold)
            acc = 0.
            feedback = "Your answer is not in \\boxed{}." if pred is None else ""
            if pred is not None:
                try:
                    acc = float(_math_equivalent(gold, pred, self.cfg.timeout_s))
                except Exception:
                    # Symbols are case-sensitive; only backend failures use the source's whitespace fallback.
                    acc = float(bool(pred.strip()) and "".join(pred.split()) == "".join(gold.split()))
                if not acc:
                    feedback = "Your boxed answer is incorrect."
            if row.get("truncated", row.get("meta.truncated", False)):
                feedback = "Your response was truncated."
            results.append(_result(acc + self.cfg.format_weight * (pred is not None), feedback, acc))
        return results


@REWARDS.register("mcq", source="Self-Distill/SDPO-main/verl/utils/reward_score/feedback/mcq.py")
class MCQ(Verifier):
    """Sparse multiple-choice correctness from boxed letters or the final Answer: X."""
    class Config(BaseConfig):
        answer_field: str = "answer"

    def __call__(self, prompts, responses, meta):
        self.validate_rows(prompts, responses, meta)
        results = []
        for text, row in zip(responses, meta):
            boxed = last_boxed(text)
            matches = re.findall(r"\b([A-J])\b", boxed) if boxed else re.findall(r"Answer:\s*([A-J])\b", text)
            gold = row.get(self.cfg.answer_field, row.get("ground_truth", ""))
            results.append(_result(float(bool(matches) and matches[-1] == str(gold).strip())))
        return results


def _resource_limits(timeout_s):
    resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(timeout_s)),) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64,) * 2)


def _run_code(code, test, timeout_s, memory_mb):
    with tempfile.TemporaryDirectory(prefix="lmkd-code-") as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(code + ("\n" + test if isinstance(test, str) else ""))
        # Apply address-space restrictions AFTER exec: the trainer may have huge CUDA/vLLM mappings.
        # A tiny trusted bootstrap sets memory/filesystem/network restrictions before running the solution.
        bootstrap = Path(__file__).with_name("_sandbox.py")
        proc = subprocess.Popen([sys.executable, "-I", str(bootstrap), tmp, str(path), str(memory_mb)], cwd=tmp,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, preexec_fn=lambda: _resource_limits(timeout_s), start_new_session=True)
        try:
            out, err = proc.communicate(input="" if isinstance(test, str) else str(test["input"]), timeout=timeout_s)
            return proc.returncode, out, err
        finally:
            # Kill descendants too, including on timeout; their filesystem/network restrictions inherit.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)


@REWARDS.register("code_sandbox", source="Self-Distill/SDPO-main/verl/utils/reward_score/feedback/code.py:820-870")
class CodeSandbox(Verifier):
    """Sparse code test rewards with resource limits, denied network and writes confined to a temporary directory.

    Security: this is not a complete hostile-code sandbox; only run on trusted machines.
    Requires Linux Landlock and seccomp; unsupported hosts fail closed with feedback.
    """
    class Config(BaseConfig):
        timeout_s: float = Field(default=10., gt=0)
        memory_mb: int = Field(default=1024, ge=64)

    def __call__(self, prompts, responses, meta):
        self.validate_rows(prompts, responses, meta)
        return [self._score(text, row) for text, row in zip(responses, meta)]

    def _score(self, text, row):
        fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        code = fences[-1] if fences else text
        ground = row.get("ground_truth", row.get("meta.ground_truth", ""))
        try:
            tests = json.loads(ground) if isinstance(ground, str) else ground
        except (ValueError, TypeError):
            tests = ground
        if isinstance(tests, str):
            tests = [tests]
        if not isinstance(tests, list) or not tests:
            return _result(0., "Failed to parse ground truth test cases.")
        for test in tests:
            try:
                status, out, err = _run_code(code, test, self.cfg.timeout_s, self.cfg.memory_mb)
            except subprocess.TimeoutExpired:
                return _result(0., "Execution timeout.")
            except (OSError, subprocess.SubprocessError, KeyError) as exc:
                return _result(0., f"Execution failed: {exc}"[:2000])
            expected = "" if isinstance(test, str) else str(test["output"]).strip()
            if status != 0 or (not isinstance(test, str) and out.strip() != expected):
                input_text = "test code" if isinstance(test, str) else str(test["input"])
                return _result(0., f"input: {input_text}\nexpected: {expected}\ngot: {out}\nstderr: {err}"[:2000])
        return _result(1.)


def _completion_text(value):
    return "\n".join(str(m["content"]) for m in value) if isinstance(value, list) else str(value)


def as_trl_reward(verifier: Verifier, prompt_field="prompt"):
    """Adapt a batch verifier to TRL's completion/column reward API (scores are Python floats)."""
    def reward(completions, **columns):
        count = len(completions)
        prompts = columns.get("prompts", columns.get(prompt_field, [""] * count))
        rows = [{k: v[i] for k, v in columns.items() if isinstance(v, (list, tuple)) and len(v) == count}
                for i in range(count)]
        results = verifier([_completion_text(p) for p in prompts], [_completion_text(c) for c in completions], rows)
        if len(results) != count:
            raise ValueError("verifier must return one result per completion")
        return [float(r["score"]) for r in results]
    reward.__name__ = verifier.registry_name
    return reward
