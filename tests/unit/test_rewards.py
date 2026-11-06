"""T25: verifiers return one score and actionable feedback per completion."""

import json
import mmap
import time
from concurrent.futures import ThreadPoolExecutor

from lmkd.rl.rewards import MCQ, CodeSandbox, MathVerify, as_trl_reward, last_boxed


def test_last_boxed():
    assert last_boxed(r"a \boxed{\frac{1}{2}} b \boxed{3}") == "3"
    assert last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{broken") is None


def test_math_verify_reward():
    verifier = MathVerify()
    result = verifier(["p", "p"], [r"\boxed{1/2}", "0.5"], [{"answer": "0.5"}] * 2)
    assert result[0]["acc"] == 1.0
    assert result[1]["score"] == 0.0 and "boxed" in result[1]["feedback"]


def test_math_verify_fallback(monkeypatch):
    def missing(*args):
        raise ImportError("simulate unavailable verification backend")

    monkeypatch.setattr("lmkd.rl.rewards._math_equivalent", missing)
    assert MathVerify()(["p"], [r"\boxed{A B}"], [{"answer": "AB"}])[0]["score"] == 1.0


def test_math_verify_from_worker_thread():
    with ThreadPoolExecutor(1) as executor:
        result = executor.submit(MathVerify(), ["p"], [r"\boxed{2}"], [{"answer": "2"}]).result(15)
    assert result[0]["score"] == 1.0


def test_as_trl_reward():
    reward = as_trl_reward(MCQ())
    assert reward(
        completions=["Answer: A", [{"role": "assistant", "content": r"\boxed{B}"}]],
        prompts=["p", "p"],
        answer=["A", "C"],
    ) == [1.0, 0.0]
    assert reward.__name__ == "mcq"


def test_code_sandbox_feedback():
    verifier = CodeSandbox(CodeSandbox.Config(timeout_s=0.3, memory_mb=256))
    meta = [{"ground_truth": json.dumps([{"input": "2\n", "output": "4"}])}]
    assert verifier(["p"], ["print(int(input()) * 2)"], meta)[0]["score"] == 1.0
    bad = verifier(["p"], ["print(3)"], meta)[0]
    assert bad["score"] == 0.0 and "expected" in bad["feedback"]
    timeout = verifier(["p"], ["while True: pass"], meta)[0]
    assert timeout["score"] == 0.0 and "timeout" in timeout["feedback"].lower()


def test_code_sandbox_test_code():
    result = CodeSandbox()(["p"], ["def add(a, b): return a+b"], [{"ground_truth": "assert add(2, 3) == 5"}])
    assert result[0]["score"] == 1.0


def test_math_verify_timeout(monkeypatch):
    def hung(*args, **kwargs):
        time.sleep(2)
        return []

    monkeypatch.setattr("lmkd.rl.rewards.parse", hung)
    start = time.monotonic()
    result = MathVerify(MathVerify.Config(timeout_s=0.05))(["p"], [r"\boxed{1}"], [{"answer": "2"}])
    assert result[0]["score"] == 0.0
    assert time.monotonic() - start < 0.5


def test_code_sandbox_restricts_network_and_writes(tmp_path):
    outside = tmp_path / "outside.txt"
    code = (
        "import socket\n"
        "try:\n socket.socket()\nexcept PermissionError:\n pass\nelse:\n raise AssertionError('network allowed')\n"
        f"try:\n open({str(outside)!r}, 'w').write('bad')\n"
        "except PermissionError:\n pass\nelse:\n raise AssertionError('outside write allowed')\n"
        "open('inside.txt', 'w').write('ok')\nprint('ok')"
    )
    result = CodeSandbox()(["p"], [code], [{"ground_truth": [{"input": "", "output": "ok"}]}])
    assert result[0]["score"] == 1.0, result
    assert not outside.exists()


def test_code_sandbox_after_large_parent_mapping():
    # CUDA/vLLM can reserve far more virtual memory than the verifier's child limit.
    with mmap.mmap(-1, 1024**3):
        result = CodeSandbox(CodeSandbox.Config(memory_mb=64))(
            ["p"], ["print('ok')"], [{"ground_truth": [{"input": "", "output": "ok"}]}]
        )
    assert result[0]["score"] == 1.0, result
