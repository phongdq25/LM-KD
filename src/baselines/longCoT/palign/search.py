"""P-ALIGN prefix binary search algorithm and templates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

JUDGE_TEMPLATE = """
You are a reasoning evaluator.

You are given a partial reasoning prefix extracted from a longer chain-of-thought.
Your task is to judge whether this prefix already contains the essential logical structure and key transformations needed to complete the solution.

- Reply "[ENOUGH]" if the prefix establishes the core reasoning steps such that the remaining reasoning is straightforward or routine.
- Reply "[NOT_ENOUGH]" if any crucial reasoning step is still missing, making it difficult to reliably complete the solution.

Reply with exactly one token: [ENOUGH] or [NOT_ENOUGH].

Question:
{question}

Partial reasoning:
{reasoning_part}
"""

CONTINUE_TEMPLATE = (
    "Please continue from the draft and solve the problem step by step, "
    "and put your final answer within \\boxed{{}}. "
    "I will provide you with some prior knowledge as a draft to assist you in solving the question."
    "*Question*:{question}\n*Prefix*:{prefix}"
)

PALIGN_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def assemble_palign(prefix: str, continuation: str) -> str:
    """Join the teacher prefix and verified student continuation exactly as P-ALIGN.

    Source: SGL/src/sgl/palign/build.py:22-23.
    """
    return f"<Begin_of_Prefix>{prefix}<End_of_Prefix>\n{continuation.strip()}"


@dataclass
class PrefixSearch:
    """Per-question binary search over sentences for the shortest sufficient prefix.

    Source: SGL/src/sgl/palign/truncate.py:53-91.
    """

    question: str
    sentences: list[str]
    left: int = 1
    right: int = 0
    best: int | None = None
    rounds: int = 0

    def __post_init__(self) -> None:
        self.right = len(self.sentences)

    @property
    def done(self) -> bool:
        return self.left > self.right

    @property
    def mid(self) -> int:
        return (self.left + self.right) // 2

    def prompt(self, template: str = JUDGE_TEMPLATE) -> str:
        return template.format(question=self.question, reasoning_part=" ".join(self.sentences[: self.mid]))

    def update(self, reply: str) -> None:
        midpoint = self.mid
        if "[ENOUGH]" in reply or reply.strip() == "ENOUGH":
            self.best, self.right = midpoint, midpoint - 1
        else:
            self.left = midpoint + 1
        self.rounds += 1

    def result(self) -> tuple[str, int, bool]:
        count = len(self.sentences) if self.best is None else self.best
        return " ".join(self.sentences[:count]), count, self.best is not None


def run_searches(
    searches: list[PrefixSearch],
    judge: Callable[[list[str]], list[str]],
    batch_size: int = 512,
    max_rounds: int = 20,
    template: str = JUDGE_TEMPLATE,
) -> None:
    """Advance open searches in lockstep, issuing one batch per round when it fits.

    Source: SGL/src/sgl/palign/truncate.py:94-107.
    """
    for _ in range(max_rounds):
        active = [s for s in searches if not s.done]
        if not active:
            break
        for start in range(0, len(active), batch_size):
            group = active[start : start + batch_size]
            replies = judge([search.prompt(template) for search in group])
            if len(replies) != len(group):
                raise ValueError("judge returned a different number of replies")
            for search, reply in zip(group, replies):
                search.update(reply)
