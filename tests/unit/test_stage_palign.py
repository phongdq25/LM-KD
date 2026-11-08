"""P-ALIGN search and assembly tests."""

from __future__ import annotations

from baselines.longCoT.palign import PrefixSearch, assemble_palign, run_searches


def test_binary_search_lockstep() -> None:
    searches = [PrefixSearch("Q", [f"sentence {i}." for i in range(8)]) for _ in range(4)]
    batch_sizes: list[int] = []

    def judge(prompts: list[str]) -> list[str]:
        batch_sizes.append(len(prompts))
        return ["[ENOUGH]" if p.count("sentence") >= 3 else "[NOT_ENOUGH]" for p in prompts]

    run_searches(searches, judge, batch_size=8, max_rounds=20)
    assert all(s.result()[1:3] == (3, True) for s in searches)
    assert len(batch_sizes) <= 4
    assert all(size == 4 for size in batch_sizes)


def test_no_sufficient_keeps_all() -> None:
    search = PrefixSearch("Q", ["A.", "B.", "C."])
    run_searches([search], lambda prompts: ["[NOT_ENOUGH]"] * len(prompts))
    assert search.result()[:3] == ("A. B. C.", 3, False)


def test_assemble_format() -> None:
    assert assemble_palign("First.", "  Final ") == "<Begin_of_Prefix>First.<End_of_Prefix>\nFinal"
