"""Prompt formatting and thinking block reconciliation (SGL data/prepare.py:32-101)."""
from __future__ import annotations

from typing import Any

# Byte-identical to SGL data/prepare.py:24
PROMPT_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.{problem}"
PALIGN_EMPTY_THINK = "<think>\n\n</think>\n\n"


def close_open_thinking(prompt: str) -> str:
    """Close a thinking block the template opened but the run asked not to use."""
    return f"{prompt}</think>\n\n" if prompt.rstrip().endswith("<think>") else prompt


def palign_close_thinking(prompt: str) -> str:
    """Close an open `<think>` the way P-ALIGN does (two newlines inside the block)."""
    stripped = prompt.rstrip()
    if stripped.endswith("<think>"):
        return stripped[: -len("<think>")] + PALIGN_EMPTY_THINK
    return prompt


def reconcile_thinking_markers(prompt: str, response: str) -> str:
    """Ensure response does not open a thinking block if the prompt already did or did not."""
    body = response.lstrip()
    if prompt.rstrip().endswith("<think>"):
        return body[len("<think>") :].lstrip("\n") if body.startswith("<think>") else response
    return body.replace("<think>", "", 1).replace("</think>", "", 1).lstrip("\n")


def build_prompt(
    tokenizer: Any,
    problem: str,
    *,
    chat_template: bool = True,
    thinking_mode: str = "on",
    instruction_separator: str = "",
    template: str = PROMPT_TEMPLATE,
) -> str:
    """Render prompt using template and optional chat_template with thinking block rules."""
    instruction = template.format(problem=instruction_separator + problem)
    if not chat_template:
        return instruction

    if thinking_mode == "on":
        enable_thinking = True
        palign_prompt = False
    elif thinking_mode == "off_closed":
        enable_thinking = False
        palign_prompt = False
    elif thinking_mode == "palign_closed":
        enable_thinking = False
        palign_prompt = True
    else:
        raise ValueError(
            f"unknown thinking_mode: {thinking_mode!r}, must be 'on', 'off_closed', or 'palign_closed'"
        )

    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:                       # template/tokenizer without extra chat-template kwargs
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

    if enable_thinking:
        return prompt
    return palign_close_thinking(prompt) if palign_prompt else close_open_thinking(prompt)


def probe(tokenizer: Any, problem: str = "PROBLEM", **kw) -> str:
    """Render sample prompt to inspect template formatting."""
    return build_prompt(tokenizer, problem, **kw)

