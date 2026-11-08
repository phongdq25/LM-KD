"""Shared text generation utilities for evaluation plugins.

Coordinates rollout engines (HF or vLLM), weight synchronization, and token decoding.
"""

from __future__ import annotations

import logging
from typing import Any

from lmkd.core.registry import ROLLOUT
from lmkd.data.transforms.tokenize import render_prompt
from lmkd.rollout.base import Rollout, SamplingSpec

logger = logging.getLogger(__name__)


def _sync_engine_weights(rt: Any, engine: Any) -> None:
    if not hasattr(engine, "update_weights"):
        return
    engine_inst = getattr(rt, "engine", None)
    if engine_inst is not None and hasattr(engine_inst, "iter_full_state_dict"):
        try:
            student = getattr(rt, "student_model", None)
            if student is not None:
                engine.update_weights(engine_inst.iter_full_state_dict(student))
        except Exception as e:
            logger.debug("Rollout weight sync skipped: %s", e)


def generate_for_records(
    rt: Any,
    records: list[dict],
    sampling: SamplingSpec,
    engine_spec: dict | str | None,
    prompt_ids_fn: Any = None,
    *,
    use_chat_template: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> list[list[Rollout]]:
    """Generate rollouts for evaluation records, syncing weights and decoding tokens.

    Uses rt.rollout if configured and engine_spec is None; otherwise builds the requested
    ROLLOUT engine and runs setup(rt).
    """
    if engine_spec is None and getattr(rt, "rollout", None) is not None:
        engine = rt.rollout
    else:
        spec = engine_spec or "hf"
        if isinstance(spec, str):
            spec = {"type": spec}
        engine = ROLLOUT.build(spec)
        engine.setup(rt)

    _sync_engine_weights(rt, engine)

    if prompt_ids_fn is not None:
        prompt_ids_list = [prompt_ids_fn(r) for r in records]
    else:
        tok = getattr(rt, "student_tokenizer", None)
        if tok is None:
            raise ValueError("Runtime does not have student_tokenizer for prompt encoding.")
        bos_id = getattr(tok, "bos_token_id", None)
        prompt_ids_list = []
        tmpl_kwargs = chat_template_kwargs or {}
        should_use_chat = use_chat_template or bool(tmpl_kwargs)

        for r in records:
            raw_p = r.get("prompt")
            if raw_p is None:
                raw_p = r.get("question") or r.get("instruction") or r.get("messages") or r.get("conversations") or ""

            rendered = render_prompt(
                tok,
                raw_p,
                use_chat_template=should_use_chat,
                system_prompt=system_prompt,
                **tmpl_kwargs,
            )
            ids = tok(rendered, add_special_tokens=False)["input_ids"]
            if bos_id is not None and (not ids or ids[0] != bos_id):
                ids = [bos_id] + ids
            prompt_ids_list.append(ids)

    rollouts = engine.generate(prompt_ids_list, sampling)

    tok = getattr(rt, "student_tokenizer", None)
    if tok is not None:
        for prompt_rollouts in rollouts:
            for r in prompt_rollouts:
                if not r.text:
                    r.text = tok.decode(r.response_ids, skip_special_tokens=True)

    return rollouts
