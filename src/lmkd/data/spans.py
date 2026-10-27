"""Span and token offset mapping service (SGL segment/sentence.py:54-88, data/prepare.py:148-159)."""
from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Offset = tuple[int, int]


def encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[int]]:
    """Tokenize text once, returning input_ids and each token's starting character index."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("encode_with_offsets requires a fast PreTrainedTokenizerFast")
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return encoding["input_ids"], [start for start, _ in encoding["offset_mapping"]]


def char_cuts_to_token_spans(
    cuts: list[int], token_starts: list[int], offset: int = 0
) -> list[tuple[int, int]]:
    """Map character cut boundaries to absolute token spans.

    A token belongs to the step holding its first character. Drops zero-token steps.
    """
    if len(cuts) < 2:
        return []
    bounds = [bisect.bisect_left(token_starts, cut) for cut in cuts]
    return [
        (offset + start, offset + end)
        for start, end in zip(bounds, bounds[1:])
        if end > start
    ]


def verify_round_trip(
    tokenizer: Any,
    input_ids: list[int],
    spans: list[tuple[int, int]] | list[dict[str, int]],
    text: str,
) -> bool:
    """Verify that decoded token spans reconstruct the original text."""
    pieces = []
    for s in spans:
        if isinstance(s, dict):
            start, end = s["token_start"], s["token_end"]
        else:
            start, end = s
        pieces.append(tokenizer.decode(input_ids[start:end]))
    decoded = "".join(pieces)
    return decoded.strip() == text.strip()



@dataclass
class SpanResult:
    """Result of :func:`prompt_response_offsets`: per-token char offsets (``len == len(input_ids)`` when ``ok``)."""
    offsets: list[Offset] | None
    ok: bool
    reason: str = ""


def _fail(reason: str) -> SpanResult:
    return SpanResult(None, False, reason)


def _enc_offsets(tokenizer: Any, text: str, add_special_tokens: bool) -> tuple[list[int], list[Offset]]:
    enc = tokenizer(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    return list(enc["input_ids"]), [(int(s), int(e)) for s, e in enc["offset_mapping"]]


def offsets_monotonic(offsets: Sequence[Offset]) -> bool:
    """``s <= e`` per token and non-decreasing along the sequence, ignoring zero-width tokens (BOS/EOS).

    A token repeating the previous token's span exactly is accepted: byte-level BPE gives the same span to the
    pieces of one multi-byte character."""
    prev: Offset | None = None
    for s, e in offsets:
        if s > e:
            return False
        if s == e:
            continue
        if prev is not None and (s, e) != prev and s < prev[1]:
            return False
        prev = (s, e)
    return True


def prompt_response_offsets(tokenizer: Any, prompt_text: str, response_text: str, input_ids: Sequence[int],
                            prompt_len: int, *, add_eos: bool, prompt_add_special_tokens: bool = True) -> SpanResult:
    """Char offsets of every token of ``input_ids`` as built by ``tokenize_prompt_response``.

    Prompt and response are tokenized SEPARATELY (response offsets shifted by ``len(prompt_text)``, EOS gets
    ``(L, L)``), exactly like the training ids, then truncation is replayed from ``prompt_len`` / ``len(input_ids)``
    (right: cut the tail; left_prompt: drop the first ``len(p_ids) - prompt_len`` prompt tokens). The rebuilt ids
    must equal ``input_ids`` (round-trip check) or ``ok=False, reason="ids_mismatch@<pos>"``.
    Source: IMPACT/code/sra_span_utils.py:283-314 (``compute_separate_offsets``; fixed: uses the real ids).
    """
    p_ids, p_off = _enc_offsets(tokenizer, prompt_text, prompt_add_special_tokens)
    r_ids, r_off = _enc_offsets(tokenizer, response_text, False)
    shift = len(prompt_text)
    r_off = [(s + shift, e + shift) for s, e in r_off]
    if add_eos and tokenizer.eos_token_id is not None:
        r_ids.append(tokenizer.eos_token_id)
        r_off.append((shift + len(response_text),) * 2)
    cut = max(len(p_ids) - prompt_len, 0)
    full_ids = (p_ids[cut:] + r_ids)[: len(input_ids)]
    full_off = (p_off[cut:] + r_off)[: len(input_ids)]
    want = list(input_ids)
    if full_ids != want:
        n = min(len(full_ids), len(want))
        pos = next((i for i in range(n) if full_ids[i] != want[i]), n)
        return _fail(f"ids_mismatch@{pos}")
    if not offsets_monotonic(full_off):
        return _fail("offsets_not_monotonic")
    return SpanResult(full_off, True)


def ids_offsets(tokenizer: Any, ids: Sequence[int], text_start: int = 0, *,
                skip_special_tokens: bool = False) -> list[Offset]:
    """Offsets for ids that have no matching source text (rollouts): cumulative-decode prefix lengths.

    ``offset_k = (text_start + len(decode(ids[:k])), text_start + len(decode(ids[:k+1])))``. O(n^2) decode,
    fine up to ~4k tokens. A multi-byte character split between two tokens decodes to ``\ufffd`` for the first
    piece, so offsets are approximate there. ``skip_special_tokens=True`` gives special tokens zero width."""
    pref = [0]
    for k in range(1, len(ids) + 1):
        text = tokenizer.decode(list(ids[:k]), skip_special_tokens=skip_special_tokens,
                                clean_up_tokenization_spaces=False)
        pref.append(max(len(text), pref[-1]))
    return [(text_start + pref[k], text_start + pref[k + 1]) for k in range(len(ids))]


def char_to_token(offsets: Sequence[Offset], c: int) -> int:
    """Index of the first token whose end is ``> c`` (== ``len(offsets)`` if none)."""
    return bisect.bisect_right([e for _, e in offsets], c)


def char_range_to_tokens(offsets: Sequence[Offset], start: int, end: int) -> tuple[int, int]:
    """Half-open token range ``[tok_lo, tok_hi)`` of the tokens overlapping the char range ``[start, end)``."""
    lo = char_to_token(offsets, start)
    hi = bisect.bisect_left([s for s, _ in offsets], end)
    return lo, max(hi, lo)
