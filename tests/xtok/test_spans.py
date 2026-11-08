from __future__ import annotations

import pytest

from lmkd.data.spans import (
    char_range_to_tokens,
    char_to_token,
    ids_offsets,
    prompt_response_offsets,
)
from lmkd.data.transforms.tokenize import tokenize_prompt_response


def test_fixture_tokenizers_differ(xtok_tokenizers):
    stu, tea = xtok_tokenizers
    text = "A: w12 w3"
    assert len(stu(text, add_special_tokens=False)["input_ids"]) != len(
        tea(text, add_special_tokens=False)["input_ids"]
    )


def test_offsets_match_tokenize_sft(xtok_samples, xtok_tokenizers):
    n_checked = 0
    for side, tok in zip(("student", "teacher"), xtok_tokenizers):
        for s in xtok_samples:
            ptxt = s["meta.prompt_text"] if side == "student" else s["meta.teacher_prompt_text"]
            res = prompt_response_offsets(
                tok, ptxt, s["meta.response_text"], s[f"{side}.input_ids"], s[f"{side}.prompt_len"], add_eos=True
            )
            assert res.ok, res.reason
            assert len(res.offsets) == len(s[f"{side}.input_ids"])
            full = ptxt + s["meta.response_text"]
            plen = s[f"{side}.prompt_len"]
            for k in range(plen, len(res.offsets) - 1):  # response tokens, last one is EOS
                a, b = res.offsets[k]
                n_checked += 1
                assert full[a:b].strip() == tok.decode([s[f"{side}.input_ids"][k]]).strip()
            L = len(full)
            assert res.offsets[-1] == (L, L)  # EOS
    assert n_checked > 100


def test_offsets_mismatch_detected(xtok_samples, xtok_tokenizers):
    stu, _ = xtok_tokenizers
    s = xtok_samples[0]
    ids = list(s["student.input_ids"])
    ids[3] = (ids[3] + 1) % 60 + 3
    res = prompt_response_offsets(
        stu, s["meta.prompt_text"], s["meta.response_text"], ids, s["student.prompt_len"], add_eos=True
    )
    assert not res.ok and res.reason.startswith("ids_mismatch")
    assert res.reason == "ids_mismatch@3"


@pytest.mark.parametrize("truncation", ["left_prompt", "right"])
def test_truncation_replayed(xtok_tokenizers, truncation):
    for tok in xtok_tokenizers:
        prompt = "Q: w1 + w2 = Q: w3 + w4 = Q: w5 + w6 ="
        resp = "A: w7 w8 w9"
        n_p = len(tok(prompt)["input_ids"])
        n_r = len(tok(resp, add_special_tokens=False)["input_ids"]) + 1
        max_len = n_p + 2 if truncation == "right" else n_p + n_r - 3  # right: cut tail; left_prompt: cut 3
        got = tokenize_prompt_response(tok, prompt, resp, max_len, True, truncation)
        assert got is not None
        ids, plen = got
        res = prompt_response_offsets(tok, prompt, resp, ids, plen, add_eos=True)
        assert res.ok, res.reason
        assert len(res.offsets) == len(ids) == max_len
        assert (plen < n_p) == (truncation == "left_prompt")


def test_ids_offsets_monotonic(xtok_tokenizers, xtok_samples):
    for tok, side in zip(xtok_tokenizers, ("student", "teacher")):
        ids = xtok_samples[3][f"{side}.input_ids"]
        offs = ids_offsets(tok, ids)
        assert all(a <= b for a, b in offs)
        assert all(offs[k][1] == offs[k + 1][0] for k in range(len(offs) - 1))
        assert offs[-1][1] == len(tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        shifted = ids_offsets(tok, ids, text_start=10)
        assert shifted[0][0] == 10


def test_char_lookup():
    offs = [(0, 0), (0, 2), (2, 5), (5, 5), (5, 9)]
    assert char_to_token(offs, 0) == 1
    assert char_to_token(offs, 2) == 2
    assert char_to_token(offs, 5) == 4
    assert char_to_token(offs, 9) == 5
    assert char_range_to_tokens(offs, 1, 6) == (1, 5)
    assert char_range_to_tokens(offs, 2, 5) == (2, 3)
