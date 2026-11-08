from __future__ import annotations

import json

import torch

from lmkd.data.vocab_map import VocabMap, normalize_token, space_mask


def test_overlap_orders(xtok_tokenizers):
    stu, tea = xtok_tokenizers
    vm = VocabMap.build(stu, tea)
    sv = {normalize_token(k): v for k, v in sorted(stu.get_vocab().items(), key=lambda kv: kv[1])}
    tv = {normalize_token(k): v for k, v in sorted(tea.get_vocab().items(), key=lambda kv: kv[1])}
    toks = [k for k in sv if k in tv]
    assert toks, "toy tokenizers must share some tokens (<pad>, <eos>, single characters)"
    assert vm.overlap_s.tolist() == [sv[k] for k in toks]  # student vocab order (DSKDv2)
    assert vm.overlap_t.tolist() == [tv[k] for k in toks]
    srt = sorted(toks)
    assert vm.overlap_sorted_s.tolist() == [sv[k] for k in srt]  # string order
    assert vm.overlap_sorted_t.tolist() == [tv[k] for k in srt]
    assert set(vm.overlap_s.tolist()) == set(vm.overlap_sorted_s.tolist())
    assert set(vm.overlap_t.tolist()) == set(vm.overlap_sorted_t.tolist())
    assert vm.overlap_s.tolist() != vm.overlap_sorted_s.tolist() or len(toks) < 3
    assert not vm.same_vocab
    assert vm.student_vocab_size == len(stu) and vm.teacher_vocab_size == len(tea)


def test_exact_string_map_and_exact_variant(xtok_tokenizers):
    stu, tea = xtok_tokenizers
    vm = VocabMap.build(stu, tea)
    eos_t, eos_s = tea.eos_token_id, stu.eos_token_id
    assert vm.tea2stu[eos_t] == eos_s
    assert vm.stu2tea[eos_s] == eos_t
    mapped = vm.tea2stu >= 0
    assert (vm.stu2tea[vm.tea2stu[mapped]] >= 0).all()
    # ETA variant: unknown teacher tokens resolve to the student's unk id (fast tokenizer behaviour, kept)
    assert vm.tea2stu_exact.shape == (len(tea),)
    assert (vm.tea2stu_exact[~mapped] == stu.unk_token_id).sum() > 0


def test_mapping_file_and_inverse(xtok_tokenizers, tmp_path):
    stu, tea = xtok_tokenizers
    f = tmp_path / "map.json"
    f.write_text(json.dumps({"0": 0, "1": 0, "2": 1}))
    vm = VocabMap.build(stu, tea, mapping_file=str(f))
    assert vm.tea2stu[:3].tolist() == [0, 0, 1]
    assert (vm.tea2stu[3:] == -1).all()
    assert vm.stu2tea[0] == 0  # first teacher id kept (max_align_num = 1)
    assert vm.stu2tea[1] == 2
    assert (vm.stu2tea[2:] == -1).all()


def test_space_mask_fallback(xtok_tokenizers):
    stu, tea = xtok_tokenizers
    vm = VocabMap.build(stu, tea, space_mask_mode="space+special")
    toks = tea.convert_ids_to_tokens(list(range(len(tea))))
    for i, t in enumerate(toks):
        expect = t.startswith("Ġ") or i in set(tea.all_special_ids)
        assert bool(vm.space_mask_t[i]) == expect
    assert any(t.startswith("Ġ") for t in toks)
    assert vm.space_mask_s[stu.eos_token_id] and not vm.space_mask_s[stu.convert_tokens_to_ids("w3")]
    no_special = space_mask(tea, "space", len(tea) + 5)  # padded embedding rows -> False
    assert no_special.shape == (len(tea) + 5,) and not no_special[-5:].any()
    assert not no_special[tea.eos_token_id]


def test_save_load_roundtrip(xtok_tokenizers, tmp_path):
    stu, tea = xtok_tokenizers
    vm = VocabMap.build(stu, tea, space_mask_mode="space")
    vm.save(tmp_path / "vm.pt")
    vm2 = VocabMap.load(tmp_path / "vm.pt")
    for name in ("overlap_s", "overlap_t", "overlap_sorted_s", "tea2stu", "stu2tea", "tea2stu_exact", "space_mask_t"):
        assert torch.equal(getattr(vm, name), getattr(vm2, name))
    assert vm2.same_vocab == vm.same_vocab and vm2.space_mask_s is not None


def test_same_vocab(xtok_tokenizers):
    stu, _ = xtok_tokenizers
    vm = VocabMap.build(stu, stu)
    assert vm.same_vocab
    assert vm.tea2stu.tolist() == list(range(len(stu)))
