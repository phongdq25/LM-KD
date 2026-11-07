from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lmkd.core.registry import TEACHER_INPUTS, TRANSFORMS
from lmkd.data.collator import Collator
from lmkd.data.transforms.base import run_transforms


def _rt(ctx, max_length=64):
    stu, tea = ctx.student_tokenizer, ctx.teacher_tokenizer
    return SimpleNamespace(
        student_tokenizer=stu,
        teacher_tokenizer=tea,
        data_ctx=ctx,
        config=SimpleNamespace(data=SimpleNamespace(max_length=max_length, pad_to_multiple_of=None)),
        collator=Collator({"student": stu.pad_token_id, "teacher": tea.pad_token_id}, pad_to_multiple_of=None),
    )


def _student_batch(records, ctx, rt, n=4):
    tok = TRANSFORMS.build({"type": "tokenize_sft", "use_chat_template": False, "keep_text": False})
    samples = run_transforms(records[:n], [tok], ctx)
    return rt.collator(samples), samples


def _builder(rt, aligners=()):
    b = TEACHER_INPUTS.build({"type": "retokenize", "aligners": list(aligners)})
    b.setup(rt)
    return b


def test_retokenize_builds_teacher(xtok_records, xtok_ctx):
    rt = _rt(xtok_ctx)
    batch, samples = _student_batch(xtok_records, xtok_ctx, rt)
    student_ids = batch["student.input_ids"].clone()
    out = _builder(rt, ["exact_token", {"type": "lcs_span"}]).build(batch, rt)
    stu, tea = xtok_ctx.student_tokenizer, xtok_ctx.teacher_tokenizer
    assert torch.equal(out["student.input_ids"], student_ids)  # student side untouched
    for b, s in enumerate(samples):
        ps, ls = s["student.prompt_len"], len(s["student.input_ids"])
        t_len = int(out["teacher.attention_mask"][b].sum())
        t_ids = out["teacher.input_ids"][b, :t_len].tolist()
        pt = int(out["teacher.prompt_len"][b])
        t_resp = tea.decode(t_ids[pt:], skip_special_tokens=True).strip()
        s_resp = stu.decode(s["student.input_ids"][ps:], skip_special_tokens=True).strip()
        assert t_resp == s_resp
        assert (out["teacher.labels"][b, :pt] == -100).all() and (out["teacher.labels"][b, pt:t_len] != -100).all()
        assert t_ids[-1] == tea.eos_token_id
        assert out["student.offsets"].shape[1] == student_ids.shape[1]
        assert (out["student.offsets"][b, ls:] == -1).all()
    assert out["align.eta_t_for_s"].shape == student_ids.shape
    assert out["align.eta_s_for_t"].shape == out["teacher.input_ids"].shape
    assert "align.sra_s_idx" in out and isinstance(out["align.sra_s_idx"], list)
    assert "seq.loss_mask" not in out  # nothing failed


def test_retokenize_student_offsets_cover_text(xtok_records, xtok_ctx):
    rt = _rt(xtok_ctx)
    batch, samples = _student_batch(xtok_records, xtok_ctx, rt, n=3)
    out = _builder(rt).build(batch, rt)
    stu = xtok_ctx.student_tokenizer
    for b, s in enumerate(samples):
        ids = s["student.input_ids"]
        ps = s["student.prompt_len"]
        p_text = stu.decode(ids[:ps], clean_up_tokenization_spaces=False)
        r_text = stu.decode(ids[ps:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        full = p_text + r_text
        offs = out["student.offsets"][b, : len(ids)].tolist()
        for k in range(ps, len(ids) - 1):
            a, e = offs[k]
            assert full[a:e].strip() == stu.decode([ids[k]]).strip()
        assert offs[-1] == [len(full), len(full)]


def test_retokenize_only_if_missing(xtok_records, xtok_ctx):
    rt = _rt(xtok_ctx)
    batch, _ = _student_batch(xtok_records, xtok_ctx, rt)
    batch["teacher.input_ids"] = torch.zeros(4, 3, dtype=torch.long)
    out = _builder(rt).build(batch, rt)
    assert out["teacher.input_ids"].shape == (4, 3)  # kept as is
    assert "student.offsets" not in out


def test_retokenize_failed_rows_are_masked_not_dropped(xtok_records, xtok_ctx):
    rt = _rt(xtok_ctx, max_length=3)  # teacher prompt never fits -> failure
    batch, _ = _student_batch(xtok_records, xtok_ctx, rt, n=3)
    out = _builder(rt, ["exact_token"]).build(batch, rt)
    assert out.batch_size == 3
    assert (out["seq.loss_mask"] == 0).all()
    assert out["teacher.input_ids"].shape[0] == 3 and out["align.eta_t_for_s"].shape == out["student.input_ids"].shape


def test_retokenize_provided_keys():
    cls = TEACHER_INPUTS.get("retokenize")
    cfg = cls.Config(aligners=["exact_token", {"type": "char_span"}])
    keys = cls.provided_keys(cfg)
    assert "teacher.offsets" in keys and "align.eta_t_for_s" in keys and "align.span_pairs" in keys
    assert TEACHER_INPUTS.meta("retokenize").status == "stable"


@pytest.mark.parametrize("name", ["same", "pretokenized", "retokenize"])
def test_builders_registered(name):
    assert name in TEACHER_INPUTS
