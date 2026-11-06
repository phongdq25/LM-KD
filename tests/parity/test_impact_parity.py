"""IMPACT tensor parity against verbatim original functions on tiny cross-tokenizer data."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch
from tests.parity._ref import impact_ref as ref

from baselines.cross_tokenizer.impact import Impact, vectorized_teacher_increments
from lmkd.core.registry import ALIGNERS
from lmkd.core.state import TrainState
from lmkd.data.collator import Collator
from lmkd.data.vocab_map import VocabMap
from lmkd.losses.base import LossContext, PrepareContext
from lmkd.teachers.signals import TeacherOutput

pytest_plugins = ("tests.xtok.conftest",)


class _EmbModel(torch.nn.Module):
    """Tiny embedding holder exposing the original projection API."""

    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.config = SimpleNamespace(hidden_size=dim)

    def get_input_embeddings(self):
        return self.emb


def _runtime(stu_tok, tea_tok, *, scores=None):
    torch.manual_seed(150)
    sm = _EmbModel(len(stu_tok), 3)
    tm = _EmbModel(len(tea_tok), 4)
    adapter = SimpleNamespace(embed_tokens=lambda model: model.get_input_embeddings())
    vm = VocabMap.build(stu_tok, tea_tok)
    return SimpleNamespace(
        student_model=sm,
        teacher_model=tm,
        student_adapter=adapter,
        teacher_adapter=adapter,
        student_tokenizer=stu_tok,
        teacher_tokenizer=tea_tok,
        data_ctx=SimpleNamespace(vocab_map=vm),
        extra={"impact_layer_scores": scores} if scores is not None else {},
        config=SimpleNamespace(train=SimpleNamespace(seed=7)),
    )


@pytest.mark.parity
@pytest.mark.parametrize("max_overlap", [None, 10])
def test_W_matches_ref(xtok_tokenizers, max_overlap):
    rt = _runtime(*xtok_tokenizers)
    crit = Impact(Impact.Config(max_overlap=max_overlap, layer_mapper={"type": "first", "k": 2}))
    crit.prepare(PrepareContext(rt))
    expected = ref.compute_vocab_projection(
        rt.teacher_model, rt.student_model, rt.teacher_tokenizer, rt.student_tokenizer, max_overlap=max_overlap
    )
    torch.testing.assert_close(crit.W, expected, atol=1e-5, rtol=0)
    assert "W" in crit.extra_state() and "mapper" in crit.extra_state()


def _case(samples, xtok_ctx, mode):
    aligned = []
    aligner = ALIGNERS.build({"type": "boundary_interp", "mode": mode})
    for sample in samples[:2]:
        row = dict(sample)
        row.update(aligner(row, xtok_ctx))
        aligned.append(row)
    pads = {"student": xtok_ctx.student_tokenizer.pad_token_id, "teacher": xtok_ctx.teacher_tokenizer.pad_token_id}
    return Collator(pads, pad_to_multiple_of=None)(aligned), aligned


@pytest.mark.parity
@pytest.mark.parametrize("mode", ["linear", "nearest", "containing"])
@pytest.mark.parametrize(
    "mapper,choose_align,choose_layer",
    [
        ("topk_bi", "BI", "BI"),
        ("first", "BI", "FIRST"),
        ("ratio_1to1", "1-1", "BI"),
        ("railkd", "1-random", "BI"),
        ("alpkd", "1-all", "BI"),
    ],
)
def test_impact_loss_parity(mapper, choose_align, choose_layer, mode, xtok_samples, xtok_ctx):
    torch.manual_seed(151)
    batch, rows = _case(xtok_samples, xtok_ctx, mode)
    B, Ts = batch["student.input_ids"].shape
    Tt = batch["teacher.input_ids"].shape[1]
    student = [torch.randn(B, Ts, 3) for _ in range(3)]
    teacher = [torch.randn(B, Tt, 4) for _ in range(4)]
    scores = torch.tensor([0.5, 0.2, 0.8])
    rt = _runtime(xtok_ctx.student_tokenizer, xtok_ctx.teacher_tokenizer, scores=scores)
    spec = {"type": mapper, "k": 2, "tau": 0.7}
    if mapper == "railkd":
        spec["seed"] = 7
    crit = Impact(Impact.Config(layer_mapper=spec, boundary_mode=mode))
    crit.prepare(PrepareContext(rt))
    s_side = SimpleNamespace(num_layers=2, hidden=lambda l: student[l])
    t_side = SimpleNamespace(num_layers=3, hidden=lambda l: teacher[l])
    ctx = LossContext(batch, s_side, [], TeacherOutput(side=t_side), None, TrainState(), rt)
    got = crit(ctx).terms["impact"]
    expected = ref.compute_impact_loss(
        student,
        teacher,
        [r["student.offsets"] for r in rows],
        [r["teacher.offsets"] for r in rows],
        crit.W,
        batch["student.attention_mask"],
        batch["teacher.attention_mask"],
        top_k=2,
        bi_tau=0.7,
        L_T=3,
        L_S=2,
        choose_layer=choose_layer,
        choose_align=choose_align,
        random_seed=7,
        boundary_mode=mode,
        precomputed_layer_scores=scores,
    )
    assert got.reduction == "seq_sum"
    torch.testing.assert_close(got.value / B, expected, atol=1e-5, rtol=0)


@pytest.mark.parity
def test_legacy_scale(xtok_samples, xtok_ctx):
    batch, _ = _case(xtok_samples, xtok_ctx, "linear")
    torch.manual_seed(154)
    B, Ts = batch["student.input_ids"].shape
    Tt = batch["teacher.input_ids"].shape[1]
    sh = [torch.randn(B, Ts, 3) for _ in range(3)]
    th = [torch.randn(B, Tt, 4) for _ in range(4)]
    rt = _runtime(xtok_ctx.student_tokenizer, xtok_ctx.teacher_tokenizer)
    sides = (
        SimpleNamespace(num_layers=2, hidden=lambda l: sh[l]),
        TeacherOutput(side=SimpleNamespace(num_layers=3, hidden=lambda l: th[l])),
    )
    values = []
    for legacy in (False, True):
        crit = Impact(Impact.Config(layer_mapper={"type": "first", "k": 2}, legacy_scale=legacy))
        crit.prepare(PrepareContext(rt))
        ctx = LossContext(batch, sides[0], [], sides[1], None, TrainState(), rt)
        values.append(crit(ctx).terms["impact"])
    assert values[1].reduction == "token_sum"
    torch.testing.assert_close(values[1].value, values[0].value / B)


@pytest.mark.parity
@pytest.mark.slow
def test_vectorized_faster():
    torch.manual_seed(152)
    B, T, Ds, Dt = 4, 128, 16, 24
    hidden = torch.randn(B, T, Dt)
    W = torch.randn(Dt, Ds)
    mask = torch.ones(B, T, dtype=torch.long)
    offsets = [[(i, i + 1) for i in range(T)] for _ in range(B)]
    lo = torch.arange(T).view(1, T, 1).expand(B, -1, 2)
    hi = lo.clone()
    lo = lo + torch.tensor([0, 1]).view(1, 1, 2)
    hi = lo.clone()
    w = torch.zeros(B, T, 2)
    projected = hidden @ W
    vectorized_teacher_increments(projected, lo, hi, w)
    ref.compute_teacher_increments_batch(offsets, offsets, hidden, W, mask, mask)
    t0 = time.perf_counter()
    for _ in range(2):
        ref.compute_teacher_increments_batch(offsets, offsets, hidden, W, mask, mask)
    slow = (time.perf_counter() - t0) / 2
    t0 = time.perf_counter()
    for _ in range(10):
        vectorized_teacher_increments(projected, lo, hi, w)
    fast = (time.perf_counter() - t0) / 10
    assert slow / fast >= 10, (slow, fast)
