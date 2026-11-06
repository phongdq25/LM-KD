"""Aligner parity tests against verbatim copies of the original code (tests/parity/_ref/xtok_ref.py)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from baselines.cross_tokenizer.impact import boundary_coeffs, boundary_table
from lmkd.core.batch import FIELDS
from lmkd.core.registry import ALIGNERS
from lmkd.data.vocab_map import VocabMap

pytestmark = pytest.mark.parity


def _ref():
    from tests.parity._ref import xtok_ref

    return xtok_ref


def build(spec):
    return ALIGNERS.build(spec)


# --------------------------------------------------------------------------- exact_token
def test_exact_token_parity(xtok_samples, xtok_ctx):
    ref = _ref()
    stu, tea = xtok_ctx.student_tokenizer, xtok_ctx.teacher_tokenizer
    al = build("exact_token")
    total = 0
    for s in xtok_samples:
        out = al(s, xtok_ctx)
        ps, pt = s["student.prompt_len"], s["teacher.prompt_len"]
        t_al, s_al = ref.align_sequences(
            tea.convert_ids_to_tokens(s["teacher.input_ids"][pt:]),
            stu.convert_ids_to_tokens(s["student.input_ids"][ps:]),
            stu,
            tea,
        )
        exp_t = [-1] * len(s["student.input_ids"])
        exp_s = [-1] * len(s["teacher.input_ids"])
        for ti, sj in zip(t_al, s_al):
            exp_t[ps + sj] = pt + ti
            exp_s[pt + ti] = ps + sj
        assert out["align.eta_t_for_s"] == exp_t and out["align.eta_s_for_t"] == exp_s
        for sp, tp in enumerate(out["align.eta_t_for_s"]):  # the two maps are inverse of each other
            if tp >= 0:
                assert out["align.eta_s_for_t"][tp] == sp
        assert all(v == -1 for v in out["align.eta_t_for_s"][:ps])
        total += len(t_al)
    assert total > 0


def test_exact_token_hand_example(xtok_ctx):
    stu, tea = xtok_ctx.student_tokenizer, xtok_ctx.teacher_tokenizer
    ids_s = stu("A: w1", add_special_tokens=False)["input_ids"] + [stu.eos_token_id]
    ids_t = tea("A: w1", add_special_tokens=False)["input_ids"] + [tea.eos_token_id]
    s = {"student.input_ids": ids_s, "student.prompt_len": 0, "teacher.input_ids": ids_t, "teacher.prompt_len": 0}
    out = build("exact_token")(s, xtok_ctx)
    assert out["align.eta_t_for_s"][-1] == len(ids_t) - 1  # EOS <-> EOS
    assert out["align.eta_s_for_t"][-1] == len(ids_s) - 1


# --------------------------------------------------------------------------- char_span
def test_char_span_parity(xtok_samples, xtok_ctx):
    ref = _ref()
    al = build({"type": "char_span", "span_of": "context"})
    npairs = 0
    for s in xtok_samples:
        out = al(s, xtok_ctx)
        ps, pt = s["student.prompt_len"], s["teacher.prompt_len"]
        ls, lt = len(s["student.input_ids"]), len(s["teacher.input_ids"])
        s_pos = list(range(ps - 1, ls - 1))  # original positions p (label p != -100); target t = p + 1
        t_pos = list(range(pt - 1, lt - 1))
        nb = ref._span_align(
            [tuple(s["student.offsets"][p]) for p in s_pos], [tuple(s["teacher.offsets"][p]) for p in t_pos]
        )
        exp = {(s_pos[i] + 1, t_pos[j] + 1) for i, n in enumerate(nb) for j in n}
        assert {tuple(p) for p in out["align.span_pairs"]} == exp
        npairs += len(exp)
        assert all(ps <= a < ls and pt <= b < lt for a, b in out["align.span_pairs"])
    assert npairs > 0


def test_char_span_target_mode(xtok_samples, xtok_ctx):
    s = xtok_samples[0]
    a = build({"type": "char_span", "span_of": "target"})(s, xtok_ctx)["align.span_pairs"]
    b = build({"type": "char_span", "span_of": "context"})(s, xtok_ctx)["align.span_pairs"]
    assert a and b and a != b


# --------------------------------------------------------------------------- boundary_interp
def _rand_offsets(rng, n, zero_width=True):
    pos, offs = 0, []
    for _ in range(n):
        w = rng.choice([0, 1, 2, 3, 5]) if zero_width else rng.choice([1, 2, 3, 5])
        offs.append((pos, pos + w))
        pos += w
    return offs


@pytest.mark.parametrize("mode", ["linear", "nearest", "containing"])
def test_boundary_coeffs_equals_original(mode):
    ref = _ref()
    rng = random.Random(0)
    for trial in range(40):
        n = rng.randint(1, 12)
        t_off = _rand_offsets(rng, n)
        if trial % 4 == 0:  # non-monotone offsets: the literal scan must still agree
            t_off = [tuple(sorted((rng.randint(0, 20), rng.randint(0, 20)))) for _ in range(n)]
        P = torch.randn(n + 1, 5, dtype=torch.float64)
        end = t_off[-1][1] + 3
        cs = list(range(-1, end + 1))
        table = boundary_table(cs, t_off, mode)
        for c, (lo, hi, w) in zip(cs, table):
            exp = ref._get_teacher_boundary_state(c, t_off, P, mode=mode)
            got = (1 - w) * P[lo] + w * P[hi]
            assert torch.allclose(got, exp, atol=1e-6), (t_off, c, mode)
            assert (lo, hi, w) == boundary_coeffs(c, t_off, mode)


def test_boundary_interp_aligner(xtok_samples, xtok_ctx):
    ref = _ref()
    s = xtok_samples[2]
    out = build({"type": "boundary_interp", "mode": "linear"})(s, xtok_ctx)
    n_s, n_t = len(s["student.input_ids"]), len(s["teacher.input_ids"])
    assert np.asarray(out["align.impact_lo"]).shape == (n_s, 2)
    assert out["meta.impact_mode"] == "linear"
    P = torch.randn(n_t + 1, 4, dtype=torch.float64)
    P[0] = 0
    t_off = [tuple(o) for o in s["teacher.offsets"]]
    lo, hi, w = (torch.tensor(out[f"align.impact_{k}"]) for k in ("lo", "hi", "w"))
    for i in range(n_s):
        a_i, b_i = s["student.offsets"][i]
        for col, c in enumerate((a_i, b_i)):
            exp = ref._get_teacher_boundary_state(c, t_off, P, mode="linear")
            got = (1 - w[i, col]) * P[lo[i, col]] + w[i, col] * P[hi[i, col]]
            assert torch.allclose(got, exp, atol=1e-6)


# --------------------------------------------------------------------------- dtw_edit
def test_dtw_minED_parity(xtok_samples, xtok_ctx):
    editdistance = pytest.importorskip("editdistance")
    ref = _ref()
    from lmkd.data.aligners.dtw_edit import dtw

    rng = random.Random(1)
    vocab = ["a", "ab", "abc", "b", "bc", "x", "xy", "w1", "w12", "Ġw", "Ġ", "3"]
    for _ in range(10):
        s1 = [rng.choice(vocab) for _ in range(rng.randint(1, 8))]
        s2 = [rng.choice(vocab) for _ in range(rng.randint(1, 8))]
        mine = dtw(s1, s2, norm_func=editdistance.eval)
        orig = ref.dtw_min_ed(s1, s2, norm_func=editdistance.eval)
        assert mine[0] == orig[0] and mine[1] == orig[1] and mine[2] == orig[2] and mine[3] == orig[3]
        assert np.array_equal(mine[4], orig[4])


@pytest.mark.parametrize("require_id_map", [True, False])
def test_dtw_minED_aligner(xtok_samples, xtok_ctx, require_id_map):
    editdistance = pytest.importorskip("editdistance")
    ref = _ref()
    stu, tea = xtok_ctx.student_tokenizer, xtok_ctx.teacher_tokenizer
    vm = VocabMap.build(stu, tea)
    xtok_ctx.vocab_map = vm
    try:
        al = build({"type": "dtw_edit", "require_id_map": require_id_map})
        n_pairs = 0
        for s in xtok_samples[:8]:
            out = al(s, xtok_ctx)["align.dtw_t_for_s"]
            s_ids, t_ids = s["student.input_ids"], s["teacher.input_ids"]
            ps, pt = s["student.prompt_len"], s["teacher.prompt_len"]
            ctx_s, ctx_t = s_ids[ps - 1 : len(s_ids) - 1], t_ids[pt - 1 : len(t_ids) - 1]

            def dist(a, b):
                return editdistance.eval(a.replace("Ġ", ""), b.replace("▁", ""))  # teacher byte-level, student word

            _, _, _, b2t, _ = ref.dtw_min_ed(
                tea.convert_ids_to_tokens(ctx_t), stu.convert_ids_to_tokens(ctx_s), norm_func=dist
            )
            exp = [-1] * len(s_ids)
            for k, js in enumerate(b2t):
                if len(js) == 1 and (not require_id_map or ctx_s[k] == int(vm.tea2stu[ctx_t[js[0]]])):
                    exp[ps + k] = pt + js[0]
            assert out == exp
            assert all(v == -1 or v >= pt for v in out)
            n_pairs += sum(v >= 0 for v in out)
        assert n_pairs > 0 or require_id_map  # the id check is strict: tiny vocabularies may have no pair
    finally:
        xtok_ctx.vocab_map = None


def test_dtw_needs_vocab_map(xtok_samples, xtok_ctx):
    pytest.importorskip("editdistance")
    xtok_ctx.vocab_map = None
    with pytest.raises(RuntimeError, match="vocab map"):
        build("dtw_edit")(xtok_samples[0], xtok_ctx)


def test_dtw_full_sequence_lengths(xtok_samples, xtok_ctx):
    editdistance = pytest.importorskip("editdistance")
    ref = _ref()
    al = build({"type": "dtw_edit", "full_sequence": True, "require_id_map": False})
    s = xtok_samples[1]
    out = al(s, xtok_ctx)
    ls, lt = len(s["student.input_ids"]), len(s["teacher.input_ids"])
    s2t = np.asarray(out["align.dtw_s2t"])
    t2s = np.asarray(out["align.dtw_t2s"])
    assert s2t[:, 0].max() == ls - 1 and s2t[:, 1].max() == lt - 1  # whole sequence, special tokens kept
    assert s2t.min() == 0 and t2s[:, 0].max() == lt - 1 and t2s[:, 1].max() == ls - 1
    stu_tok = xtok_ctx.student_tokenizer.convert_ids_to_tokens(s["student.input_ids"])
    tea_tok = xtok_ctx.teacher_tokenizer.convert_ids_to_tokens(s["teacher.input_ids"])
    assert [tuple(p) for p in out["align.dtw_s2t"]] == ref.dtw_alignment(stu_tok, tea_tok, ref.dist_fn_edit)
    assert [tuple(p) for p in out["align.dtw_t2s"]] == ref.dtw_alignment(tea_tok, stu_tok, ref.dist_fn_edit)
    assert out["align.dtw_s2t_cost"] == [
        float(editdistance.eval(stu_tok[i], tea_tok[j])) for i, j in out["align.dtw_s2t"]
    ]
    assert len(out["align.dtw_t2s_cost"]) == len(out["align.dtw_t2s"])


# --------------------------------------------------------------------------- lcs_span
def test_lcs_span_parity(xtok_samples, xtok_ctx):
    ref = _ref()
    al = build("lcs_span")
    for s in xtok_samples:
        out = al(s, xtok_ctx)
        s_pool, t_pool = ref.prepare_pooler_v2(
            [s["student.prompt_len"]],
            [torch.tensor(s["student.offsets"])],
            [s["teacher.prompt_len"]],
            [torch.tensor(s["teacher.offsets"])],
        )
        for mine, pool in ((out["align.sra_s_idx"], s_pool), (out["align.sra_t_idx"], t_pool)):
            exp = pool["safe_idx"][0].masked_fill(~pool["mask"][0], -1)
            assert torch.equal(torch.tensor(mine), exp)
    n_spans = [len(al(s, xtok_ctx)["align.sra_s_idx"]) for s in xtok_samples[:4]]
    assert max(n_spans) > 1


def test_lcs_span_n_span_max(xtok_samples, xtok_ctx):
    ref = _ref()
    s = xtok_samples[0]
    out = build({"type": "lcs_span", "n_span_max": 2})(s, xtok_ctx)
    assert len(out["align.sra_s_idx"]) <= 2
    ref.N_SPAN = 2
    try:
        pool, _ = ref.prepare_pooler_v2(
            [s["student.prompt_len"]],
            [torch.tensor(s["student.offsets"])],
            [s["teacher.prompt_len"]],
            [torch.tensor(s["teacher.offsets"])],
        )
    finally:
        ref.N_SPAN = 1024
    assert torch.equal(torch.tensor(out["align.sra_s_idx"]), pool["safe_idx"][0].masked_fill(~pool["mask"][0], -1))


# --------------------------------------------------------------------------- parent_groups
def test_parent_groups_hand_example(xtok_ctx):
    # response "abcd": student tokens ab|c|d, teacher tokens a|bc|d -> shared boundaries 0, 3?, 4
    s = {
        "student.input_ids": [9, 1, 2, 3, 4],
        "student.prompt_len": 1,
        "teacher.input_ids": [9, 5, 6, 7, 8],
        "teacher.prompt_len": 1,
        "student.offsets": [[0, 0], [0, 2], [2, 3], [3, 4], [4, 4]],
        "teacher.offsets": [[0, 0], [0, 1], [1, 3], [3, 4], [4, 4]],
        "meta.prompt_text": "",
    }
    out = build("parent_groups")(s, xtok_ctx)
    assert out["align.parent_groups_s"] == [[1, 2], [3, 4]]
    assert out["align.parent_groups_t"] == [[1, 2], [3, 4]]


def test_parent_groups_real_sample(xtok_samples, xtok_ctx):
    s = xtok_samples[0]
    out = build("parent_groups")(s, xtok_ctx)
    g_s, g_t = np.asarray(out["align.parent_groups_s"]), np.asarray(out["align.parent_groups_t"])
    assert g_s.shape[0] == g_t.shape[0] and g_s.ndim == 2
    flat = g_s[g_s >= 0]
    assert sorted(flat.tolist()) == list(range(s["student.prompt_len"], len(s["student.input_ids"])))
    flat_t = g_t[g_t >= 0]
    assert sorted(flat_t.tolist()) == list(range(s["teacher.prompt_len"], len(s["teacher.input_ids"])))


# --------------------------------------------------------------------------- tokenkit_chunk
def _tokenkit():
    try:
        import tokenkit  # noqa: F401
    except ImportError:
        src = Path(__file__).resolve().parents[3] / "white-box-distillation/cross-tokenizer/IMPACT/tokenkit-main"
        if not src.exists():
            pytest.skip("tokenkit not installed")
        sys.path.insert(0, str(src))
    return pytest.importorskip("tokenkit.align")


class _FakeKind:
    replacements: dict = {"bos": None, "eos": None}


class _FakeByteify:
    """Byteify-like view: ``Ġ`` prefix on every token but the first, like tokenkit does for word starts."""

    model_kind_cls = _FakeKind

    def __init__(self, tok, word_level):
        self.tok, self.word_level = tok, word_level

    def convert_ids_to_tokens(self, ids):
        toks = self.tok.convert_ids_to_tokens([int(i) for i in ids])
        if not self.word_level:
            return toks
        return [t if k == 0 else "Ġ" + t for k, t in enumerate(toks)]


def test_tokenkit_chunk_one_chunk_per_token(xtok_samples, xtok_ctx):
    align = _tokenkit()
    stu, tea = xtok_ctx.student_tokenizer, xtok_ctx.teacher_tokenizer
    fake = (_FakeByteify(stu, True), _FakeByteify(tea, False))
    al = build({"type": "tokenkit_chunk", "student_byteify": "x", "teacher_byteify": "y"})
    al._tok = fake
    got_any = False
    for s in xtok_samples[:8]:
        out = al(s, xtok_ctx)
        s_ids, t_ids = s["student.input_ids"][:-1], s["teacher.input_ids"][:-1]
        ms, mt = align.get_unconstrained_alignments(
            np.asarray([t_ids]),
            np.asarray([s_ids]),
            np.ones((1, len(t_ids)), dtype=np.int64),
            np.ones((1, len(s_ids)), dtype=np.int64),
            fake[1],
            fake[0],
        )
        assert not (ms.sum(-1) > 1).any() and not (mt.sum(-1) > 1).any()
        exp_s = [int(r.argmax()) if r.any() else -1 for r in ms[0]]
        exp_t = [int(r.argmax()) if r.any() else -1 for r in mt[0]]
        assert out["align.chunk_s"] == exp_s + [-1]  # exclude_last: last position never in a chunk
        assert out["align.chunk_t"] == exp_t + [-1]
        got_any |= max(exp_s) >= 0
    assert got_any


# --------------------------------------------------------------------------- field specs
@pytest.mark.parametrize(
    "key,kind,dtype,pad,side",
    [
        ("align.eta_t_for_s", "token", torch.long, -1, "student"),
        ("align.eta_s_for_t", "token", torch.long, -1, "teacher"),
        ("align.dtw_t_for_s", "token", torch.long, -1, "student"),
        ("align.dtw_s2t", "ragged", torch.long, 0, None),
        ("align.dtw_s2t_cost", "ragged", torch.float32, 0.0, None),
        ("align.dtw_t2s", "ragged", torch.long, 0, None),
        ("align.dtw_t2s_cost", "ragged", torch.float32, 0.0, None),
        ("align.span_pairs", "ragged", torch.long, 0, None),
        ("align.sra_s_idx", "ragged", torch.long, -1, None),
        ("align.sra_t_idx", "ragged", torch.long, -1, None),
        ("align.impact_lo", "token_k", torch.long, 0, "student"),
        ("align.impact_hi", "token_k", torch.long, 0, "student"),
        ("align.impact_w", "token_k", torch.float32, 0.0, "student"),
        ("align.chunk_s", "token", torch.long, -1, "student"),
        ("align.chunk_t", "token", torch.long, -1, "teacher"),
        ("align.parent_groups_s", "ragged", torch.long, -1, None),
        ("student.offsets", "token_k", torch.long, -1, "student"),
        ("teacher.offsets", "token_k", torch.long, -1, "teacher"),
    ],
)
def test_field_specs(key, kind, dtype, pad, side):
    spec = FIELDS.get(key)
    assert (spec.kind, spec.dtype, spec.pad_value, spec.side) == (kind, dtype, pad, side)


def test_seven_aligners_stable():
    from lmkd.core.registry import ALIGNERS as REG

    names = {"exact_token", "dtw_edit", "char_span", "lcs_span", "boundary_interp", "tokenkit_chunk", "parent_groups"}
    assert names <= set(REG)
    assert all(REG.meta(n).status == "stable" for n in names)
