from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from baselines.longCoT import RenderCoT, Segment, TokenizeCoT
from baselines.longCoT.sgl_spectral import AllocateRegion, Provenance, SelectEnergyTopP, StepTransitions
from lmkd.data.context import DataContext

SGL_ROOT = Path(__file__).resolve().parents[3] / "Long-CoT-Disitllation" / "SpectralGuidedLearning"
SGL_SRC = SGL_ROOT / "src"
ASSETS_DIR = SGL_ROOT / "tests/regression/assets"

pytestmark = [
    pytest.mark.parity,
    pytest.mark.skipif(not SGL_SRC.exists(), reason="SGL repo not found"),
]

if SGL_SRC.exists() and str(SGL_SRC) not in sys.path:
    sys.path.insert(0, str(SGL_SRC))


def _load_sgl_tokenizer(tok_path: Path):
    try:
        return AutoTokenizer.from_pretrained(str(tok_path))
    except Exception:
        return PreTrainedTokenizerFast.from_pretrained(str(tok_path))


def test_build_record_parity():
    from sgl.data.prepare import build_record

    tok_path = ASSETS_DIR / "tokenizer"
    if not tok_path.exists():
        pytest.skip("SGL tokenizer asset not found")
    tokenizer = _load_sgl_tokenizer(tok_path)
    ctx = DataContext(student_tokenizer=tokenizer, max_length=4096)

    sample_file = ASSETS_DIR / "palign_sample.jsonl"
    if not sample_file.exists():
        pytest.skip("palign_sample.jsonl not found")

    with open(sample_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:5]

    render_t = RenderCoT(
        RenderCoT.Config(problem_field="problem", response_fields=["response"], thinking_mode="on", chat_template=True)
    )
    tokenize_t = TokenizeCoT(TokenizeCoT.Config())
    segment_t = Segment(Segment.Config(segmenter="sentence"))

    for i, row in enumerate(rows):
        row = dict(row)
        row.setdefault("id", i)
        problem = row.get("problem") or row.get("input")
        response = row.get("response") or row.get("output")
        ref_record = build_record(
            tokenizer,
            problem,
            response,
            max_tokens=4096,
            chat_template=True,
            enable_thinking=True,
            segmenter="sentence",
        )
        ref_record["id"] = row["id"]
        assert ref_record is not None

        s = dict(row)
        s = render_t(s, ctx)
        s = tokenize_t(s, ctx)
        s = segment_t(s, ctx)

        assert s["student.input_ids"] == ref_record["input_ids"]
        assert list(s["seq.response_span"]) == ref_record["response_token_span"]
        ref_spans = [(step["token_start"], step["token_end"]) for step in ref_record["steps"]]
        assert list(s["seq.steps"]) == ref_spans


def test_masks_weights_parity():
    from sgl.allocation.build import build_loss_mask, record_step_spans, select_steps_by_energy
    from sgl.data.prepare import build_record

    tok_path = ASSETS_DIR / "tokenizer"
    if not tok_path.exists():
        pytest.skip("SGL tokenizer asset not found")
    tokenizer = _load_sgl_tokenizer(tok_path)
    ctx = DataContext(student_tokenizer=tokenizer, max_length=4096)

    sample_file = ASSETS_DIR / "palign_sample.jsonl"
    with open(sample_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:5]

    render_t = RenderCoT(
        RenderCoT.Config(problem_field="problem", response_fields=["response"], thinking_mode="on", chat_template=True)
    )
    tokenize_t = TokenizeCoT(TokenizeCoT.Config())
    segment_t = Segment(Segment.Config(segmenter="sentence"))
    select_t = SelectEnergyTopP(SelectEnergyTopP.Config(p=0.95))

    for i, row in enumerate(rows):
        row = dict(row)
        row.setdefault("id", i)
        rng = random.Random(1000 + row["id"])
        problem = row.get("problem") or row.get("input")
        response = row.get("response") or row.get("output")
        ref_record = build_record(
            tokenizer,
            problem,
            response,
            max_tokens=4096,
            chat_template=True,
            enable_thinking=True,
            segmenter="sentence",
        )
        ref_record["id"] = row["id"]
        n_steps = len(ref_record["steps"])
        strengths = [round(rng.uniform(0.0, 1.0), 6) for _ in range(n_steps)]
        entropies = [round(rng.uniform(0.0, 3.0), 6) for _ in range(n_steps)]

        step_spans = record_step_spans(ref_record)
        ref_selected = select_steps_by_energy(strengths, 0.95)
        ref_loss_mask = build_loss_mask(len(ref_record["input_ids"]), step_spans, ref_selected)
        for position in range(ref_record["response_token_span"][1], len(ref_loss_mask)):
            ref_loss_mask[position] = 1

        s = dict(row)
        s["signals.step_strength"] = strengths
        s["signals.step_entropy"] = entropies
        s = render_t(s, ctx)
        s = tokenize_t(s, ctx)
        s = segment_t(s, ctx)
        s = select_t(s, ctx)

        assert s["seq.loss_mask"] == ref_loss_mask


def test_provenance_transitions_parity():
    from sgl.data.prepare import build_record
    from sgl.transforms.provenance import prefix_token_count
    from sgl.transforms.step_transitions import build_transition_fields

    tok_path = ASSETS_DIR / "tokenizer"
    if not tok_path.exists():
        pytest.skip("SGL tokenizer asset not found")
    tokenizer = _load_sgl_tokenizer(tok_path)
    ctx = DataContext(student_tokenizer=tokenizer, max_length=4096)

    sample_file = ASSETS_DIR / "palign_sample.jsonl"
    with open(sample_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:5]

    render_t = RenderCoT(
        RenderCoT.Config(problem_field="problem", response_fields=["response"], thinking_mode="on", chat_template=True)
    )
    tokenize_t = TokenizeCoT(TokenizeCoT.Config())
    segment_t = Segment(Segment.Config(segmenter="sentence"))
    select_t = SelectEnergyTopP(SelectEnergyTopP.Config(p=0.95))
    prov_t = Provenance(Provenance.Config(junction_tokens=2, marker="<End_of_Prefix>"))
    trans_t = StepTransitions(StepTransitions.Config(min_step_tokens=8))

    for i, row in enumerate(rows):
        row = dict(row)
        row.setdefault("id", i)
        rng = random.Random(1000 + row["id"])
        # Inject marker into response if absent to test junction
        problem = row.get("problem") or row.get("input")
        raw_response = (row.get("response") or row.get("output")).replace(". ", ".\n\n<End_of_Prefix>\n\n", 1)
        row["response"] = raw_response
        ref_record = build_record(
            tokenizer,
            problem,
            raw_response,
            max_tokens=4096,
            chat_template=True,
            enable_thinking=True,
            segmenter="sentence",
        )
        ref_record["id"] = row["id"]
        n_steps = len(ref_record["steps"])
        strengths = [round(rng.uniform(0.0, 1.0), 6) for _ in range(n_steps)]

        s = dict(row)
        s["signals.step_strength"] = strengths
        s = render_t(s, ctx)
        s = tokenize_t(s, ctx)
        s = segment_t(s, ctx)
        s = select_t(s, ctx)
        s = prov_t(s, ctx)
        s = trans_t(s, ctx)

        start, end = ref_record["response_token_span"]
        ref_n_prefix = prefix_token_count(tokenizer, ref_record["response"], ref_record["input_ids"][start:end])
        assert s["meta.prefix_boundary"] == start + (ref_n_prefix or 0)

        ref_trans = build_transition_fields(
            ref_record["response"],
            s["meta.token_starts"],
            start,
            len(ref_record["input_ids"]),
            min_step_tokens=8,
        )
        assert s["seq.step_id"] == ref_trans["step_id"]
        assert list(s["seq.step_end"]) == ref_trans["step_end"]
        assert list(s["seq.pair_src"]) == ref_trans["pair_src"]


def _prepared(tokenizer, ctx, n=6):
    """Records with a mid-response <End_of_Prefix> marker + deterministic signals (SGL regression style)."""

    with open(ASSETS_DIR / "palign_sample.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:n]
    t_render = RenderCoT(RenderCoT.Config())
    t_tok, t_seg = TokenizeCoT(TokenizeCoT.Config()), Segment(Segment.Config())
    t_prov = Provenance(Provenance.Config())
    t_sel = SelectEnergyTopP(SelectEnergyTopP.Config(p=0.95))
    raws, recs = [], []
    for i, row in enumerate(rows):
        resp = row.get("response") or row.get("output")
        cut = resp.find(". ", len(resp) // 3)
        resp = resp[: cut + 2] + "\n\n<End_of_Prefix>\n\n" + resp[cut + 2 :]
        problem = row.get("problem") or row.get("input")
        s = {"problem": problem, "response": resp, "meta.id": str(i)}
        s = t_seg(t_tok(t_render(s, ctx), ctx), ctx)
        rng = random.Random(i)
        k = len(s["seq.steps"])
        s["signals.step_strength"] = [rng.uniform(0.1, 1) for _ in range(k)]
        s["signals.step_entropy"] = [rng.uniform(0, 3) for _ in range(k)]
        s["signals.step_gain"] = [rng.uniform(0.1, 1) for _ in range(k)]
        s = t_prov(s, ctx)  # prefix_boundary (+ token_source over all response tokens)
        s = t_sel(s, ctx)
        raws.append((problem, resp, i))
        recs.append(s)
    return raws, recs


@pytest.mark.parametrize(
    "sgl_name,mine",
    [
        ("region-gain", "region-gain"),
        ("sarw", "sarw"),
        ("region-cont", "region-cont"),
    ],
)
def test_all_allocation_variants_parity(sgl_name, mine):
    from sgl.allocation.build import build_example
    from sgl.allocation.region import region_gain_ratio, step_regions
    from sgl.data.prepare import build_record

    tokenizer = _load_sgl_tokenizer(ASSETS_DIR / "tokenizer")
    ctx = DataContext(student_tokenizer=tokenizer, max_length=4096)
    raws, recs = _prepared(tokenizer, ctx)
    regs = [step_regions([tuple(x) for x in s["seq.steps"]], s["meta.prefix_boundary"]) for s in recs]
    assert any("P" in r and "C" in r for r in regs)  # the marker really splits the steps
    t = AllocateRegion(AllocateRegion.Config(variant=mine, gamma=1.0, tail_fraction=0.1))
    t.fit([dict(s) for s in recs], ctx)  # corpus-level fit, as run_transforms does
    _, _, ratio = region_gain_ratio(
        [s["signals.step_gain"] for s in recs], [[e - b for b, e in s["seq.steps"]] for s in recs], regs, 1.0, 0.1
    )
    assert t.ratio == pytest.approx(ratio) if mine in ("sarw", "region-gain") else True
    for i, (s, (problem, resp, sid)) in enumerate(zip(recs, raws)):
        ref = build_record(tokenizer, problem, resp, 4096, chat_template=True, enable_thinking=True)
        ref["id"] = sid
        want, _ = build_example(
            ref,
            s["signals.step_strength"],
            s["signals.step_entropy"],
            0.95,
            sgl_name,
            2.0,
            0.5,
            2.0,
            1e-8,
            42,
            regs[i],
            ratio,
        )
        out = t(dict(s), ctx)
        assert out["seq.loss_mask"] == want["loss_mask"], (sgl_name, i)
        assert all(abs(a - b) < 1e-9 for a, b in zip(out["seq.token_weights"], want["loss_weights"])), (sgl_name, i)


def test_token_source_parity():
    from sgl.data.prepare import build_record
    from sgl.transforms.provenance import prefix_token_count

    tokenizer = _load_sgl_tokenizer(ASSETS_DIR / "tokenizer")
    ctx = DataContext(student_tokenizer=tokenizer, max_length=4096)
    raws, recs = _prepared(tokenizer, ctx, n=4)
    for J in (0, 3):
        prov = Provenance(Provenance.Config(junction_tokens=J))
        for s, (problem, resp, _) in zip(recs, raws):
            ref = build_record(tokenizer, problem, resp, 4096, chat_template=True, enable_thinking=True)
            start, end = ref["response_token_span"]
            n_prefix = prefix_token_count(tokenizer, ref["response"], ref["input_ids"][start:end]) or 0
            # SGL provenance.build, applied to the same loss_mask (labels != -100 also holds on every response token)
            want = [0] * len(ref["input_ids"])
            for pos, keep in enumerate(s["seq.loss_mask"]):
                if keep:
                    want[pos] = 1 if pos < start + n_prefix else (3 if n_prefix and pos < start + n_prefix + J else 2)
            assert prov(dict(s), ctx)["seq.token_source"] == want
