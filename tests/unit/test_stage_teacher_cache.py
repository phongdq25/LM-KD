"""Offline target-aligned preference teacher caches."""

from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from lmkd.data.context import DataContext
from lmkd.data.transforms.listwise import TokenizeListwise, tokenize_fingerprint
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl, write_jsonl
from lmkd.pipeline.stage import StageContext
from lmkd.pipeline.stages.preference import TeacherCache


def _case(tiny_models, tmp_path):
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    transform = TokenizeListwise(TokenizeListwise.Config(format_plain_prompts=False))
    sample = transform(
        {"meta.id": "p1", "prompt": "Q: w1", "responses": ["A: w2", "A: w3"]}, DataContext(tok, max_length=32)
    )
    incoming = tmp_path / "in"
    incoming.mkdir()
    write_jsonl(incoming / "samples.jsonl", [sample])
    ctx = StageContext(
        rt=SimpleNamespace(student_tokenizer=tok),
        inputs={"data": Artifact("prepare_dataset", "k", incoming, "in")},
        output=Artifact("teacher_cache", "k", tmp_path / "out", "out"),
    )
    return ctx


def test_teacher_cache_topk_batch_and_resume(tiny_models, tmp_path):
    ctx = _case(tiny_models, tmp_path)
    stage = TeacherCache(TeacherCache.Config(model=tiny_models["teacher"], dtype="fp32", k=4, batch_size=2, fields=[0]))
    first = stage.run(ctx)
    second = stage.run(ctx)
    rows = read_jsonl(ctx.output.path / "samples.jsonl")
    chosen, rejected = rows[0]["group.members"]
    assert first["computed"] == 1 and second["computed"] == 0
    assert first["tokenize_fingerprint"] == second["tokenize_fingerprint"]
    assert len(chosen["cache.topk_ids"]) == len(chosen["student.input_ids"])
    assert len(chosen["cache.topk_ids"][0]) == 4
    assert any(sum(p) > 0 for p in chosen["cache.topk_p"])
    assert all(sum(p) == 0 for p in rejected["cache.topk_p"])
    assert all(
        row["meta.cache_tokenize_fp"]
        == tokenize_fingerprint(
            row["student.input_ids"],
            row["student.labels"],
        )
        for row in (chosen, rejected)
    )


def test_teacher_cache_adv_sparse(tiny_models, tmp_path):
    ctx = _case(tiny_models, tmp_path)
    stage = TeacherCache(
        TeacherCache.Config(
            model=tiny_models["teacher"],
            reference_model=tiny_models["student"],
            dtype="fp32",
            kind="adv_sparse",
            k=3,
            adv_k_max=6,
            batch_size=2,
            fields=[1],
        )
    )
    stage.run(ctx)
    row = read_jsonl(ctx.output.path / "samples.jsonl")[0]["group.members"][1]
    ids = torch.tensor(row["cache.adv_ids"])
    vals = torch.tensor(row["cache.adv_val"])
    assert ids.shape[-1] == 6
    assert torch.isfinite(vals).all()
    assert (ids >= 0).any()


def test_teacher_cache_token_logp(tiny_models, tmp_path):
    ctx = _case(tiny_models, tmp_path)
    stage = TeacherCache(
        TeacherCache.Config(model=tiny_models["teacher"], dtype="fp32", kind="token_logp", batch_size=2)
    )
    stage.run(ctx)
    rows = read_jsonl(ctx.output.path / "samples.jsonl")[0]["group.members"]
    assert all(len(row["cache.token_logp"]) == len(row["student.labels"]) for row in rows)
    assert all(any(x < 0 for x in row["cache.token_logp"]) for row in rows)
