"""Listwise preference rows and group-preserving batches."""

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from lmkd.core.batch import Batch
from lmkd.core.config import LossTermSpec, load_recipe
from lmkd.core.validate import check_recipe
from lmkd.data.collator import Collator
from lmkd.data.context import DataContext
from lmkd.data.dataset import PreparedDataset
from lmkd.data.producers_listwise import ListwiseProducer
from lmkd.data.transforms.listwise import TokenizeListwise


def _context(tiny_models):
    return DataContext(AutoTokenizer.from_pretrained(tiny_models["student"]), max_length=32)


def _sample(ctx, sid="x", n=2):
    responses = [f"A: w{i}" for i in range(n)]
    return TokenizeListwise(TokenizeListwise.Config(format_plain_prompts=False))(
        {"meta.id": sid, "prompt": "Q: w1", "responses": responses},
        ctx,
    )


def _runtime(samples, ctx, micro=4, batch=8):
    return SimpleNamespace(
        train_dataset=PreparedDataset(samples),
        collator=Collator({"student": ctx.student_tokenizer.pad_token_id}, pad_to_multiple_of=None),
        config=SimpleNamespace(
            train=SimpleNamespace(batch_size=batch, micro_batch_size=micro, seed=1), data=SimpleNamespace(num_workers=0)
        ),
        engine=SimpleNamespace(world_size=1, rank=0),
    )


def test_tokenize_listwise_rows(tiny_models):
    ctx = _context(tiny_models)
    rows = _sample(ctx)["group.members"]
    assert [row["group.index"] for row in rows] == [0, 1]
    assert all(row["group.size"] == 2 for row in rows)
    assert all(row["student.labels"][: row["student.prompt_len"]] == [-100] * row["student.prompt_len"] for row in rows)
    assert all(row["student.input_ids"][-1] == ctx.student_tokenizer.eos_token_id for row in rows)


def test_left_truncation(tiny_models):
    ctx = _context(tiny_models)
    cfg = TokenizeListwise.Config(max_prompt_length=3, max_length=8, format_plain_prompts=False)
    row = TokenizeListwise(cfg)({"prompt": "w0 w1 w2 w3 w4", "chosen": " w5", "rejected": " w6"}, ctx)
    assert row is not None
    assert (
        row["group.members"][0]["student.input_ids"][:3]
        == ctx.student_tokenizer(
            "w2 w3 w4",
            add_special_tokens=False,
        )["input_ids"]
    )


def test_producer_flatten_and_split(tiny_models):
    ctx = _context(tiny_models)
    producer = ListwiseProducer(ListwiseProducer.Config(shuffle=False))
    producer.setup(_runtime([_sample(ctx, str(i)) for i in range(4)], ctx))
    batch = producer.next(SimpleNamespace(epoch=0))
    assert batch.batch_size == 8
    assert batch["group.index"].tolist() == [0, 1] * 4
    assert all(batch.meta["group.uid"][i] == batch.meta["group.uid"][i + 1] for i in range(0, 8, 2))
    assert [part.batch_size for part in batch.split(2)] == [4, 4]


def test_producer_rejects_bad_micro(tiny_models):
    ctx = _context(tiny_models)
    producer = ListwiseProducer(ListwiseProducer.Config(shuffle=False))
    with pytest.raises(ValueError, match="micro_batch_size"):
        producer.setup(_runtime([_sample(ctx)], ctx, micro=3))


def test_split_guard_from_core():
    batch = Batch({"group.index": torch.tensor([0, 1, 0, 1])})
    with pytest.raises(ValueError, match="cuts a group"):
        batch.split(4)


def test_padding_groups(tiny_models):
    ctx = _context(tiny_models)
    producer = ListwiseProducer(ListwiseProducer.Config(shuffle=False, pad_groups_to=3))
    producer.setup(_runtime([_sample(ctx, "a", 2), _sample(ctx, "b", 3)], ctx, micro=3, batch=6))
    batch = producer.next(SimpleNamespace(epoch=0))
    assert batch["group.index"].tolist() == [0, 1, 2, 0, 1, 2]
    assert batch["group.valid"].tolist() == [True, True, False, True, True, True]
    assert batch["student.labels"][2].eq(-100).all()
    assert batch["student.attention_mask"][2].any()


@pytest.mark.parametrize("name", ["dckd", "vpd"])
def test_preference_recipe_graph(name):
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "recipes" / "preference" / f"{name}.yaml"
    recipe = load_recipe(path)
    recipe.loss.terms = [LossTermSpec(criterion="nll")]
    report = check_recipe(recipe)
    assert report.ok, report.render()


def test_tokenize_sft_messages_field(tiny_models):
    from lmkd.data.transforms.tokenize import TokenizeSFT

    ctx = _context(tiny_models)
    tok_sft = TokenizeSFT(TokenizeSFT.Config(use_chat_template=False))
    sample = {
        "messages": [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    res = tok_sft(sample, ctx)
    assert res is not None
    assert "student.input_ids" in res
    assert res["meta.response_text"] == "Hi there!"

