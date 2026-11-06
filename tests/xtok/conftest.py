"""Cross-tokenizer fixtures: student = WordLevel (tests/conftest), teacher = byte-level BPE trained on the toy corpus."""

from __future__ import annotations

import json

import pytest

from lmkd.core.config import ModelSpec
from lmkd.core.registry import TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import run_transforms
from lmkd.models.loading import load_tokenizer


@pytest.fixture(scope="session")
def xtok_models(tmp_path_factory, tiny_models, toy_jsonl):
    from tests.conftest import _make_gpt2
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    rows = [json.loads(line) for line in open(toy_jsonl)]
    corpus = [f"{r['prompt']} {r['response']}" for r in rows] + [r["response"] for r in rows]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=120, special_tokens=["<pad>", "<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tk.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", pad_token="<pad>")
    root = tmp_path_factory.mktemp("xtok")
    path = root / "teacher_x"
    fast.save_pretrained(path)
    _make_gpt2(path, 48, 3, len(fast), 2)
    return {"student": tiny_models["student"], "teacher": str(path)}


@pytest.fixture(scope="session")
def xtok_tokenizers(xtok_models):
    return (
        load_tokenizer(ModelSpec(path=xtok_models["student"])),
        load_tokenizer(ModelSpec(path=xtok_models["teacher"])),
    )


@pytest.fixture(scope="session")
def xtok_ctx(xtok_tokenizers):
    stu, tea = xtok_tokenizers
    return DataContext(student_tokenizer=stu, teacher_tokenizer=tea, max_length=64)


@pytest.fixture(scope="session")
def xtok_records(toy_jsonl):
    return [json.loads(line) for line in open(toy_jsonl)]


@pytest.fixture(scope="session")
def xtok_samples(xtok_records, xtok_ctx):
    """32 toy samples: tokenize_sft(teacher, same prompt text) + token_offsets(teacher)."""
    tfs = [
        TRANSFORMS.build(
            {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"}
        ),
        TRANSFORMS.build({"type": "token_offsets", "teacher": True}),
    ]
    out = run_transforms(xtok_records, tfs, xtok_ctx)
    assert len(out) == len(xtok_records)
    return out
