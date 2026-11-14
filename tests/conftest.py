"""Shared fixtures: a tiny word-level tokenizer and tiny random GPT-2 student/teacher saved to a
temp dir, plus a toy jsonl dataset. Everything runs on CPU in a few seconds, no network."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", str(Path(tempfile.gettempdir()) / "triton_cache"))

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from lmkd.plugins import load_plugins


@pytest.fixture(scope="session", autouse=True)
def _auto_load_plugins():
    load_plugins([])


WORDS = [f"w{i}" for i in range(60)] + ["Q:", "A:", "+", "="]


def _make_tokenizer(path, eos_is_pad: bool = False):
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    for w in WORDS:
        vocab[w] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token="<eos>", unk_token="<unk>", pad_token="<eos>" if eos_is_pad else "<pad>"
    )
    fast.save_pretrained(path)
    return fast


def _make_gpt2(path, n_embd: int, n_layer: int, vocab: int, seed: int):
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=vocab, n_positions=128, n_embd=n_embd, n_layer=n_layer, n_head=2, bos_token_id=1, eos_token_id=1
    )
    m = GPT2LMHeadModel(cfg)
    m.save_pretrained(path)
    return m


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    tok = _make_tokenizer(root / "student")
    _make_gpt2(root / "student", 32, 2, len(tok), 0)
    _make_tokenizer(root / "teacher")
    _make_gpt2(root / "teacher", 48, 3, len(tok), 1)
    return {"student": str(root / "student"), "teacher": str(root / "teacher"), "root": root}


@pytest.fixture(scope="session")
def toy_jsonl(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "train.jsonl"
    rows = []
    for i in range(32):
        a, b = i % 7, (i * 3) % 11
        rows.append({"id": f"r{i}", "prompt": f"Q: w{a} + w{b} =", "response": f"A: w{a + b} w{a} w{b}"})
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


@pytest.fixture
def base_recipe(tiny_models, toy_jsonl, tmp_path):
    """Minimal recipe dict (SFT). Tests modify it."""
    return {
        "models": {"student": {"path": tiny_models["student"], "dtype": "fp32"}},
        "data": {
            "train": {"type": "jsonl", "path": toy_jsonl},
            "transforms": [{"type": "tokenize_sft", "use_chat_template": False}],
            "max_length": 64,
        },
        "loss": {"terms": [{"criterion": "nll"}]},
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "optim": {"lr": 1e-3},
        "train": {
            "output_dir": str(tmp_path / "run"),
            "max_steps": 4,
            "batch_size": 8,
            "micro_batch_size": 4,
            "log_every": 2,
        },
    }
