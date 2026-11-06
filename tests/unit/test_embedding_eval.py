from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lmkd.eval.embedding import EmbeddingLinearProbe, EmbeddingPair, EmbeddingSTS
from lmkd.models.adapters import HFAutoAdapter
from lmkd.plugins import load_builtin_plugins

load_builtin_plugins()


class DummyEmbeddingModel(nn.Module):
    """Returns fixed embeddings for testing metrics."""

    def __init__(self, emb_map: dict[str, torch.Tensor], hidden_size: int = 16):
        super().__init__()
        self.emb_map = emb_map
        self.hidden_size = hidden_size
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Infer text from input_ids or return dummy hidden_state
        B = input_ids.shape[0]
        # Use first token id as key
        tensors = []
        for b in range(B):
            first_id = int(input_ids[b, 0])
            tensors.append(self.emb_map.get(first_id, torch.zeros(self.hidden_size)))
        last_hidden = torch.stack(tensors).unsqueeze(1)  # [B, 1, D]
        return SimpleNamespace(last_hidden_state=last_hidden)


class DummyTokenizer:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.pad_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors=None):
        input_ids = []
        for t in texts:
            tid = self.token_to_id.get(t, 0)
            input_ids.append([tid])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 1), dtype=torch.long),
        }


def test_sts_perfect():
    pytest.importorskip("scipy")

    # Vector 1, Vector 2
    # Pair 0: high similarity -> high score (5.0)
    # Pair 1: medium similarity -> medium score (3.0)
    # Pair 2: low similarity -> low score (1.0)
    emb_map = {
        10: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        11: torch.tensor([1.0, 0.0, 0.0, 0.0]),  # cos = 1.0 -> score 5.0
        20: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        21: torch.tensor([0.7071, 0.7071, 0.0, 0.0]),  # cos = 0.7071 -> score 3.0
        30: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        31: torch.tensor([0.0, 1.0, 0.0, 0.0]),  # cos = 0.0 -> score 1.0
    }
    tok_map = {
        "text_10": 10,
        "text_11": 11,
        "text_20": 20,
        "text_21": 21,
        "text_30": 30,
        "text_31": 31,
    }

    records = [
        {"sentence1": "text_10", "sentence2": "text_11", "score": 5.0},
        {"sentence1": "text_20", "sentence2": "text_21", "score": 3.0},
        {"sentence1": "text_30", "sentence2": "text_31", "score": 1.0},
    ]

    model = DummyEmbeddingModel(emb_map, hidden_size=4)
    tok = DummyTokenizer(tok_map)

    rt = SimpleNamespace(
        student_model=model,
        student_tokenizer=tok,
        student_adapter=HFAutoAdapter(),
        engine=SimpleNamespace(device="cpu"),
    )

    evaluator = EmbeddingSTS(
        EmbeddingSTS.Config(
            data={"records": records},
            pooler="cls",
            name="test",
        )
    )
    evaluator.setup(rt)
    metrics = evaluator.evaluate(rt)

    assert "sts/test/spearman" in metrics
    assert pytest.approx(metrics["sts/test/spearman"], abs=1e-4) == 1.0


def test_pair_ap():
    pytest.importorskip("sklearn")

    # Positives have high similarity, negatives have low similarity
    emb_map = {
        10: torch.tensor([1.0, 0.0]),
        11: torch.tensor([0.9, 0.1]),  # cos high -> label 1
        20: torch.tensor([1.0, 0.0]),
        21: torch.tensor([0.8, 0.2]),  # cos high -> label 1
        30: torch.tensor([1.0, 0.0]),
        31: torch.tensor([0.0, 1.0]),  # cos low -> label 0
        40: torch.tensor([1.0, 0.0]),
        41: torch.tensor([-1.0, 0.0]),  # cos low -> label 0
    }
    tok_map = {
        "text_10": 10,
        "text_11": 11,
        "text_20": 20,
        "text_21": 21,
        "text_30": 30,
        "text_31": 31,
        "text_40": 40,
        "text_41": 41,
    }
    records = [
        {"sentence1": "text_10", "sentence2": "text_11", "label": 1},
        {"sentence1": "text_20", "sentence2": "text_21", "label": 1},
        {"sentence1": "text_30", "sentence2": "text_31", "label": 0},
        {"sentence1": "text_40", "sentence2": "text_41", "label": 0},
    ]

    model = DummyEmbeddingModel(emb_map, hidden_size=2)
    tok = DummyTokenizer(tok_map)
    rt = SimpleNamespace(
        student_model=model,
        student_tokenizer=tok,
        student_adapter=HFAutoAdapter(),
        engine=SimpleNamespace(device="cpu"),
    )

    evaluator = EmbeddingPair(
        EmbeddingPair.Config(
            data={"records": records},
            pooler="cls",
            name="test",
        )
    )
    evaluator.setup(rt)
    metrics = evaluator.evaluate(rt)

    assert "pair/test/ap" in metrics
    assert pytest.approx(metrics["pair/test/ap"], abs=1e-4) == 1.0


def test_linear_probe():
    pytest.importorskip("sklearn")

    emb_map = {
        1: torch.tensor([10.0, 0.0]),
        2: torch.tensor([11.0, 1.0]),
        3: torch.tensor([-10.0, 0.0]),
        4: torch.tensor([-11.0, 1.0]),
    }
    tok_map = {"c1_a": 1, "c1_b": 2, "c2_a": 3, "c2_b": 4}
    train_records = [
        {"text": "c1_a", "label": 0},
        {"text": "c2_a", "label": 1},
    ]
    test_records = [
        {"text": "c1_b", "label": 0},
        {"text": "c2_b", "label": 1},
    ]

    model = DummyEmbeddingModel(emb_map, hidden_size=2)
    tok = DummyTokenizer(tok_map)
    rt = SimpleNamespace(
        student_model=model,
        student_tokenizer=tok,
        student_adapter=HFAutoAdapter(),
        engine=SimpleNamespace(device="cpu"),
    )

    evaluator = EmbeddingLinearProbe(
        EmbeddingLinearProbe.Config(
            train={"records": train_records},
            test={"records": test_records},
            pooler="cls",
            name="test",
        )
    )
    evaluator.setup(rt)
    metrics = evaluator.evaluate(rt)

    assert "cls/test/macro_f1" in metrics
    assert pytest.approx(metrics["cls/test/macro_f1"], abs=1e-4) == 1.0


def test_pure_metric_functions_without_runtime():
    pytest.importorskip("scipy")
    from lmkd.eval.embedding import encode_texts, sts_spearman

    emb_map = {
        10: torch.tensor([1.0, 0.0]),
        11: torch.tensor([1.0, 0.0]),
        20: torch.tensor([1.0, 0.0]),
        21: torch.tensor([0.0, 1.0]),
    }
    tok = DummyTokenizer({"a": 10, "b": 11, "c": 20, "d": 21})
    model = DummyEmbeddingModel(emb_map, hidden_size=2)
    assert encode_texts(model, tok, ["a", "d"], "cls").tolist() == [[1.0, 0.0], [0.0, 1.0]]
    records = [{"sentence1": "a", "sentence2": "b", "score": 5.0}, {"sentence1": "c", "sentence2": "d", "score": 1.0}]
    assert sts_spearman(model, tok, records, pooler="cls") == {"spearman": pytest.approx(1.0)}
