import pytest
import torch

from lmkd.core.batch import FIELDS, Batch, FieldSpec
from lmkd.data.collator import Collator
from lmkd.data.labels import make_labels


def test_make_labels_target_aligned():
    lab = make_labels([5, 6, 7, 8, 9], prompt_len=2)
    assert lab.tolist() == [-100, -100, 7, 8, 9]
    assert make_labels([5, 6], prompt_len=0).tolist()[0] == -100  # position 0 never supervised


def test_collator_pads_by_fieldspec():
    FIELDS.register(FieldSpec("test.topk", "token_k", torch.long, -1, "student"), force=True)
    samples = [
        {
            "student.input_ids": [1, 2, 3],
            "student.attention_mask": [1, 1, 1],
            "student.labels": [-100, 2, 3],
            "student.prompt_len": 1,
            "seq.token_weights": [1.0, 2.0, 3.0],
            "test.topk": [[1, 2]] * 3,
            "meta.id": "a",
        },
        {
            "student.input_ids": [4],
            "student.attention_mask": [1],
            "student.labels": [-100],
            "student.prompt_len": 1,
            "seq.token_weights": [5.0],
            "test.topk": [[3, 4]],
            "meta.id": "b",
        },
    ]
    b = Collator({"student": 0}, pad_to_multiple_of=4)(samples)
    assert b["student.input_ids"].shape == (2, 4)
    assert b["student.input_ids"][1].tolist() == [4, 0, 0, 0]
    assert b["student.labels"][1].tolist() == [-100] * 4
    assert b["seq.token_weights"][1].tolist() == [5.0, 1.0, 1.0, 1.0]  # pad value from FieldSpec
    assert b["test.topk"].shape == (2, 4, 2) and b["test.topk"][1, 1].tolist() == [-1, -1]
    assert b.meta["meta.id"] == ["a", "b"]


def test_unknown_key_is_inferred_with_warning():
    samples = [
        {
            "student.input_ids": [1, 2, 3],
            "infer.tok": [0.5, 1.0, 2.0],
            "infer.vec": [1, 2],
            "infer.sc": 3,
            "infer.rag": [1, 2, 3, 4],
            "infer.txt": "x",
        },
        {
            "student.input_ids": [4],
            "infer.tok": [7.0],
            "infer.vec": [3, 4],
            "infer.sc": 4,
            "infer.rag": [5],
            "infer.txt": "y",
        },
    ]
    with pytest.warns(UserWarning, match="no FieldSpec"):
        b = Collator({"student": 0}, pad_to_multiple_of=None)(samples)
    assert b["infer.tok"].shape == (2, 3) and b["infer.tok"].dtype == torch.float32
    assert b["infer.tok"][1].tolist() == [7.0, 0.0, 0.0]
    assert b["infer.vec"].shape == (2, 2) and b["infer.vec"].dtype == torch.long
    assert b["infer.sc"].tolist() == [3, 4]
    assert isinstance(b["infer.rag"], list) and b["infer.rag"][1].tolist() == [5]
    assert b.meta["infer.txt"] == ["x", "y"]
    assert FIELDS.get("infer.tok").side == "student"  # registered once, reused afterwards
    with pytest.raises(KeyError, match="FieldSpec"):
        FIELDS.get("never.seen")


def test_batch_split_select():
    b = Batch({"student.input_ids": torch.arange(8).view(4, 2)}, {"meta.id": list("abcd")})
    parts = b.split(2)
    assert len(parts) == 2 and parts[1]["student.input_ids"][0].tolist() == [4, 5]
    assert parts[1].meta["meta.id"] == ["c", "d"]
    with pytest.raises(KeyError, match="namespaced"):
        b["nonamespace"] = 1
