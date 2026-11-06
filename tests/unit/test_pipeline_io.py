from __future__ import annotations

import json
from pathlib import Path

import torch

from lmkd.pipeline.io import ResumableWriter, load_samples_dir, save_samples, shard_of


def test_resumable_writer_skips_done(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    with ResumableWriter(path) as writer:
        writer.write({"meta.id": "1", "val": "a"})
        writer.write({"meta.id": "2", "val": "b"})
        writer.write({"meta.id": "3", "val": "c"})

    # Corrupt the end of the file with an incomplete JSON object
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"meta.id": "4", "broken')

    # Reopen with ResumableWriter
    with ResumableWriter(path) as writer:
        assert writer.done("1")
        assert writer.done("2")
        assert writer.done("3")
        assert not writer.done("4")
        writer.write({"meta.id": "5", "val": "e"})

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["meta.id"] for r in rows]
    assert ids == ["1", "2", "3", "5"]


def test_shard_of():
    items = list(range(10))
    assert shard_of(items, 1, 3) == [1, 4, 7]
    assert shard_of(items, 0, 3) == [0, 3, 6, 9]
    assert shard_of(items, 2, 3) == [2, 5, 8]


def test_save_and_load_samples(tmp_path: Path):
    samples = [
        {"meta.id": "0", "student.input_ids": [10, 20, 30], "student.attention_mask": torch.tensor([1, 1, 1])},
        {"meta.id": "1", "student.input_ids": [40, 50], "student.attention_mask": torch.tensor([1, 1])},
    ]
    out_dir = tmp_path / "saved"
    save_samples(out_dir, samples)
    loaded = load_samples_dir(out_dir)
    assert len(loaded) == 2
    assert loaded[0]["meta.id"] == "0"
    assert loaded[0]["student.input_ids"] == [10, 20, 30]
    assert loaded[0]["student.attention_mask"] == [1, 1, 1]
    assert loaded[1]["meta.id"] == "1"
