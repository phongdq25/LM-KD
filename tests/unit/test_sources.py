from __future__ import annotations

from pathlib import Path

import pytest

from lmkd.core.registry import SOURCES


def test_parquet_source(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    table = pa.Table.from_pydict(
        {
            "id": ["row_0", "row_1", "row_2"],
            "prompt": ["p0", "p1", "p2"],
            "response": ["r0", "r1", "r2"],
        }
    )
    path = tmp_path / "test.parquet"
    pq.write_table(table, str(path))

    src = SOURCES.build({"type": "parquet", "path": str(path)})
    records = list(src.records())
    assert len(records) == 3
    assert records[0]["meta.id"] == "row_0"
    assert records[1]["meta.id"] == "row_1"
    assert records[2]["meta.id"] == "row_2"
    assert records[0]["prompt"] == "p0"

    fp = src.fingerprint()
    assert fp["size"] == path.stat().st_size


def test_hf_dataset_source_disk(tmp_path: Path):
    datasets = pytest.importorskip("datasets")

    ds = datasets.Dataset.from_list(
        [
            {"id": "hf_0", "prompt": "hp0", "response": "hr0"},
            {"id": "hf_1", "prompt": "hp1", "response": "hr1"},
        ]
    )
    disk_path = tmp_path / "hf_disk"
    ds.save_to_disk(str(disk_path))

    src = SOURCES.build({"type": "hf_dataset", "path": str(disk_path)})
    records = list(src.records())
    assert len(records) == 2
    assert records[0]["meta.id"] == "hf_0"
    assert records[1]["meta.id"] == "hf_1"
    assert records[0]["response"] == "hr0"

    fp = src.fingerprint()
    assert "path" in fp
