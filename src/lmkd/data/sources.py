"""Raw data sources: return an iterable of python dict records.

Each record gets a stable ``meta.id`` (from the record's ``id`` field or its row index).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import SOURCES
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl


class Source(Plugin):
    """Raw record reader (jsonl, csv, parquet, HF dataset): ``records()`` yields dicts."""
    def records(self) -> Iterator[dict]:
        raise NotImplementedError

    def fingerprint(self) -> dict:
        return self.cfg.model_dump()


@SOURCES.register("jsonl")
class JsonlSource(Source):
    """JSON-lines file (one record per line). Also accepts a .json file containing a list."""

    class Config(BaseConfig):
        path: str
        limit: int | None = None
        id_field: str = "id"

    def records(self) -> Iterator[dict]:
        p = Path(self.cfg.path)
        if p.suffix == ".json":
            with open(p, encoding="utf-8") as f:
                rows = json.load(f)
        else:
            with open(p, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        for i, r in enumerate(rows):
            if self.cfg.limit is not None and i >= self.cfg.limit:
                break
            r = dict(r)
            r.setdefault("meta.id", str(r.get(self.cfg.id_field, i)))
            yield r

    def fingerprint(self) -> dict:
        p = Path(self.cfg.path)
        st = p.stat()
        return {**self.cfg.model_dump(), "size": st.st_size, "mtime": int(st.st_mtime)}


@SOURCES.register("csv")
class CsvSource(Source):
    """CSV with a header row (e.g. TALAS ``,text,source``). Values are strings."""

    class Config(BaseConfig):
        path: str
        limit: int | None = None
        id_field: str = "id"
        delimiter: str = ","

    def records(self) -> Iterator[dict]:
        import csv

        with open(self.cfg.path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=self.cfg.delimiter))
        for i, r in enumerate(rows):
            if self.cfg.limit is not None and i >= self.cfg.limit:
                break
            r = dict(r)
            r.setdefault("meta.id", str(r.get(self.cfg.id_field) or i))
            yield r

    def fingerprint(self) -> dict:
        st = Path(self.cfg.path).stat()
        return {**self.cfg.model_dump(), "size": st.st_size, "mtime": int(st.st_mtime)}


@SOURCES.register("hf_dataset")
class HFDatasetSource(Source):
    """HuggingFace `datasets` (hub name or save_to_disk path)."""

    class Config(BaseConfig):
        path: str
        split: str = "train"
        name: str | None = None
        limit: int | None = None
        id_field: str = "id"

    def records(self) -> Iterator[dict]:
        try:
            import datasets
        except ImportError as err:
            raise ImportError(
                "HFDatasetSource requires 'datasets'. Run `pip install -e '.[data]'`."
            ) from err

        p = Path(self.cfg.path)
        if p.exists() and (
            p.is_dir()
            or (p / "dataset_dict.json").exists()
            or (p / "dataset_info.json").exists()
            or (p / "state.json").exists()
        ):
            ds = datasets.load_from_disk(str(p))
            if hasattr(datasets, "DatasetDict") and isinstance(ds, datasets.DatasetDict):
                ds = ds[self.cfg.split]
        else:
            ds = datasets.load_dataset(self.cfg.path, self.cfg.name, split=self.cfg.split)

        for i, row in enumerate(ds):
            if self.cfg.limit is not None and i >= self.cfg.limit:
                break
            r = dict(row)
            val = r.get(self.cfg.id_field)
            r.setdefault("meta.id", str(val if val is not None else i))
            yield r

    def fingerprint(self) -> dict:
        fp = dict(self.cfg.model_dump())
        p = Path(self.cfg.path)
        if p.exists():
            try:
                st = p.stat()
                fp["size"] = st.st_size
                fp["mtime"] = int(st.st_mtime)
            except Exception:
                pass
            try:
                import datasets

                ds = datasets.load_from_disk(str(p))
                if hasattr(datasets, "DatasetDict") and isinstance(ds, datasets.DatasetDict):
                    ds = ds[self.cfg.split]
                if hasattr(ds, "_fingerprint") and ds._fingerprint:
                    fp["fingerprint"] = ds._fingerprint
            except Exception:
                pass
        return fp


@SOURCES.register("parquet")
class ParquetSource(Source):
    """Parquet table source read via pyarrow."""

    class Config(BaseConfig):
        path: str
        limit: int | None = None
        id_field: str = "id"

    def records(self) -> Iterator[dict]:
        try:
            import pyarrow.parquet as pq
        except ImportError as err:
            raise ImportError(
                "ParquetSource requires 'pyarrow'. Run `pip install -e '.[data]'`."
            ) from err

        table = pq.read_table(self.cfg.path)
        rows = table.to_pylist()
        for i, r in enumerate(rows):
            if self.cfg.limit is not None and i >= self.cfg.limit:
                break
            r = dict(r)
            val = r.get(self.cfg.id_field)
            r.setdefault("meta.id", str(val if val is not None else i))
            yield r

    def fingerprint(self) -> dict:
        st = Path(self.cfg.path).stat()
        return {**self.cfg.model_dump(), "size": st.st_size, "mtime": int(st.st_mtime)}


@SOURCES.register("artifact_records", source="SGL/src/sgl/palign/build.py:22-62")
class ArtifactRecords(Source):
    """Read records.jsonl from an upstream pipeline node (P-ALIGN assembly output)."""

    class Config(BaseConfig):
        node: str

    def bind(self, art: Artifact) -> None:
        """Resolve the named upstream artifact before records are read."""
        self.artifact = art

    def records(self) -> Iterator[dict]:
        if not hasattr(self, "artifact"):
            raise RuntimeError("artifact_records must be bound to an upstream node")
        yield from read_jsonl(self.artifact.path / "records.jsonl")
