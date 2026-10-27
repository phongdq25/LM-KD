"""Helper I/O, serialization, resume and sharding for pipeline stages."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _to_serializable(val: Any) -> Any:
    if hasattr(val, "tolist"):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_to_serializable(x) for x in val]
    if isinstance(val, dict):
        return {k: _to_serializable(v) for k, v in val.items()}
    return val


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            clean = {k: _to_serializable(v) for k, v in r.items()}
            f.write(json.dumps(clean, default=str) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ResumableWriter:
    """Per-sample resume writer appending to a jsonl file."""

    def __init__(self, path: Path | str, key: str = "meta.id"):
        self.path = Path(path)
        self.key = key
        self.seen_ids: set[str] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        valid_bytes = 0
        if self.path.exists():
            with open(self.path, "rb") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        valid_bytes += len(line)
                        continue
                    try:
                        obj = json.loads(stripped.decode("utf-8"))
                        if self.key in obj:
                            self.seen_ids.add(str(obj[self.key]))
                        valid_bytes += len(line)
                    except Exception:
                        # Corrupted / incomplete line at EOF: stop here
                        break
            if valid_bytes < self.path.stat().st_size:
                with open(self.path, "r+b") as f:
                    f.truncate(valid_bytes)

        self._file = open(self.path, "a", encoding="utf-8")  # noqa: SIM115  (kept open across write() calls)

    def done(self, sid: str) -> bool:
        return str(sid) in self.seen_ids

    def write(self, row: dict) -> None:
        clean = {k: _to_serializable(v) for k, v in row.items()}
        line = json.dumps(clean, default=str)
        self._file.write(line + "\n")
        self._file.flush()
        if self.key in row:
            self.seen_ids.add(str(row[self.key]))

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self) -> ResumableWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def shard_of(items: list, index: int, count: int) -> list:
    """Select the slice of items belonging to this shard index: items[index::count]."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if index < 0 or index >= count:
        raise ValueError(f"index {index} out of range [0, {count})")
    return items[index::count]


def save_samples(dir: Path | str, samples: list[dict]) -> None:
    """Save dataset samples to directory: Arrow if datasets is available, plus samples.jsonl."""
    p = Path(dir)
    p.mkdir(parents=True, exist_ok=True)

    clean_samples = [{k: _to_serializable(v) for k, v in s.items()} for s in samples]
    write_jsonl(p / "samples.jsonl", clean_samples)

    try:
        import datasets

        ds = datasets.Dataset.from_list(clean_samples)
        ds.save_to_disk(str(p / "arrow"))
    except Exception:
        pass


def load_samples_dir(dir: Path | str) -> list[dict]:
    """Load samples from directory (trying Arrow first, falling back to samples.jsonl)."""
    p = Path(dir)
    arrow_dir = p / "arrow"
    if arrow_dir.exists():
        try:
            import datasets

            ds = datasets.load_from_disk(str(arrow_dir))
            return [dict(row) for row in ds]
        except Exception:
            pass

    jsonl_path = p / "samples.jsonl"
    if jsonl_path.exists():
        return read_jsonl(jsonl_path)

    raise FileNotFoundError(f"no valid samples found in {dir}")

