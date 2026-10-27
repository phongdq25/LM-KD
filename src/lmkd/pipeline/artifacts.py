"""Content-addressed artifact store (DESIGN §10).

Layout: ``<root>/<stage_name>/<key>/`` with ``manifest.json`` written LAST (its presence = done).
key = sha256(stage fingerprint + input keys + code version)[:16].
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def stable_hash(obj: Any, n: int = 16) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def content_hash(path: Path | str) -> str:
    p = Path(path)
    if p.suffix == ".parquet":
        try:
            import pandas as pd

            frame = pd.read_parquet(p)
            payload = frame.to_json(orient="records", double_precision=15).encode()
            return hashlib.sha256(payload).hexdigest()
        except Exception:
            pass
    payload = p.read_bytes()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class Artifact:
    """Output of one pipeline node: content-hash ``key``, directory, and completion state."""
    stage: str
    key: str
    path: Path
    node: str = ""

    @property
    def done(self) -> bool:
        return (self.path / "manifest.json").exists()

    @property
    def partial_path(self) -> Path:
        return self.path.parent / f"{self.key}.partial"

    @property
    def partial(self) -> Artifact:
        return Artifact(node=self.node, stage=self.stage, key=self.key, path=self.partial_path)

    def manifest(self) -> dict:
        return json.loads((self.path / "manifest.json").read_text())


class ArtifactStore:
    """Content-addressed storage of stage outputs (key = hash of stage, fingerprint, input keys, code version)
    with atomic finalize.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def artifact(
        self,
        node: str,
        stage_name: str | dict = "",
        fingerprint: dict | None = None,
        input_keys: dict[str, str] | list[str] | None = None,
        code_version: str = "1",
    ) -> Artifact:
        if isinstance(stage_name, dict):
            # Compatibility with old signature: artifact(stage, fp, in_keys, code_version)
            stg = node
            node_name = node
            fp = stage_name
            in_keys: Any = fingerprint if isinstance(fingerprint, (dict, list, tuple)) else {}
            code_v = str(input_keys) if input_keys is not None else code_version
        else:
            node_name = node
            stg = stage_name or node
            fp = fingerprint or {}
            in_keys = input_keys or {}
            code_v = code_version

        sorted_inputs: Any
        if isinstance(in_keys, dict):
            sorted_inputs = {k: str(in_keys[k]) for k in sorted(in_keys)}
        elif isinstance(in_keys, (list, tuple)):
            sorted_inputs = sorted(str(x) for x in in_keys)
        else:
            sorted_inputs = str(in_keys)

        key = stable_hash({"stage": stg, "fp": fp, "inputs": sorted_inputs, "code": code_v})
        return Artifact(node=node_name, stage=stg, key=key, path=self.root / stg / key)

    def finalize(self, art: Artifact, info: dict) -> None:
        work_dir = art.partial_path if art.partial_path.exists() else art.path
        work_dir.mkdir(parents=True, exist_ok=True)

        files = {}
        for p in sorted(work_dir.rglob("*")):
            if p.is_file() and p.name not in ("manifest.json", "manifest.json.tmp"):
                rel = str(p.relative_to(work_dir))
                files[rel] = content_hash(p)

        cfg_info = info.get("config", {})
        inputs_info = info.get("inputs", {})
        extra_info = {
            k: v
            for k, v in info.items()
            if k not in ("stage", "node", "key", "time", "config", "inputs", "files")
        }

        data = {
            "stage": art.stage,
            "node": art.node,
            "key": art.key,
            "time": time.time(),
            "config": cfg_info,
            "inputs": inputs_info,
            "info": extra_info,
            "files": files,
        }
        for k, v in extra_info.items():
            if k not in data:
                data[k] = v

        tmp = work_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.rename(work_dir / "manifest.json")

        if work_dir != art.path:
            if art.path.exists():
                shutil.rmtree(art.path)
            art.path.parent.mkdir(parents=True, exist_ok=True)
            work_dir.rename(art.path)

    def gc(self, keep: set[str]) -> int:
        deleted = 0
        if not self.root.exists():
            return 0
        for stage_dir in self.root.iterdir():
            if stage_dir.is_dir():
                for key_dir in stage_dir.iterdir():
                    if key_dir.is_dir():
                        raw_key = key_dir.name.split(".")[0]
                        if raw_key not in keep:
                            shutil.rmtree(key_dir)
                            deleted += 1
        return deleted
