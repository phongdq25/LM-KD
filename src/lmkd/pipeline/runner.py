"""Pipeline runner (K10/T17): DAG planning, content-addressed artifact cache,
per-sample resume, sharding/merge, and CLI integration.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lmkd.core import registry as R
from lmkd.core.config import RecipeConfig
from lmkd.core.registry import STAGES
from lmkd.data.context import DataContext
from lmkd.models.loading import load_tokenizer
from lmkd.pipeline.artifacts import Artifact, ArtifactStore
from lmkd.pipeline.stage import Stage, StageContext
from lmkd.train.runtime import Runtime

RESERVED = ("stage", "type", "name", "inputs", "force", "dataset")


@dataclass
class PlannedNode:
    """A pipeline entry resolved into a built stage, its inputs and flags."""
    name: str
    stage_name: str
    stage: Stage
    inputs: dict[str, str]
    dataset: bool
    force: bool


def split_entry(entry: dict) -> tuple[str, str, dict | None, bool, bool, dict]:
    """Split reserved runner keys from stage config."""
    e = dict(entry)
    stage = e.pop("stage", None) or e.pop("type", None)
    if stage is None:
        raise ValueError(f"pipeline entry without 'stage': {entry}")
    name = e.pop("name", None) or stage
    inputs = e.pop("inputs", None)
    force = bool(e.pop("force", False))
    dataset = bool(e.pop("dataset", False))
    return stage, name, inputs, force, dataset, e


def plan(cfg: RecipeConfig) -> list[PlannedNode]:
    """Parse, build and validate pipeline DAG nodes without running them."""
    nodes: list[PlannedNode] = []
    names: set[str] = set()
    prev_name: str | None = None

    for entry in cfg.pipeline:
        stage_type, name, inputs_spec, force, dataset, params = split_entry(entry)
        if name in names:
            raise ValueError(f"duplicate pipeline stage name '{name}' (set `name:`)")

        stage = STAGES.build({"type": stage_type, **params})
        if stage.resources.python_env is not None:
            raise NotImplementedError("stage resources.python_env execution is not implemented")

        if inputs_spec is None:
            resolved_inputs = {"prev": prev_name} if prev_name is not None else {}
        else:
            resolved_inputs = dict(inputs_spec)

        missing_nodes = [src for src in resolved_inputs.values() if src not in names]
        if missing_nodes:
            raise ValueError(
                f"pipeline stage '{name}' needs inputs {missing_nodes} that are not defined earlier"
            )

        if getattr(stage, "inputs", None):
            missing_aliases = [alias for alias in stage.inputs if alias not in resolved_inputs]
            if missing_aliases:
                raise ValueError(
                    f"pipeline stage '{name}' ({stage_type}) missing required input aliases: {missing_aliases}"
                )

        nodes.append(
            PlannedNode(
                name=name,
                stage_name=stage_type,
                stage=stage,
                inputs=resolved_inputs,
                dataset=dataset,
                force=force,
            )
        )
        names.add(name)
        prev_name = name

    return nodes


def build_runtime_for_pipeline(cfg: RecipeConfig) -> Runtime:
    """Build lightweight runtime for pipeline runner (seeds, engine, tokenizers, no model weights)."""
    from lmkd.plugins import load_plugins

    load_plugins(cfg.imports)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    engine = R.ENGINES.build(cfg.engine)
    rt = Runtime(config=cfg, engine=engine)

    rt.student_tokenizer = load_tokenizer(cfg.models.student)
    if cfg.models.teacher is not None:
        rt.teacher_tokenizer = load_tokenizer(cfg.models.teacher)

    artifacts_root = getattr(cfg, "artifacts_dir", None) or (Path(cfg.train.output_dir) / "artifacts")
    rt.artifacts = ArtifactStore(artifacts_root)
    rt.data_ctx = DataContext(
        student_tokenizer=rt.student_tokenizer,
        teacher_tokenizer=rt.teacher_tokenizer,
        max_length=cfg.data.max_length,
        artifacts=rt.artifacts,
    )
    return rt


def _as_set(value: str | list[str] | tuple | set[str] | None) -> set[str] | None:
    if value is None:
        return None
    return {value} if isinstance(value, str) else set(value)


def _shard_dir(art: Artifact, idx: int, count: int) -> Path:
    return art.path.parent / f"{art.key}.shard-{idx}-of-{count}"


def _run_shard(rt: Any, n: PlannedNode, art: Artifact, store: ArtifactStore, in_arts: dict[str, Artifact],
               input_keys: dict[str, str], shard: tuple[int, int], forced: bool) -> None:
    """Run ONE shard of a node; the last shard to finish merges all shards into the artifact."""
    if not getattr(n.stage, "supports_shards", False):
        raise ValueError(f"stage '{n.name}' ({n.stage_name}) does not support sharding")
    idx, count = shard
    shard_dir = _shard_dir(art, idx, count)
    shard_art = Artifact(node=n.name, stage=n.stage_name, key=f"{art.key}.shard-{idx}-of-{count}", path=shard_dir)
    ctx = StageContext(rt=rt, inputs=in_arts, output=shard_art, shard=shard, node=n.name, force=forced)
    info = n.stage.run(ctx) or {}
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "manifest.json").write_text(json.dumps({"shard": list(shard), "info": info}, default=str))

    all_dirs = [_shard_dir(art, i, count) for i in range(count)]
    if all((d / "manifest.json").exists() for d in all_dirs):
        merge_ctx = StageContext(rt=rt, inputs=in_arts, output=art.partial, shard=shard, node=n.name, force=True)
        merge_info = n.stage.merge(merge_ctx, all_dirs) or {}
        store.finalize(art, {"inputs": input_keys, **merge_info})


def _run_whole(rt: Any, n: PlannedNode, art: Artifact, store: ArtifactStore, in_arts: dict[str, Artifact],
               input_keys: dict[str, str], shard: tuple[int, int], forced: bool) -> None:
    ctx = StageContext(rt=rt, inputs=in_arts, output=art.partial, shard=shard, node=n.name, force=forced)
    info = n.stage.run(ctx) or {}
    store.finalize(art, {"inputs": input_keys, **info})


def _dataset_samples(executed: list[PlannedNode], arts: dict[str, Artifact]) -> list[dict] | None:
    """Samples of the dataset node: the last node marked ``dataset: true`` (if it can ``load_samples``),
    else the last executed node that can."""
    marked = [n for n in executed if n.dataset]
    if marked:
        try:
            return marked[-1].stage.load_samples(arts[marked[-1].name])
        except NotImplementedError:
            pass
    for n in reversed(executed):
        try:
            return n.stage.load_samples(arts[n.name])
        except NotImplementedError:
            continue
    return None


def run_pipeline(
    rt: Any,
    *,
    until: str | None = None,
    only: list[str] | set[str] | str | None = None,
    force: tuple | list | set | str = (),
    shard: tuple[int, int] = (0, 1),
    dry_run: bool = False,
) -> list[dict] | None:
    """Run ``rt.config.pipeline`` in DAG order.

    Returns the samples of the stage designated for dataset (marked `dataset: true` or
    last stage with `load_samples`), or None if no stage produces training samples.
    """
    cfg = rt.config
    nodes = plan(cfg)
    artifacts_root = getattr(cfg, "artifacts_dir", None) or (Path(cfg.train.output_dir) / "artifacts")
    store = getattr(rt, "artifacts", None) or ArtifactStore(artifacts_root)
    rt.artifacts = store

    only_set = _as_set(only)
    force_set = _as_set(force) or set()

    arts: dict[str, Artifact] = {}
    executed: list[PlannedNode] = []

    for n in nodes:
        in_arts = {slot: arts[src] for slot, src in n.inputs.items()}
        input_keys = {slot: arts[src].key for slot, src in n.inputs.items()}

        n.stage.setup(rt)
        art = store.artifact(node=n.name, stage_name=n.stage_name, fingerprint=n.stage.fingerprint(),
                             input_keys=input_keys, code_version=n.stage.code_version)
        arts[n.name] = art
        executed.append(n)

        forced = n.force or n.name in force_set or "all" in force_set
        if only_set is not None and n.name not in only_set:
            if not art.done:
                raise RuntimeError(f"upstream node '{n.name}' not built; run without --only")
        elif dry_run:
            print(f"{n.name} {art.key} {'cached' if art.done else 'TO RUN'}")
        elif not art.done or forced:
            run = _run_shard if shard[1] > 1 else _run_whole
            run(rt, n, art, store, in_arts, input_keys, shard, forced)

        if art.done and not dry_run:
            n.stage.attach(art, rt)

        if until is not None and n.name == until:
            break

    if dry_run:
        return None
    rt.extra["pipeline_artifacts"] = arts
    return _dataset_samples(executed, arts)
