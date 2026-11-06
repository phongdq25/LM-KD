from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from lmkd.core.config import recipe_from_dict
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import STAGES
from lmkd.pipeline.artifacts import ArtifactStore
from lmkd.pipeline.runner import PlannedNode, plan, run_pipeline
from lmkd.pipeline.stage import Stage, StageContext
from lmkd.train.runtime import Runtime

# ----------------------------------------------------------------------------- Test Stages
_CALL_COUNTS = {}


@STAGES.register("_test_runner_a", force=True)
class _StageA(Stage):
    class Config(BaseConfig):
        val: int = 1

    def run(self, ctx: StageContext):
        _CALL_COUNTS["a"] = _CALL_COUNTS.get("a", 0) + 1
        ctx.output.path.mkdir(parents=True, exist_ok=True)
        (ctx.output.path / "res.txt").write_text(f"a_{self.cfg.val}")
        return {"val": self.cfg.val}


@STAGES.register("_test_runner_b", force=True)
class _StageB(Stage):
    inputs: ClassVar[dict[str, str]] = {"data": "file"}

    class Config(BaseConfig):
        factor: int = 2

    def run(self, ctx: StageContext):
        _CALL_COUNTS["b"] = _CALL_COUNTS.get("b", 0) + 1
        ctx.output.path.mkdir(parents=True, exist_ok=True)
        (ctx.output.path / "res.txt").write_text(f"b_{self.cfg.factor}")
        return {"factor": self.cfg.factor}


@STAGES.register("_test_runner_c", force=True)
class _StageC(Stage):
    inputs: ClassVar[dict[str, str]] = {"data": "file", "signals": "file"}

    class Config(BaseConfig):
        tag: str = "c"

    def run(self, ctx: StageContext):
        _CALL_COUNTS["c"] = _CALL_COUNTS.get("c", 0) + 1
        ctx.output.path.mkdir(parents=True, exist_ok=True)
        return {"tag": self.cfg.tag}


@STAGES.register("_test_failing_stage", force=True)
class _FailingStage(Stage):
    class Config(BaseConfig):
        fail: bool = True

    def run(self, ctx: StageContext):
        _CALL_COUNTS["fail"] = _CALL_COUNTS.get("fail", 0) + 1
        ctx.output.path.mkdir(parents=True, exist_ok=True)
        (ctx.output.path / "partial.txt").write_text("in progress")
        if self.cfg.fail:
            raise RuntimeError("stage failed deliberately")
        return {"done": True}


@STAGES.register("_test_shard_stage", force=True)
class _ShardStage(Stage):
    supports_shards = True

    class Config(BaseConfig):
        n: int = 10

    def run(self, ctx: StageContext):
        _CALL_COUNTS["shard_run"] = _CALL_COUNTS.get("shard_run", 0) + 1
        ctx.output.path.mkdir(parents=True, exist_ok=True)
        idx, count = ctx.shard
        (ctx.output.path / f"part_{idx}.txt").write_text(f"shard {idx}/{count}")
        return {"shard_idx": idx}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]):
        _CALL_COUNTS["shard_merge"] = _CALL_COUNTS.get("shard_merge", 0) + 1
        ctx.output.path.mkdir(parents=True, exist_ok=True)
        all_parts = []
        for d in shard_dirs:
            for p in d.glob("part_*.txt"):
                all_parts.append(p.read_text())
        (ctx.output.path / "merged.txt").write_text("\n".join(all_parts))
        return {"merged_count": len(all_parts)}


def _make_dummy_rt(tmp_path: Path, pipeline: list[dict]) -> Runtime:
    cfg = recipe_from_dict(
        {
            "models": {"student": {"path": "dummy"}},
            "data": {"train": {"type": "jsonl", "path": str(tmp_path / "train.jsonl")}},
            "loss": {"terms": [{"criterion": "nll"}]},
            "train": {"output_dir": str(tmp_path / "out")},
            "artifacts_dir": str(tmp_path / "artifacts"),
            "pipeline": pipeline,
        }
    )
    rt = Runtime(config=cfg, engine=None)
    rt.artifacts = ArtifactStore(tmp_path / "artifacts")
    return rt


# ----------------------------------------------------------------------------- Tests
def test_plan_named_inputs(tmp_path: Path):
    pipeline = [
        {"stage": "_test_runner_a", "name": "node_a"},
        {"stage": "_test_runner_b", "name": "node_b", "inputs": {"data": "node_a"}},
        {
            "stage": "_test_runner_c",
            "name": "node_c",
            "inputs": {"data": "node_a", "signals": "node_b"},
            "dataset": True,
        },
    ]
    cfg = _make_dummy_rt(tmp_path, pipeline).config
    nodes = plan(cfg)
    assert len(nodes) == 3
    assert isinstance(nodes[0], PlannedNode)
    assert nodes[2].name == "node_c"
    assert nodes[2].inputs == {"data": "node_a", "signals": "node_b"}
    assert nodes[2].dataset is True

    # Duplicate name
    bad_pipeline_dup = [
        {"stage": "_test_runner_a", "name": "dup"},
        {"stage": "_test_runner_a", "name": "dup"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        plan(_make_dummy_rt(tmp_path, bad_pipeline_dup).config)

    # Input referencing later or non-existent node
    bad_pipeline_future = [
        {"stage": "_test_runner_b", "name": "first", "inputs": {"data": "later"}},
        {"stage": "_test_runner_a", "name": "later"},
    ]
    with pytest.raises(ValueError, match="needs inputs .* that are not defined earlier"):
        plan(_make_dummy_rt(tmp_path, bad_pipeline_future).config)


def test_cache_skip_and_invalidate(tmp_path: Path):
    _CALL_COUNTS.clear()
    pipeline = [
        {"stage": "_test_runner_a", "name": "a", "val": 10},
        {"stage": "_test_runner_b", "name": "b", "inputs": {"data": "a"}, "factor": 3},
    ]

    # Run 1st time: both should run
    rt1 = _make_dummy_rt(tmp_path, pipeline)
    run_pipeline(rt1)
    assert _CALL_COUNTS["a"] == 1
    assert _CALL_COUNTS["b"] == 1

    # Run 2nd time: both cached, neither should run
    rt2 = _make_dummy_rt(tmp_path, pipeline)
    run_pipeline(rt2)
    assert _CALL_COUNTS["a"] == 1
    assert _CALL_COUNTS["b"] == 1

    # Change parameter of b: only b should run
    pipeline_b_changed = [
        {"stage": "_test_runner_a", "name": "a", "val": 10},
        {"stage": "_test_runner_b", "name": "b", "inputs": {"data": "a"}, "factor": 99},
    ]
    rt3 = _make_dummy_rt(tmp_path, pipeline_b_changed)
    run_pipeline(rt3)
    assert _CALL_COUNTS["a"] == 1
    assert _CALL_COUNTS["b"] == 2

    # Change parameter of a: both a and downstream b should run
    pipeline_a_changed = [
        {"stage": "_test_runner_a", "name": "a", "val": 42},
        {"stage": "_test_runner_b", "name": "b", "inputs": {"data": "a"}, "factor": 99},
    ]
    rt4 = _make_dummy_rt(tmp_path, pipeline_a_changed)
    run_pipeline(rt4)
    assert _CALL_COUNTS["a"] == 2
    assert _CALL_COUNTS["b"] == 3


def test_partial_not_done(tmp_path: Path):
    _CALL_COUNTS.clear()
    pipeline = [{"stage": "_test_failing_stage", "name": "fail_node", "fail": True}]
    rt = _make_dummy_rt(tmp_path, pipeline)

    with pytest.raises(RuntimeError, match="deliberately"):
        run_pipeline(rt)
    assert _CALL_COUNTS["fail"] == 1

    # Manifest must not exist
    store = rt.artifacts
    node = plan(rt.config)[0]
    art = store.artifact(node.name, node.stage_name, node.stage.fingerprint(), {})
    assert not art.done
    assert not (art.path / "manifest.json").exists()

    # Second run without failure should run and finalize
    pipeline_fixed = [{"stage": "_test_failing_stage", "name": "fail_node", "fail": False}]
    rt_fixed = _make_dummy_rt(tmp_path, pipeline_fixed)
    run_pipeline(rt_fixed)
    assert _CALL_COUNTS["fail"] == 2
    node_fixed = plan(rt_fixed.config)[0]
    art_fixed = store.artifact(node_fixed.name, node_fixed.stage_name, node_fixed.stage.fingerprint(), {})
    assert art_fixed.done


def test_force_and_only(tmp_path: Path):
    _CALL_COUNTS.clear()
    pipeline = [
        {"stage": "_test_runner_a", "name": "a", "val": 1},
        {"stage": "_test_runner_b", "name": "b", "inputs": {"data": "a"}},
    ]
    rt = _make_dummy_rt(tmp_path, pipeline)
    run_pipeline(rt)
    assert _CALL_COUNTS["a"] == 1
    assert _CALL_COUNTS["b"] == 1

    # force=("b",): only b reruns
    run_pipeline(rt, force=("b",))
    assert _CALL_COUNTS["a"] == 1
    assert _CALL_COUNTS["b"] == 2

    # only=("c",) when c has unbuilt upstream node -> clear error
    pipeline_abc = [
        {"stage": "_test_runner_a", "name": "a", "val": 100},  # unbuilt parameter
        {"stage": "_test_runner_b", "name": "b", "inputs": {"data": "a"}},
    ]
    rt_unbuilt = _make_dummy_rt(tmp_path, pipeline_abc)
    with pytest.raises(RuntimeError, match="upstream node 'a' not built; run without --only"):
        run_pipeline(rt_unbuilt, only=["b"])


def test_shards_merge(tmp_path: Path):
    _CALL_COUNTS.clear()
    pipeline = [{"stage": "_test_shard_stage", "name": "sharded", "n": 10}]
    rt = _make_dummy_rt(tmp_path, pipeline)

    # Run shard 0/2
    run_pipeline(rt, shard=(0, 2))
    assert _CALL_COUNTS.get("shard_run", 0) == 1
    assert _CALL_COUNTS.get("shard_merge", 0) == 0

    # Run shard 1/2
    run_pipeline(rt, shard=(1, 2))
    assert _CALL_COUNTS.get("shard_run", 0) == 2
    assert _CALL_COUNTS.get("shard_merge", 0) == 1

    node = plan(rt.config)[0]
    art = rt.artifacts.artifact(node.name, node.stage_name, node.stage.fingerprint(), {})
    assert art.done
    manifest = art.manifest()
    assert "merged.txt" in manifest["files"]
