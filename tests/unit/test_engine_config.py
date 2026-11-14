from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from lmkd.core.config import load_recipe, load_yaml_with_bases
from lmkd.train.engine import AccelerateEngine, Engine


def test_distributed_engine_defaults_to_deepspeed():
    assert AccelerateEngine.Config().backend == "deepspeed"
    cfg = AccelerateEngine.Config(backend="deepspeed", deepspeed_config={"zero_optimization": {"stage": 3}})
    assert cfg.deepspeed_config["zero_optimization"]["stage"] == 3


@pytest.mark.parametrize("backend", ["fsdp", "fsdp2", "ddp", "unknown"])
def test_distributed_engine_rejects_other_backends(backend):
    with pytest.raises(ValidationError, match="backend"):
        AccelerateEngine.Config(backend=backend)


@pytest.mark.parametrize("field", ["fsdp", "find_unused_parameters", "grad_bucket_mb"])
def test_distributed_engine_rejects_removed_options(field):
    with pytest.raises(ValidationError, match=field):
        AccelerateEngine.Config(**{field: {}})


# ----------------------------------------------------------------------------- T05: DS config patch
def _engine(world_size: int = 2, **cfg) -> AccelerateEngine:
    eng = object.__new__(AccelerateEngine)  # skip PartialState: config logic only
    Engine.__init__(eng, AccelerateEngine.Config(**cfg))
    eng.world_size, eng.rank = world_size, 0
    return eng


def test_ds_config_patched_from_train_settings():
    eng = _engine(
        2,
        batch_size=32,
        micro_batch_size=4,
        gradient_clipping=1.0,
        mixed_precision="bf16",
        deepspeed_config={
            "zero_optimization": {"stage": 2},
            "gradient_clipping": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "optimizer": {"type": "AdamW"},
            "scheduler": {"type": "WarmupLR"},
        },
    )
    ds = eng.build_ds_config()
    assert ds["train_micro_batch_size_per_gpu"] == 4
    assert ds["gradient_accumulation_steps"] == 4  # 32 / (4 x 2)
    assert ds["train_batch_size"] == 32 and ds["gradient_clipping"] == 1.0
    assert ds["bf16"]["enabled"] and not ds["fp16"]["enabled"]
    assert "optimizer" not in ds and "scheduler" not in ds  # ours, never DS-built
    assert AccelerateEngine.handles_clipping


def test_ds_config_no_clipping_and_bad_batch():
    ds = _engine(1, batch_size=8, micro_batch_size=8, gradient_clipping=None, mixed_precision="no").build_ds_config()
    assert ds["gradient_clipping"] == 0.0 and ds["gradient_accumulation_steps"] == 1
    assert not ds["bf16"]["enabled"] and not ds["fp16"]["enabled"]
    with pytest.raises(ValueError, match="divisible"):
        _engine(2, batch_size=8, micro_batch_size=3).build_ds_config()
    with pytest.raises(ValueError, match="batch_size"):
        _engine(2).build_ds_config()


def test_teacher_shard_placement_not_supported():
    with pytest.raises(NotImplementedError, match="teacher placement 'shard'"):
        _engine().place_frozen(torch.nn.Linear(2, 2), "shard")


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_engine_zero_base_recipes(stage):
    root = Path(__file__).resolve().parents[2]
    raw = load_yaml_with_bases(root / "recipes/_base_" / f"engine_zero{stage}.yaml")
    assert raw["engine"]["type"] == "accelerate" and "_delete_" in raw["engine"]
    eng = _engine(
        2, **{k: v for k, v in raw["engine"].items() if k not in ("type", "_delete_") and not str(v).startswith("${")}
    )
    eng.cfg.batch_size, eng.cfg.micro_batch_size = 16, 4
    ds = eng.build_ds_config()
    assert ds["zero_optimization"]["stage"] == stage and ds["gradient_accumulation_steps"] == 2
    if stage == 2:
        assert ds["zero_optimization"]["offload_optimizer"]["device"] == "cpu"


def test_zero2_recipe_mirrors_train_section():
    root = Path(__file__).resolve().parents[2]
    cfg = load_recipe(root / "recipes/same_tokenizer/fkl_zero2.yaml", ["train.batch_size=64"])
    eng = AccelerateEngine.Config(**{k: v for k, v in cfg.engine.items() if k != "type"})
    assert cfg.engine["type"] == "accelerate" and "device" not in cfg.engine  # local block replaced
    assert eng.batch_size == 64 and eng.micro_batch_size == cfg.train.micro_batch_size
    assert eng.gradient_clipping == cfg.grad.clip_norm


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_zero0_low_precision_rejected(precision):
    # DeepSpeed 0.19.8 stage 0 + bf16/fp16 = FP16_UnfusedOptimizer: extra init optimizer step and
    # (fp16) unscaled loss; verified on GPU.
    eng = _engine(
        2,
        batch_size=8,
        micro_batch_size=2,
        mixed_precision=precision,
        deepspeed_config={"zero_optimization": {"stage": 0}},
    )
    with pytest.raises(ValueError, match="stage 0"):
        eng.build_ds_config()
    ok = _engine(
        2, batch_size=8, micro_batch_size=2, mixed_precision="no", deepspeed_config={"zero_optimization": {"stage": 0}}
    ).build_ds_config()
    assert ok["gradient_accumulation_steps"] == 2


def test_accelerate_engine_setup_infers_from_train_and_grad():
    eng = _engine(2)
    assert eng.cfg.batch_size is None
    assert eng.cfg.micro_batch_size is None
    assert eng.cfg.gradient_clipping is None

    rt = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(batch_size=64, micro_batch_size=8), grad=SimpleNamespace(clip_norm=1.5)
        )
    )
    eng.setup(rt)
    assert eng.cfg.batch_size == 64
    assert eng.cfg.micro_batch_size == 8
    assert eng.cfg.gradient_clipping == 1.5
