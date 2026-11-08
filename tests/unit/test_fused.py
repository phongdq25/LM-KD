"""Unit tests for fused loss path and CPU fallbacks (T26)."""

from __future__ import annotations

import warnings

import pytest
import torch

from lmkd.core.config import recipe_from_dict
from lmkd.losses.sft import WeightedNLL
from lmkd.train.builder import build_runtime


def test_fused_auto_falls_back_on_cpu(base_recipe, tiny_models):
    """Recipe tiny, token_kd fused: auto on CPU: loss equals fused: off (atol 1e-6); exactly 1 warning."""
    base_recipe["models"]["teacher"] = {
        "path": tiny_models["teacher"],
        "dtype": "fp32",
        "placement": "cpu",
    }
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    base_recipe["loss"] = {
        "terms": [{"criterion": "token_kd", "params": {"divergence": "gjsd", "beta": 0.5, "fused": "auto"}}]
    }
    base_recipe["engine"] = {"type": "local", "device": "cpu", "mixed_precision": "no"}

    # Run with fused: auto
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        rt_auto = build_runtime(recipe_from_dict(base_recipe))
        rt_auto.begin()
        batch = rt_auto.producer.next(rt_auto.state)
        rt_auto.algorithm.augment(batch, rt_auto)
        micro = batch.to("cpu")
        outs_auto = rt_auto.algorithm.loss(micro, rt_auto)
        loss_auto = list(outs_auto[0].out.terms.values())[0].value.item()
        rt_auto.end()

    # Check exactly 1 warning about fused fallback
    fused_warns = [w for w in recorded if "fused" in str(w.message).lower()]
    assert len(fused_warns) == 1, f"Expected 1 fused fallback warning, got {len(fused_warns)}: {fused_warns}"

    # Run with fused: off
    base_recipe["loss"]["terms"][0]["params"]["fused"] = "off"
    rt_off = build_runtime(recipe_from_dict(base_recipe))
    rt_off.begin()
    outs_off = rt_off.algorithm.loss(micro, rt_off)
    loss_off = list(outs_off[0].out.terms.values())[0].value.item()
    rt_off.end()

    assert abs(loss_auto - loss_off) < 1e-6


def test_fused_on_raises_with_weights(base_recipe):
    """weighted_nll fused: on -> ValueError stating 'token weights' as reason."""
    cfg = WeightedNLL.Config(fused="on")
    criterion = WeightedNLL(cfg)
    batch = {
        "student.input_ids": torch.tensor([[1, 2, 3]]),
        "student.labels": torch.tensor([[-100, 2, 3]]),
        "student.attention_mask": torch.tensor([[1, 1, 1]]),
        "seq.token_weights": torch.tensor([[1.0, 0.5, 0.5]]),
    }
    from unittest.mock import MagicMock

    from lmkd.core.batch import Batch
    from lmkd.losses.base import LossContext

    ctx = LossContext(
        batch=Batch(batch),
        student=MagicMock(),
        student_views=[],
        teacher=None,
        modules=MagicMock(),
        state=MagicMock(),
        rt=MagicMock(),
    )
    with pytest.raises(ValueError, match="token weights"):
        criterion.forward(ctx)
