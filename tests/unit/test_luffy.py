"""Unit tests for LUFFY baseline components and criteria."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from baselines import luffy as baselines_luffy
from baselines.luffy import (
    LUFFY,
    GRPOSplit,
    LUFFYGroupProducer,
    LUFFYLoss,
    LUFFYPrefixWeight,
    luffy_loss,
)
from baselines.self_distill import luffy as self_distill_luffy
from lmkd.core.batch import Batch
from lmkd.core.registry import ADV_TRANSFORMS, ADVANTAGES, CRITERIONS, PRODUCERS
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.models.logits_view import LogitsView


def test_luffy_exports_and_aliases():
    """Verify that both baselines.luffy and baselines.self_distill.luffy export canonical components."""
    assert LUFFY is LUFFYLoss
    assert baselines_luffy.LUFFYLoss is LUFFYLoss
    assert self_distill_luffy.LUFFYLoss is LUFFYLoss
    assert baselines_luffy.GRPOSplit is GRPOSplit
    assert baselines_luffy.LUFFYPrefixWeight is LUFFYPrefixWeight
    assert baselines_luffy.LUFFYGroupProducer is LUFFYGroupProducer


def test_luffy_registry_discovery():
    """Verify that all LUFFY plugins are properly registered in the global registries."""
    assert "luffy" in CRITERIONS
    assert "luffy_loss" in CRITERIONS
    assert "grpo_split" in ADVANTAGES
    assert "luffy_prefix_weight" in ADV_TRANSFORMS
    assert "luffy_group" in PRODUCERS
    assert "luffy_rollout" in PRODUCERS


def test_luffy_criterion_forward():
    """Test forward pass of LUFFYLoss criterion via LossContext."""
    torch.manual_seed(42)
    B, T, V = 4, 8, 32

    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :2] = -100  # prompt / masked tokens
    adv = torch.randn(B, T)
    old_logp = torch.randn(B, T)
    prefix_mask = torch.tensor(
        [
            [True] * 8,
            [False] * 8,
            [True] * 4 + [False] * 4,
            [False] * 8,
        ]
    )

    batch = Batch(
        {
            "student.labels": labels,
            "student.attention_mask": torch.ones(B, T, dtype=torch.long),
            "rollout.advantage": adv,
            "rollout.old_logp": old_logp,
            "rollout.prefix_mask": prefix_mask,
        }
    )

    criterion = CRITERIONS.build(
        {
            "type": "luffy",
            "off_policy_reshape": "p_div_p_0.1",
            "loss_remove_clip": True,
            "loss_remove_token_mean": False,
        }
    )

    s_out = SimpleNamespace(logits=LogitsView(logits=logits))
    ctx = LossContext(
        batch=batch,
        student=s_out,
        student_views=[],
        teacher=None,
        modules=None,
        state=TrainState(),
        rt=None,
    )
    output = criterion(ctx)

    assert "pg" in output.terms
    loss_val = output.terms["pg"].value
    assert torch.isfinite(loss_val)
    assert loss_val.ndim == 0

    # Check metrics
    for m in ("off_pg_loss", "on_pg_loss", "approx_kl", "off_policy_prob", "on_policy_prob"):
        assert m in output.metrics
        assert torch.isfinite(output.metrics[m])


def test_luffy_criterion_sft_multitask_mode():
    """Test LUFFYLoss criterion with use_sft_multitask_loss=True."""
    torch.manual_seed(42)
    B, T, V = 4, 8, 32

    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    adv = torch.randn(B, T)
    old_logp = torch.randn(B, T)
    prefix_mask = torch.tensor(
        [
            [True] * 8,
            [False] * 8,
            [False] * 8,
            [False] * 8,
        ]
    )

    batch = Batch(
        {
            "student.labels": labels,
            "student.attention_mask": torch.ones(B, T, dtype=torch.long),
            "rollout.advantage": adv,
            "rollout.old_logp": old_logp,
            "rollout.prefix_mask": prefix_mask,
        }
    )

    criterion = CRITERIONS.build(
        {
            "type": "luffy",
            "use_sft_multitask_loss": True,
            "sft_loss_coef": 0.5,
        }
    )

    s_out = SimpleNamespace(logits=LogitsView(logits=logits))
    ctx = LossContext(
        batch=batch,
        student=s_out,
        student_views=[],
        teacher=None,
        modules=None,
        state=TrainState(),
        rt=None,
    )
    output = criterion(ctx)

    assert "pg" in output.terms
    assert "sft_loss" in output.metrics
    assert "on_pg_loss" in output.metrics
    assert torch.isfinite(output.terms["pg"].value)


def test_luffy_functional_convenience():
    """Test functional luffy_loss helper."""
    torch.manual_seed(42)
    B, T, V = 2, 6, 16
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    old_logp = torch.randn(B, T)
    adv = torch.randn(B, T)
    prefix_mask = torch.tensor([[True] * 6, [False] * 6])

    loss = luffy_loss(logits, old_logp, labels, adv, prefix_mask=prefix_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_grpo_split_plugin():
    """Test GRPOSplit advantage plugin with Batch input."""
    adv_plugin = ADVANTAGES.build({"type": "grpo_split", "norm_by_std": False})

    rewards = torch.tensor([1.0, 0.0, 0.5, 0.8])
    uids = ["q1", "q1", "q2", "q2"]
    on_policy = torch.tensor([False, True, True, True])

    batch = Batch(
        {"rollout.reward": rewards, "group.on_policy_mask": on_policy},
        meta={"group.uid": uids},
    )

    adv = adv_plugin(batch)
    assert adv.shape == rewards.shape
    # q1 on-policy mean is rewards[1] = 0.0. So rewards[0] - 0.0 = 1.0; rewards[1] - 0.0 = 0.0
    assert torch.isclose(adv[0], torch.tensor(1.0))
    assert torch.isclose(adv[1], torch.tensor(0.0))
    # q2 on-policy mean is (0.5 + 0.8) / 2 = 0.65
    assert torch.isclose(adv[2], torch.tensor(0.5 - 0.65))
    assert torch.isclose(adv[3], torch.tensor(0.8 - 0.65))


def test_luffy_prefix_weight_plugin():
    """Test LUFFYPrefixWeight advantage transform."""
    transform = ADV_TRANSFORMS.build({"type": "luffy_prefix_weight", "alpha": 2.0, "beta": 0.5})

    adv = torch.ones(2, 4)
    prefix_mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    batch = {"rollout.prefix_mask": prefix_mask}

    weighted = transform(adv, batch)
    expected = torch.tensor([[2.0, 2.0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
    torch.testing.assert_close(weighted, expected)
