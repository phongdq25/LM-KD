"""Parity tests for LUFFY baseline against verbatim reference code."""

from __future__ import annotations

import pytest
import torch
from tests.parity._ref.luffy_ref import (
    compute_grpo_outcome_advantage_split as ref_compute_grpo_split,
)
from tests.parity._ref.luffy_ref import (
    compute_sft_pure_loss as ref_compute_sft,
)
from tests.parity._ref.luffy_ref import (
    compute_token_on_off_policy_loss as ref_compute_luffy_loss,
)

from baselines.luffy import (
    compute_grpo_outcome_advantage_split,
    compute_sft_pure_loss,
    compute_token_on_off_policy_loss,
)

pytestmark = pytest.mark.parity


@pytest.mark.parametrize(
    "off_reshape",
    [
        "no_reshape",
        "p_div_p_0.1",
        "p_div_p_0.3",
        "p_div_p_0.5",
        "square_root",
        "pow",
        "logp",
        "p_logp",
    ],
)
@pytest.mark.parametrize("on_reshape", ["no_reshape", "p_div_p_0.1", "square_root", "pow"])
@pytest.mark.parametrize("loss_remove_clip", [False, True])
@pytest.mark.parametrize("loss_remove_token_mean", [False, True])
def test_luffy_loss_parity(off_reshape, on_reshape, loss_remove_clip, loss_remove_token_mean):
    torch.manual_seed(42)
    B, T = 4, 16

    # Log probabilities in valid ranges (< 0)
    raw_logp = torch.randn(B, T, requires_grad=True)
    log_prob = raw_logp.log_softmax(dim=-1)
    old_log_prob = torch.randn(B, T).log_softmax(dim=-1).detach()
    advantages = torch.randn(B, T).detach()

    # Mask: 1 for response tokens, 0 for pad
    eos_mask = torch.randint(1, 2, (B, T)).float()
    eos_mask[:, -3:] = 0.0  # simulate padding at the end

    # Prefix mask: 1 for off-policy demonstrations, 0 for on-policy rollouts
    prefix_mask = torch.zeros(B, T, dtype=torch.bool)
    prefix_mask[0] = True  # first row is off-policy demonstration
    prefix_mask[1, :8] = True  # second row has off-policy prefix

    actual = compute_token_on_off_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
        clip_upper_bound=1.0,
        prefix_mask=prefix_mask,
        off_policy_reshape=off_reshape,
        on_policy_reshape=on_reshape,
        loss_remove_clip=loss_remove_clip,
        loss_remove_token_mean=loss_remove_token_mean,
    )

    expected = ref_compute_luffy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
        clip_upper_bound=1.0,
        prefix_mask=prefix_mask,
        off_cliprange=0.2,
        off_policy_reshape=off_reshape,
        on_policy_reshape=on_reshape,
        loss_remove_clip=loss_remove_clip,
        loss_remove_token_mean=loss_remove_token_mean,
    )

    for key in ("pg_loss", "off_pg_loss", "on_pg_loss", "ppo_kl", "on_pg_clipfrac"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-5, msg=f"Mismatch for metric {key}")

    # Test backward pass / gradient parity
    g_actual = torch.autograd.grad(actual["pg_loss"], raw_logp, retain_graph=True)[0]
    g_expected = torch.autograd.grad(expected["pg_loss"], raw_logp)[0]
    torch.testing.assert_close(g_actual, g_expected, atol=1e-6, rtol=1e-5)


def test_luffy_loss_with_target_probs_and_clipping():
    torch.manual_seed(123)
    B, T = 3, 10
    raw_logp = torch.randn(B, T, requires_grad=True)
    log_prob = raw_logp.log_softmax(dim=-1)
    old_log_prob = torch.randn(B, T).log_softmax(dim=-1).detach()
    advantages = torch.randn(B, T).detach()
    eos_mask = torch.ones(B, T)
    prefix_mask = torch.tensor([[True] * 10, [False] * 10, [True] * 5 + [False] * 5])
    target_probs = torch.rand(B, T) + 0.01

    actual = compute_token_on_off_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
        clip_upper_bound=1.0,
        prefix_mask=prefix_mask,
        off_max_clip=2.0,
        off_min_clip=0.1,
        all_max_clip=0.9,
        target_probs=target_probs,
    )

    expected = ref_compute_luffy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
        clip_upper_bound=1.0,
        prefix_mask=prefix_mask,
        off_cliprange=0.2,
        off_max_clip=2.0,
        off_min_clip=0.1,
        all_max_clip=0.9,
        target_probs=target_probs,
    )

    torch.testing.assert_close(actual["pg_loss"], expected["pg_loss"], atol=1e-6, rtol=1e-5)
    g_actual = torch.autograd.grad(actual["pg_loss"], raw_logp, retain_graph=True)[0]
    g_expected = torch.autograd.grad(expected["pg_loss"], raw_logp)[0]
    torch.testing.assert_close(g_actual, g_expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("use_std", [False, True])
def test_grpo_split_advantage_parity(use_std):
    torch.manual_seed(99)
    B, T = 6, 8
    # 2 groups of 3 rollouts each
    uids = ["group_0", "group_0", "group_0", "group_1", "group_1", "group_1"]
    # In group_0: sample 0 is off-policy, samples 1,2 are on-policy
    # In group_1: sample 3 is off-policy, samples 4,5 are on-policy
    on_policy_mask = torch.tensor([False, True, True, False, True, True])
    eos_mask = torch.ones(B, T)
    token_rewards = torch.zeros(B, T)
    # Assign scalar rewards at index 0
    token_rewards[:, 0] = torch.tensor([1.0, 0.5, 0.8, 0.0, 1.0, 0.0])

    actual_adv, _ = compute_grpo_outcome_advantage_split(
        token_level_rewards=token_rewards,
        eos_mask=eos_mask,
        index=uids,
        on_policy_mask=on_policy_mask,
        epsilon=1e-6,
        use_std=use_std,
    )

    expected_adv, _ = ref_compute_grpo_split(
        token_level_rewards=token_rewards,
        eos_mask=eos_mask,
        index=uids,
        on_policy_mask=on_policy_mask,
        epsilon=1e-6,
        use_std=use_std,
    )

    torch.testing.assert_close(actual_adv, expected_adv, atol=1e-6, rtol=1e-5)


def test_sft_pure_loss_parity():
    torch.manual_seed(7)
    B, T = 4, 12
    log_prob = torch.randn(B, T).log_softmax(dim=-1)
    mask = torch.randint(0, 2, (B, T)).float()
    actual = compute_sft_pure_loss(log_prob, mask)
    expected = ref_compute_sft(log_prob, mask)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
