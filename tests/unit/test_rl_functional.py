"""T25: group advantages, RLSD weighting and clipped policy gradients."""

import math

import pytest
import torch

from baselines.self_distill import effective_lambda, rlsd_reweight, sdpo_per_token
from lmkd.losses.functional.rl import ppo_clip_per_token
from lmkd.rl.advantages import grpo_advantage


@pytest.mark.parametrize("normalize", [False, True])
def test_grpo_advantage(normalize):
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 3.0])
    result = grpo_advantage(rewards, ["a", "a", "b", "b", "c"], normalize)
    expected = torch.tensor([0.5, -0.5, 0.5, -0.5, 3.0])
    if normalize:
        expected[:4] /= math.sqrt(0.5) + 1e-6
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("step,expected", [(5, 0.25), (10, 0.5), (20, 0.25), (40, 0.0)])
def test_rlsd_lambda_schedule(step, expected):
    assert effective_lambda(0.5, step, 10, 20) == expected


def test_ppo_clip_hand():
    logp = torch.tensor([math.log(1.5), math.log(0.5)], requires_grad=True)
    loss, clip, kl = ppo_clip_per_token(logp, torch.zeros(2), torch.tensor([1.0, -1.0]), 0.2, 0.2)
    torch.testing.assert_close(loss, torch.tensor([-1.2, 0.8]))
    assert clip.tolist() == [1.0, 1.0]
    torch.testing.assert_close(kl, -logp)
    loss.sum().backward()
    assert torch.equal(logp.grad, torch.zeros(2))


def test_ppo_dual_clip_and_log_ratio_bound():
    loss, _, _ = ppo_clip_per_token(torch.tensor([100.0]), torch.zeros(1), -torch.ones(1), 0.2, 0.28, 3.0)
    assert loss.item() == 3.0


def test_rlsd_lam_zero_and_detached_weights():
    adv = torch.tensor([[1.0, -1.0, 0.0]], requires_grad=True)
    teacher = torch.ones_like(adv, requires_grad=True)
    result, _ = rlsd_reweight(adv, teacher, torch.zeros_like(adv), torch.ones_like(adv), 0.0, 0.2)
    torch.testing.assert_close(result, adv)
    result.sum().backward()
    assert teacher.grad is None


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_sdpo_identical_distribution(alpha):
    s = torch.randn(2, 3, 7).log_softmax(-1).requires_grad_()
    result = sdpo_per_token(s, s.detach(), alpha)
    torch.testing.assert_close(result, torch.zeros(2, 3), atol=1e-6, rtol=0)
    result.sum().backward()
    assert torch.isfinite(s.grad).all()
