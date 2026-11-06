"""Unit tests for Self-Distillation baselines (OPSD, SDPO, RLSD)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from baselines.self_distill import (
    OPSD,
    RLSD,
    SDPO,
    STGCA,
    LambdaSchedule,
    OPSDConfig,
    PPOClip,
    ReasonFirst,
    RLSDConfig,
    RLSDLoss,
    RLSDReweight,
    SDPOConfig,
    SDPODistill,
    SDPOLoss,
    SDPOReprompt,
    TinkerRKL,
    TopKKD,
    add_tail,
    compute_self_distillation_loss,
    effective_lambda,
    generalized_jsd,
    grpo_loss,
    masked_mean,
    opsd_tinker_loss,
    ppo_clip_per_token,
    renorm_topk,
    rlsd_reweight,
    sdpo_per_token,
)
from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherOutput


def test_baselines_self_distill_exports():
    """Verify that baselines.self_distill exports all canonical criteria and components."""
    # OPSD
    assert OPSD is not None
    assert TinkerRKL is not None
    assert ReasonFirst is not None
    assert OPSDConfig is not None
    assert generalized_jsd is not None
    assert opsd_tinker_loss is not None

    # SDPO
    assert SDPO is not None
    assert SDPOLoss is not None
    assert SDPODistill is not None
    assert TopKKD is not None
    assert SDPOReprompt is not None
    assert SDPOConfig is not None
    assert sdpo_per_token is not None
    assert compute_self_distillation_loss is not None
    assert add_tail is not None

    # RLSD
    assert RLSD is not None
    assert RLSDLoss is not None
    assert PPOClip is not None
    assert RLSDReweight is not None
    assert STGCA is not None
    assert LambdaSchedule is not None
    assert RLSDConfig is not None
    assert ppo_clip_per_token is not None
    assert grpo_loss is not None
    assert rlsd_reweight is not None

    # Common
    assert masked_mean is not None
    assert renorm_topk is not None
    assert effective_lambda is not None


def test_opsd_criterion_forward():
    """Verify OPSD criterion computes Generalized JSD on student and teacher logits."""
    torch.manual_seed(42)
    B, T, V = 2, 4, 16
    s_logits = torch.randn(B, T, V)
    t_logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :1] = -100

    batch = Batch({"student.labels": labels, "student.attention_mask": torch.ones(B, T, dtype=torch.long)})
    s_out = SimpleNamespace(logits=LogitsView(logits=s_logits))
    t_out = TeacherOutput(logits=LogitsView(logits=t_logits))

    ctx = LossContext(
        batch=batch,
        student=s_out,
        student_views=[],
        teacher=t_out,
        modules=None,
        state=TrainState(),
        rt=None,
    )

    criterion = OPSD(OPSD.Config(beta=0.0, token_clip=0.05, temperature=1.0))
    loss_out = criterion(ctx)

    assert "opsd" in loss_out.terms
    assert torch.isfinite(loss_out.terms["opsd"].value)
    assert loss_out.terms["opsd"].reduction == "token_sum"
    assert "opsd/loss_mean" in loss_out.metrics


def test_sdpo_loss_forward():
    """Verify SDPO criterion forward pass."""
    torch.manual_seed(42)
    B, T, V = 2, 4, 16
    s_logits = torch.randn(B, T, V)
    t_logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :1] = -100

    batch = Batch({
        "student.labels": labels,
        "student.attention_mask": torch.ones(B, T, dtype=torch.long),
        "rollout.has_signal": torch.tensor([1.0, 1.0]),
        "rollout.old_logp": torch.randn(B, T),
    })
    s_out = SimpleNamespace(logits=LogitsView(logits=s_logits))
    t_out = TeacherOutput(logits=LogitsView(logits=t_logits))

    ctx = LossContext(
        batch=batch,
        student=s_out,
        student_views=[],
        teacher=t_out,
        modules=None,
        state=TrainState(),
        rt=None,
    )

    criterion = SDPO(SDPO.Config(alpha=0.5, k=8, is_clip=2.0))
    loss_out = criterion(ctx)

    assert "distill" in loss_out.terms
    assert torch.isfinite(loss_out.terms["distill"].value)
    assert "sdpo/distill_mean" in loss_out.metrics
    assert "sdpo/ratio_mean" in loss_out.metrics


def test_rlsd_loss_forward():
    """Verify RLSD criterion forward pass."""
    torch.manual_seed(42)
    B, T, V = 2, 4, 16
    s_logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :1] = -100

    batch = Batch({
        "student.labels": labels,
        "student.attention_mask": torch.ones(B, T, dtype=torch.long),
        "rollout.advantage": torch.randn(B, T),
        "rollout.old_logp": torch.randn(B, T),
    })
    s_out = SimpleNamespace(logits=LogitsView(logits=s_logits))

    ctx = LossContext(
        batch=batch,
        student=s_out,
        student_views=[],
        teacher=None,
        modules=None,
        state=TrainState(),
        rt=None,
    )

    criterion = RLSD(RLSD.Config(clip_low=0.2, clip_high=0.28))
    loss_out = criterion(ctx)

    assert "pg" in loss_out.terms
    assert torch.isfinite(loss_out.terms["pg"].value)
    assert "clipfrac" in loss_out.metrics
    assert "approx_kl" in loss_out.metrics


def test_rlsd_reweight_transform():
    """Verify RLSDReweight plugin executes reweighting properly."""
    torch.manual_seed(42)
    B, T = 2, 4
    adv = torch.randn(B, T)
    teacher_logp = torch.randn(B, T)
    student_logp = torch.randn(B, T)
    labels = torch.tensor([[1, 2, 3, 4], [-100, 2, 3, 4]])

    batch = Batch({"student.labels": labels})
    plugin = RLSDReweight(RLSDReweight.Config(lam=0.5, clip=0.2))

    class MockState:
        step = 10
        def __init__(self):
            self.logged = {}
        def log(self, k, v):
            self.logged[k] = v

    state = MockState()
    eff_adv = plugin(adv, batch, teacher_logp, student_logp, state)

    assert eff_adv.shape == (B, T)
    assert torch.isfinite(eff_adv).all()
    assert "rlsd/effective_lambda" in state.logged
