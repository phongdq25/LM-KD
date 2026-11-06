"""Unit tests for Self-Play Fine-Tuning baselines:

SPIN, T-SPIN, SPACE, SPPO, S-SPPO, and RSPO.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from baselines import self_play
from baselines.self_play import (
    RSPO,
    SPACE,
    SPIN,
    SPPO,
    SSPPO,
    TSPIN,
    rspo_loss,
    s_sppo_loss,
    space_loss,
    spin_loss,
    sppo_loss,
    tspin_loss,
)
from lmkd.core.batch import Batch
from lmkd.core.registry import CRITERIONS
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherOutput


def test_self_play_exports():
    """Verify that baselines.self_play exports canonical criteria and loss primitives."""
    assert self_play.SPIN is SPIN
    assert self_play.TSPIN is TSPIN
    assert self_play.SPACE is SPACE
    assert self_play.SPPO is SPPO
    assert self_play.SSPPO is SSPPO
    assert self_play.RSPO is RSPO

    assert self_play.spin_loss is spin_loss
    assert self_play.tspin_loss is tspin_loss
    assert self_play.space_loss is space_loss
    assert self_play.sppo_loss is sppo_loss
    assert self_play.s_sppo_loss is s_sppo_loss
    assert self_play.rspo_loss is rspo_loss


def test_self_play_registry_discovery():
    """Verify that each self-play criterion is registered under exactly ONE canonical name."""
    expected = ("spin", "tspin", "space", "sppo", "s_sppo", "rspo")
    for name in expected:
        assert name in CRITERIONS, f"{name} not found in CRITERIONS"

    # Redundant / duplicate aliases must NOT be registered
    disallowed = (
        "spin_loss",
        "t_spin",
        "tspin_loss",
        "space_loss",
        "sppo_loss",
        "sspppo",
        "sspppo_loss",
        "s_sppo_loss",
        "rspo_loss",
        "apsp",
        "apsp_loss",
    )
    for name in disallowed:
        assert name not in CRITERIONS, f"Disallowed alias {name} should not be registered in CRITERIONS"


def _make_dummy_context(b_pairs: int, t_len: int, v_dim: int, group_size: int = 2):
    total_seqs = b_pairs * group_size
    logits = torch.randn(total_seqs, t_len, v_dim)
    labels = torch.randint(0, v_dim, (total_seqs, t_len))
    labels[:, :2] = -100  # mask prompt tokens
    ref_logp = torch.randn(total_seqs, t_len)

    batch_dict = {
        "student.labels": labels,
        "student.input_ids": labels.clamp_min(0),
        "group.size": torch.tensor([group_size] * total_seqs),
    }
    batch = Batch(batch_dict)
    s_out = SimpleNamespace(logits=LogitsView(logits=logits), last_hidden=torch.randn(total_seqs, t_len, 32))
    t_out = TeacherOutput(token_logp=ref_logp)
    ctx = LossContext(
        batch=batch,
        student=s_out,
        student_views=[],
        teacher=t_out,
        modules=None,
        state=TrainState(),
        rt=None,
    )
    return ctx


def test_spin_criterion_forward():
    ctx = _make_dummy_context(b_pairs=2, t_len=6, v_dim=16, group_size=2)
    criterion = CRITERIONS.build({"type": "spin", "beta": 0.1})
    out = criterion(ctx)

    assert "spin" in out.terms
    assert torch.isfinite(out.terms["spin"].value)
    for k in ("loss", "reward_chosen", "reward_rejected", "accuracy", "margin"):
        assert k in out.metrics
        assert torch.isfinite(out.metrics[k])


def test_tspin_criterion_forward():
    ctx_triplet = _make_dummy_context(b_pairs=2, t_len=6, v_dim=16, group_size=3)
    criterion = CRITERIONS.build({"type": "tspin", "alpha": 1.0, "beta": 0.1})
    out = criterion(ctx_triplet)

    assert "tspin" in out.terms
    assert torch.isfinite(out.terms["tspin"].value)
    assert "loss" in out.metrics
    assert "margin_u" in out.metrics

    ctx_pair = _make_dummy_context(b_pairs=2, t_len=6, v_dim=16, group_size=2)
    out_pair = criterion(ctx_pair)
    assert torch.isfinite(out_pair.terms["tspin"].value)


def test_space_criterion_forward():
    ctx = _make_dummy_context(b_pairs=2, t_len=6, v_dim=16, group_size=2)
    criterion = CRITERIONS.build({"type": "space", "mu": 1.0})
    out = criterion(ctx)

    assert "space" in out.terms
    assert torch.isfinite(out.terms["space"].value)
    assert "loss" in out.metrics
    assert "reward_chosen" in out.metrics


def test_sppo_criterion_forward():
    ctx = _make_dummy_context(b_pairs=2, t_len=6, v_dim=16, group_size=2)
    ctx.batch["pref.chosen_probs_win"] = torch.tensor([0.8, 0.8, 0.7, 0.7])
    ctx.batch["pref.chosen_probs_lose"] = torch.tensor([0.2, 0.2, 0.3, 0.3])

    criterion = CRITERIONS.build({"type": "sppo", "beta": 0.001})
    out = criterion(ctx)

    assert "sppo" in out.terms
    assert torch.isfinite(out.terms["sppo"].value)
    assert "loss" in out.metrics


def test_s_sppo_criterion_forward():
    ctx = _make_dummy_context(b_pairs=2, t_len=6, v_dim=16, group_size=2)
    ctx.batch["pref.sim_score"] = torch.tensor([0.6, 0.6, 0.3, 0.3])
    ctx.batch["pref.chosen_probs_win"] = torch.tensor([0.8, 0.8, 0.7, 0.7])
    ctx.batch["pref.chosen_probs_lose"] = torch.tensor([0.2, 0.2, 0.3, 0.3])

    criterion = CRITERIONS.build({"type": "s_sppo", "beta": 0.001, "lam": 0.5, "use_hidden_repulsion": True})
    out = criterion(ctx)

    assert "s_sppo" in out.terms
    assert torch.isfinite(out.terms["s_sppo"].value)
    assert "sim_score" in out.metrics


def test_rspo_criterion_forward():
    ctx = _make_dummy_context(b_pairs=2, t_len=6, v_dim=16, group_size=2)
    criterion = CRITERIONS.build({"type": "rspo", "beta": 0.001, "reg_coef": 0.5, "loss_type": "sppo_reversekl"})
    out = criterion(ctx)

    assert "rspo" in out.terms
    assert torch.isfinite(out.terms["rspo"].value)
    assert "reg_loss" in out.metrics


def test_functional_autograd_gradients():
    """Verify that all functional loss primitives support autograd backward."""
    torch.manual_seed(42)
    p_c = torch.randn(4, requires_grad=True)
    p_r = torch.randn(4, requires_grad=True)
    r_c = torch.randn(4)
    r_r = torch.randn(4)

    # 1. SPIN
    loss_spin, _, _ = spin_loss(p_c, p_r, r_c, r_r, beta=0.1)
    loss_spin.sum().backward()
    assert p_c.grad is not None and torch.isfinite(p_c.grad).all()
    assert p_r.grad is not None and torch.isfinite(p_r.grad).all()

    # 2. T-SPIN
    p_c.grad, p_r.grad = None, None
    p_proto = torch.randn(4, requires_grad=True)
    loss_tspin, _, _ = tspin_loss(p_c, p_r, p_proto, alpha=1.0, beta=0.1)
    loss_tspin.sum().backward()
    assert p_c.grad is not None and p_proto.grad is not None

    # 3. SPACE
    p_c.grad, p_r.grad = None, None
    loss_space, _, _ = space_loss(p_c, p_r, r_c, r_r, mu=1.0)
    loss_space.sum().backward()
    assert p_c.grad is not None and p_r.grad is not None

    # 4. SPPO
    p_c.grad, p_r.grad = None, None
    loss_sppo, _, _ = sppo_loss(p_c, p_r, r_c, r_r, beta=0.001)
    loss_sppo.sum().backward()
    assert p_c.grad is not None and p_r.grad is not None

    # 5. S-SPPO
    p_c.grad, p_r.grad = None, None
    sim = torch.tensor([0.2, 0.4, 0.6, 0.8])
    loss_s_sppo, _, _ = s_sppo_loss(p_c, p_r, r_c, r_r, sim_score=sim, beta=0.001)
    loss_s_sppo.sum().backward()
    assert p_c.grad is not None and p_r.grad is not None

    # 6. RSPO
    p_c.grad, p_r.grad = None, None
    loss_rspo, _, _, _ = rspo_loss(p_c, p_r, r_c, r_r, beta=0.001, reg_coef=0.5)
    loss_rspo.sum().backward()
    assert p_c.grad is not None and p_r.grad is not None
