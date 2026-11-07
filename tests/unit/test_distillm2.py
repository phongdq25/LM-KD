"""Unit tests for DistiLLM-2 baseline in same_tokenizer."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from baselines.same_tokenizer import (
    DistiLLM2,
    DistiLLM2Criterion,
    DistiLLM2Producer,
    compute_dynamic_alpha,
    distillm2,
    skewed_forward_kl,
    skewed_reverse_kl,
)
from lmkd.core.batch import Batch
from lmkd.core.registry import CRITERIONS, PRODUCERS
from lmkd.core.state import TrainState
from lmkd.data.collator import Collator
from lmkd.losses.base import LossContext
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.rollout.base import Rollout
from lmkd.teachers.signals import TeacherOutput


def _make_context(
    batch_dict: dict,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    step: int = 0,
    total_steps: int = 100,
) -> LossContext:
    batch = Batch(batch_dict)
    s_view = LogitsView(logits=student_logits)
    student_side = SimpleNamespace(logits=s_view, model=SimpleNamespace())
    if teacher_logits is not None:
        t_view = LogitsView(logits=teacher_logits)
        teacher_output = TeacherOutput(logits=t_view, side=SimpleNamespace(model=SimpleNamespace()))
    else:
        teacher_output = None

    state = TrainState(step=step, total_steps=total_steps)
    return LossContext(
        batch=batch,
        student=student_side,  # type: ignore[arg-type]
        student_views=[student_side],  # type: ignore[list-item]
        teacher=teacher_output,
        modules=ModuleRegistry(),
        state=state,
        rt=SimpleNamespace(),
    )


def test_distillm2_registered_and_exported():
    assert "distillm2" in CRITERIONS
    assert "distillm_v2" not in CRITERIONS
    assert CRITERIONS.get("distillm2") is DistiLLM2Criterion
    assert DistiLLM2 is DistiLLM2Criterion
    assert distillm2.DistiLLM2 is DistiLLM2Criterion
    assert distillm2.compute_dynamic_alpha is compute_dynamic_alpha
    assert distillm2.skewed_forward_kl is skewed_forward_kl
    assert distillm2.skewed_reverse_kl is skewed_reverse_kl

    # Producer (single registration)
    assert "distillm2" in PRODUCERS
    assert PRODUCERS.get("distillm2") is DistiLLM2Producer
    assert distillm2.DistiLLM2Producer is DistiLLM2Producer


def test_distillm2_mathematical_parity():
    """Verify skewed forward KL and skewed reverse KL against formulas in distillm_trainer.py."""
    torch.manual_seed(42)
    B, T, V = 2, 6, 20
    s_logits = torch.randn(B, T, V, requires_grad=True)
    t_logits = torch.randn(B, T, V)

    s_logp = s_logits.log_softmax(-1)
    t_logp = t_logits.log_softmax(-1)
    alpha = 0.1

    # Reference implementation of skewed forward KL (distillm_trainer.py lines 1157-1160)
    log_a = math.log(alpha)
    log_1_a = math.log(1.0 - alpha)
    ref_mix_1 = torch.logsumexp(torch.stack([log_a + t_logp, log_1_a + s_logp], dim=0), dim=0)
    ref_tea_pos_kl = (t_logp.exp() * (t_logp - ref_mix_1)).sum(-1)

    our_fkl = skewed_forward_kl(s_logits, t_logits, alpha=alpha)
    entropy_p = -(t_logp.exp() * t_logp).sum(-1)
    assert torch.allclose(our_fkl - entropy_p, ref_tea_pos_kl, atol=1e-5)

    # Reference implementation of skewed reverse KL (distillm_trainer.py lines 1195-1199)
    ref_mix_2 = torch.logsumexp(torch.stack([log_1_a + t_logp, log_a + s_logp], dim=0), dim=0)
    ref_ref_pos_kl = (s_logp.exp() * (s_logp - ref_mix_2)).sum(-1)

    our_rkl = skewed_reverse_kl(s_logits, t_logits, alpha=alpha)
    assert torch.allclose(our_rkl, ref_ref_pos_kl, atol=1e-5)


def test_compute_dynamic_alpha():
    diff = torch.tensor([0.2, 0.5])
    assert compute_dynamic_alpha(diff, base_alpha=0.1, anchor=None) == 0.1

    dyn_a = compute_dynamic_alpha(diff, base_alpha=0.1, anchor=0.3)
    assert isinstance(dyn_a, torch.Tensor)
    assert dyn_a.shape == (2, 1, 1)
    assert (dyn_a >= 0.01).all() and (dyn_a <= 0.1).all()


def test_distillm2_criterion_forward_backward():
    torch.manual_seed(42)
    B, T, V = 4, 10, 30  # 4 sequences = 2 pairs interleaved: [c0, r0, c1, r1]
    s_logits = torch.randn(B, T, V, requires_grad=True)
    t_logits = torch.randn(B, T, V)

    labels = torch.randint(0, V, (B, T))
    labels[:, :3] = -100  # mask prompt tokens

    ctx = _make_context(
        batch_dict={"student.labels": labels},
        student_logits=s_logits,
        teacher_logits=t_logits,
        step=10,
        total_steps=100,
    )

    crit = DistiLLM2Criterion(
        DistiLLM2Criterion.Config(
            beta=1.0,
            base_alpha_1=0.1,
            base_alpha_2=0.1,
            gradual_beta=True,
            update_alpha=True,
        )
    )

    out = crit.forward(ctx)
    assert "distillm2" in out.terms
    loss_val = out.terms["distillm2"].value
    assert loss_val.ndim == 0
    assert loss_val > 0

    assert "loss/chosen" in out.metrics
    assert "loss/rejected" in out.metrics
    assert "beta" in out.metrics
    assert math.isclose(out.metrics["beta"].item(), 1.1, rel_tol=1e-4)

    # Check backward pass
    loss_val.backward()
    assert s_logits.grad is not None
    assert torch.isfinite(s_logits.grad).all()
    assert s_logits.grad.abs().sum() > 0

    # Extra state roundtrip
    sd = crit.extra_state()
    assert sd["logp_logq"] is not None
    assert sd["logq_logp"] is not None

    crit2 = DistiLLM2Criterion()
    crit2.load_extra_state(sd)
    assert crit2.logp_logq == sd["logp_logq"]
    assert crit2.logq_logp == sd["logq_logp"]


def test_distillm2_criterion_validation_errors():
    crit = DistiLLM2Criterion()

    # Odd batch size
    ctx_odd = _make_context(
        batch_dict={"student.labels": torch.randint(0, 20, (3, 8))},
        student_logits=torch.randn(3, 8, 20),
        teacher_logits=torch.randn(3, 8, 20),
    )
    with pytest.raises(ValueError, match="even batch size"):
        crit.forward(ctx_odd)

    # Missing teacher
    ctx_no_teacher = _make_context(
        batch_dict={"student.labels": torch.randint(0, 20, (2, 8))},
        student_logits=torch.randn(2, 8, 20),
        teacher_logits=None,
    )
    with pytest.raises(ValueError, match="requires online teacher"):
        crit.forward(ctx_no_teacher)


class _MockRollout:
    def __init__(self):
        self.call_count = 0
        self.stale_count = 0

    def mark_stale(self):
        self.stale_count += 1

    def generate(self, prompts, spec):
        self.call_count += 1
        return [[Rollout(prompt_ids=p, response_ids=[10, 20])] for p in prompts]


def test_distillm2_producer_lifecycle():
    """Verify that DistiLLM2Producer regenerates responses only at epoch boundaries."""
    samples = [
        {
            "student.input_ids": [1, 2, 3, 4],
            "student.labels": [-100, -100, 3, 4],
            "student.attention_mask": [1, 1, 1, 1],
            "student.prompt_len": 2,
            "meta.id": f"prompt_{i}",
        }
        for i in range(4)
    ]
    collator = Collator(pad_token_ids={"student": 0, "teacher": 0}, pad_to_multiple_of=None)
    rollout = _MockRollout()
    rt = SimpleNamespace(
        rollout=rollout,
        train_dataset=samples,
        collator=collator,
        engine=SimpleNamespace(world_size=1, rank=0, device=torch.device("cpu")),
        config=SimpleNamespace(
            train=SimpleNamespace(batch_size=4, seed=42),
            data=SimpleNamespace(max_length=128, pin_memory=False, num_workers=0),
        ),
    )

    prod = DistiLLM2Producer()
    prod.setup(rt)
    assert prod.steps_per_epoch() == 2
    state = TrainState()

    # Step 1: Epoch 0 starts -> generates completions for epoch 0
    b1 = prod.next(state)
    assert rollout.call_count == 1
    assert rollout.stale_count == 1
    assert b1.batch_size == 4
    assert b1["group.size"].tolist() == [2, 2, 2, 2]
    assert b1["group.index"].tolist() == [0, 1, 0, 1]
    assert state.epoch == pytest.approx(0.5)

    # Step 2: Step 2 of Epoch 0 -> DOES NOT call rollout during epoch steps
    b2 = prod.next(state)
    assert rollout.call_count == 1
    assert b2.batch_size == 4
    assert state.epoch == pytest.approx(1.0)

    # Step 3: Epoch 0 ends -> epoch boundary reached -> regenerates for Epoch 1
    b3 = prod.next(state)
    assert rollout.call_count == 2
    assert rollout.stale_count == 2
    assert prod.epoch == 1
    assert prod.batches_done == 1
    assert state.epoch == pytest.approx(1.5)

    # State dict roundtrip
    sd = prod.state_dict()
    assert sd["epoch"] == 1
    assert sd["batches_done"] == 1

    prod2 = DistiLLM2Producer()
    prod2.setup(rt)
    prod2.load_state_dict(sd)
    assert prod2.epoch == 1
    assert prod2.batches_done == 1
