from types import SimpleNamespace

import pytest
import torch
from tests.parity._ref.opsd_gjsd import generalized_jsd_loss as opsd_ref_gjsd
from tests.parity._ref.sdpo_sd_loss import compute_self_distillation_loss as sdpo_ref_sd_loss

from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.losses.functional.divergences import get_divergence
from lmkd.losses.topk_kd import TopKKD
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.providers import _fill_from_view
from lmkd.teachers.signals import TeacherOutput, TeacherSignalSpec


@pytest.mark.parity
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("clip", [None, 0.05])
def test_gjsd_parity(beta, clip):
    torch.manual_seed(42)
    B, T, V = 2, 8, 32
    s = torch.randn(B, T, V)
    t = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :2] = -100

    mask = (labels != -100).float()

    ref_val = opsd_ref_gjsd(
        s,
        t,
        labels=labels,
        beta=beta,
        temperature=1.0,
        reduction="batchmean",
        token_clip=clip,
    )

    our_fn = get_divergence("gjsd", beta=beta, token_clip=clip, clip_mode="elementwise")
    our_per_tok = our_fn(s, t)
    our_val = (our_per_tok * mask).sum() / mask.sum()

    assert torch.allclose(our_val, ref_val, atol=1e-5), (
        f"beta={beta}, clip={clip}: our={our_val.item()} != ref={ref_val.item()}"
    )


@pytest.mark.parity
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("tail", ["bucket", "renorm"])
@pytest.mark.parametrize("support", ["teacher", "student"])
def test_topk_kd_parity_sdpo(alpha, tail, support):
    torch.manual_seed(42)
    B, T, V, K = 2, 7, 30, 6
    s_logits = torch.randn(B, T, V)
    t_logits = torch.randn(B, T, V)

    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    labels[:, 1] = -100
    mask = (labels != -100).float()

    s_view = LogitsView(logits=s_logits)
    t_view = LogitsView(logits=t_logits)

    spec = TeacherSignalSpec(kind="topk", k=K, support=support, tail="none", temperature=1.0)
    batch = Batch({"student.labels": labels, "student.attention_mask": torch.ones(B, T, dtype=torch.long)})

    s_out = SimpleNamespace(logits=s_view)
    t_out = TeacherOutput(logits=t_view)
    _fill_from_view(t_out, batch, spec, "student", s_out)

    ctx = LossContext(
        batch=batch,
        student=s_out,
        student_views=[],
        teacher=t_out,
        modules=None,
        state=TrainState(),
        rt=None,
    )

    topk_kd = TopKKD(TopKKD.Config(alpha=alpha, k=K, support=support, tail=tail, temperature=1.0))
    loss_out = topk_kd(ctx)
    our_val = loss_out.terms["topk_kd"].value / mask.sum()

    # Reference SDPO calculation
    # Both student and teacher top-k log probabilities at t_out.topk_ids
    topk_ids = t_out.topk_ids
    t_topk_logp = t_out.topk_logp
    s_topk_logp = s_view.gather(topk_ids)

    ref_per_tok = sdpo_ref_sd_loss(
        student_topk_log_probs=s_topk_logp,
        teacher_topk_log_probs=t_topk_logp,
        alpha=alpha,
        distillation_topk=K,
        distillation_add_tail=(tail == "bucket"),
        full_logit_distillation=True,
    )
    ref_val = (ref_per_tok * mask).sum() / mask.sum()

    assert torch.allclose(our_val, ref_val, atol=1e-5), (
        f"alpha={alpha}, tail={tail}, support={support}: our={our_val.item()} != ref={ref_val.item()}"
    )
