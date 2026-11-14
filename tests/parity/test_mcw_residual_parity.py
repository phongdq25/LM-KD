"""Source parity on fixed, small fp32 tensors without network access."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tests.parity._ref.etp_ref import dual_ref, etp_ref
from tests.parity._ref.mcw_ref import context_ref, hidden_ref, logits_ref
from tests.parity._ref.residual_ref import pretrain_ref, residual_core_ref

from baselines.cross_tokenizer.mcw_kd import (
    ETP,
    MultiCostSemiDualOT,
    OTHidden,
    OTLogits,
    context_windows,
)
from baselines.cross_tokenizer.residualkd import ProjectorTA, ResidualKD, ResidualPretrain, residual_core
from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.data.aligners.dtw_edit import DtwEdit
from lmkd.losses.base import LossContext, ModelPairInfo
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


def _context(s, t, batch_extra=None):
    s_logits = torch.randn(1, len(s), 9, requires_grad=True)
    t_logits = torch.randn(1, len(t), 9)
    student = SimpleNamespace(logits=LogitsView(logits=s_logits), hidden=lambda _: s[None])
    teacher_side = SimpleNamespace(logits=LogitsView(logits=t_logits), hidden=lambda _: t[None])
    batch = Batch(
        {
            "student.attention_mask": torch.ones(1, len(s), dtype=torch.long),
            "teacher.attention_mask": torch.ones(1, len(t), dtype=torch.long),
            "student.labels": torch.tensor([[-100, 1, 2, 3, 4][: len(s)]]),
        }
    )
    for key, value in (batch_extra or {}).items():
        batch[key] = value
    teacher = TeacherOutput(logits=teacher_side.logits, side=teacher_side)
    modules = ModuleRegistry()
    ctx = LossContext(batch, student, [student], teacher, modules, TrainState(total_steps=10), SimpleNamespace())
    return ctx, s_logits, t_logits


def test_etp_parity():
    torch.manual_seed(9)
    cost = torch.rand(7, 5)
    got, transport = ETP()(cost)
    expected, reference_transport = etp_ref(cost)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(transport, reference_transport, atol=1e-5, rtol=0)
    torch.testing.assert_close(transport.sum(1), torch.full((7,), 1 / 7), atol=1e-5, rtol=0)


def test_ot_logits_parity():
    torch.manual_seed(4)
    s = torch.randn(4, 6, requires_grad=True)
    t = torch.randn(4, 6)
    ctx, sl, tl = _context(s, t)
    crit = OTLogits(OTLogits.Config(top_k_vocab=5, update_cost_weights=False))
    crit.build_modules(ModelPairInfo(6, 1, 9, 6, 1, 9), ctx.modules)
    got = crit(ctx).terms["ot_logits"].value
    sal = ctx.modules.get("mcw__salience")
    proj = ctx.modules.get("mcw__ot")
    expected = logits_ref(sl[0], tl[0], s, t, sal, proj, crit.cost_w, k=5)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)


def test_ot_hidden_parity():
    torch.manual_seed(7)
    s = torch.randn(4, 6, requires_grad=True)
    t = torch.randn(5, 6)
    student_tokens = ["<s>", "red", "blue", "</s>"]
    teacher_tokens = ["<s>", "re", "d", "blue", "</s>"]
    align_ctx = SimpleNamespace(
        student_tokenizer=SimpleNamespace(convert_ids_to_tokens=lambda ids: [student_tokens[i] for i in ids]),
        teacher_tokenizer=SimpleNamespace(convert_ids_to_tokens=lambda ids: [teacher_tokens[i] for i in ids]),
    )
    sample = {
        "student.input_ids": list(range(4)),
        "student.prompt_len": 1,
        "teacher.input_ids": list(range(5)),
        "teacher.prompt_len": 1,
    }
    aligned = DtwEdit(DtwEdit.Config(full_sequence=True))._mcw(sample, align_ctx)
    assert [0, 0] in aligned["align.dtw_s2t"]
    extra = {key: [torch.as_tensor(value)] for key, value in aligned.items()}
    ctx, _, _ = _context(s, t, extra)
    crit = OTHidden(OTHidden.Config(window=1, update_cost_weights=False))
    crit.build_modules(ModelPairInfo(6, 1, 9, 6, 1, 9), ctx.modules)
    got = crit(ctx).terms["ot_hidden"].value
    projected = ctx.modules.get("mcw__ot")(t)
    expected = hidden_ref(
        s,
        projected,
        aligned["align.dtw_s2t"],
        aligned["align.dtw_s2t_cost"],
        aligned["align.dtw_t2s"],
        aligned["align.dtw_t2s_cost"],
        crit.cost_w,
        window=1,
    )
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(context_windows(s, 1), context_ref(s, 1), atol=1e-5, rtol=0)


def test_dual_parity():
    torch.manual_seed(11)
    actual = MultiCostSemiDualOT(3, 4, 4, hidden_dim=8, num_inner_iters=2)
    reference = copy.deepcopy(actual)
    x, y = torch.rand(5, 4), torch.rand(3, 4)
    costs = [torch.rand(5, 3) for _ in range(3)]
    loss, each = actual(x, y, costs)
    expected, expected_each = dual_ref(reference, x, y, costs)
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=0)
    for value, target in zip(each, expected_each):
        torch.testing.assert_close(value, target, atol=1e-5, rtol=0)
    for param, ref_param in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(param, ref_param, atol=1e-5, rtol=0)


def test_residual_pretrain_parity():
    torch.manual_seed(12)
    hidden = torch.randn(2, 4, 6)
    head = nn.Linear(6, 9, bias=False)
    model = SimpleNamespace(lm_head=head)
    adapter = SimpleNamespace(lm_head=lambda _: head)
    side = SimpleNamespace(hidden=lambda _: hidden)
    labels = torch.tensor([[-100, -100, 3, 4], [-100, 2, 5, 6]])
    modules = ModuleRegistry()
    crit = ResidualPretrain(ResidualPretrain.Config(d_bottleneck=4))
    crit.build_modules(ModelPairInfo(6, 1, 9), modules)
    ctx = LossContext(
        Batch({"student.labels": labels}),
        side,
        [side],
        None,
        modules,
        TrainState(),
        SimpleNamespace(student_model=model, student_adapter=adapter),
    )
    got = crit(ctx).terms["pretrain"].value / labels.ne(-100).sum()
    expected = pretrain_ref(hidden, labels, modules.get("residual_pretrain__ta"), head)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)


def test_mixed_dtype_projector_boundaries():
    student_hidden = torch.randn(4, 6, dtype=torch.bfloat16, requires_grad=True)
    teacher_hidden = torch.randn(4, 6, dtype=torch.bfloat16)
    pairs = torch.tensor([[i, i] for i in range(4)])
    costs = torch.ones(4)
    extra = {
        "align.dtw_s2t": [pairs],
        "align.dtw_s2t_cost": [costs],
        "align.dtw_t2s": [pairs],
        "align.dtw_t2s_cost": [costs],
    }
    ctx, _, _ = _context(student_hidden, teacher_hidden, extra)
    logits = OTLogits(OTLogits.Config(top_k_vocab=5, update_cost_weights=False))
    hidden = OTHidden(OTHidden.Config(update_cost_weights=False))
    info = ModelPairInfo(6, 1, 9, 6, 1, 9)
    logits.build_modules(info, ctx.modules)
    hidden.build_modules(info, ctx.modules)
    total = logits(ctx).terms["ot_logits"].value + hidden(ctx).terms["ot_hidden"].value
    total.backward()
    assert student_hidden.grad is not None

    head = nn.Linear(6, 9, bias=False).to(torch.bfloat16)
    model = SimpleNamespace(lm_head=head)
    ctx.rt = SimpleNamespace(student_model=model, student_adapter=SimpleNamespace(lm_head=lambda _: head))
    ctx.batch["student.labels"] = torch.tensor([[-100, 1, 2, 3]])
    residual = ResidualKD(ResidualKD.Config(d_bottleneck=4))
    residual.build_modules(info, ctx.modules)
    output = residual(ctx)
    assert torch.isfinite(output.terms["res"].value)
    output.terms["res"].value.backward()


@pytest.mark.parametrize("cross", [False, True], ids=["same", "cross"])
def test_residual_parity(cross):
    torch.manual_seed(12)
    hs, ht = torch.randn(2, 4, 6), torch.randn(2, 4, 8)
    ta, sa, as_ = ProjectorTA(8, 4), nn.Linear(6, 4, bias=False), nn.Linear(4, 6, bias=False)
    nn.init.normal_(as_.weight, std=0.1)
    labels = torch.tensor([[-100, 2, 3, -100], [-100, 2, 4, -100]])
    logits = torch.randn(2, 4, 9)
    got = residual_core(hs, ht, ta, sa, as_, labels, logits, cross, 0.5, None)
    expected = residual_core_ref(hs, ht, sa, ta, as_, labels, logits, cross)
    for actual, reference in zip(got, expected):
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=0)
