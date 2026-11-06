"""Unit and parity tests for self-teacher, weight policies, prompt builders, and tinker_rkl."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tests.parity._ref.opsd_ref import opsd_tinker_loss_ref

from baselines.self_distill.opsd import TinkerRKL
from lmkd.core.batch import Batch
from lmkd.core.config import recipe_from_dict
from lmkd.core.state import TrainState
from lmkd.data.labels import make_labels
from lmkd.losses.base import LossContext
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.input_builders import PrivilegedPrompt
from lmkd.teachers.providers import HFFrozenTeacher
from lmkd.teachers.signals import TeacherOutput, TeacherSignalSpec
from lmkd.teachers.weight_policies import EMA, LoraDisabled, PeriodicSync
from lmkd.train.builder import build_runtime


def test_self_response_shift_matches_hf_frozen(base_recipe, tiny_models):
    """With privileged prompt on teacher.* and live policy, self teacher token_logp
    matches hf_frozen(position_map=response_shift) on the same weights."""
    base_recipe["models"]["teacher"] = {"path": tiny_models["student"], "dtype": "fp32"}
    base_recipe["teacher"] = {
        "provider": {"type": "self", "weight_policy": "live", "position_map": "response_shift"},
        "input": "same",
    }
    rt = build_runtime(recipe_from_dict(base_recipe))

    resp = [10, 11, 12, 1]
    s_ids = [3, 4] + resp
    t_ids = [5, 6, 7, 8, 9] + resp

    sample = lambda ids, plen, side: {
        f"{side}.input_ids": ids,
        f"{side}.attention_mask": [1] * len(ids),
        f"{side}.labels": make_labels(ids, plen).tolist(),
        f"{side}.prompt_len": plen,
    }
    b = rt.collator([{**sample(s_ids, 2, "student"), **sample(t_ids, 5, "teacher")}])

    out_self = rt.teacher.signals(b, TeacherSignalSpec(kind="sampled_logp"), None, None)

    # Compare against hf_frozen teacher
    hf_teacher = HFFrozenTeacher(HFFrozenTeacher.Config(position_map="response_shift"))
    hf_teacher.setup(rt)
    out_hf = hf_teacher.signals(b, TeacherSignalSpec(kind="sampled_logp"), None, None)

    assert out_self is not None and out_hf is not None
    assert torch.allclose(out_self.token_logp, out_hf.token_logp, atol=1e-5)


def test_privileged_prompt_keeps_response(base_recipe, tiny_models):
    """PrivilegedPrompt constructs teacher batch preserving student response tokens."""
    rt = build_runtime(recipe_from_dict(base_recipe))
    tok = rt.student_tokenizer

    # Create a batch with 2 rows and metadata
    s_resp_1 = [10, 11, 12, 1]
    s_resp_2 = [13, 14, 1]
    s_ids_1 = [2, 3] + s_resp_1
    s_ids_2 = [4] + s_resp_2

    rows = [
        {
            "student.input_ids": s_ids_1,
            "student.attention_mask": [1] * len(s_ids_1),
            "student.labels": make_labels(s_ids_1, 2).tolist(),
            "student.prompt_len": 2,
            "meta.problem": "solve 1",
            "meta.solution": "sol 1",
        },
        {
            "student.input_ids": s_ids_2,
            "student.attention_mask": [1] * len(s_ids_2),
            "student.labels": make_labels(s_ids_2, 1).tolist(),
            "student.prompt_len": 1,
            "meta.problem": "solve 2",
            "meta.solution": "sol 2",
        },
    ]
    batch = rt.collator(rows)

    builder = PrivilegedPrompt(
        PrivilegedPrompt.Config(
            template="Problem: {problem}\nSolution: {solution}",
            fields=["solution"],
            use_chat_template=False,
        )
    )
    builder.setup(rt)
    res_batch = builder.build(batch, rt)

    for b in range(2):
        sp = int(res_batch["student.prompt_len"][b].item())
        tp = int(res_batch["teacher.prompt_len"][b].item())
        s_mask = res_batch["student.attention_mask"][b]
        s_len = int(s_mask.sum().item())
        resp_len = s_len - sp

        # Verify teacher prompt is masked with -100
        assert (res_batch["teacher.labels"][b, :tp] == -100).all()

        # Invariant: teacher response tokens match student response tokens
        s_resp = res_batch["student.input_ids"][b, sp : sp + resp_len]
        t_resp = res_batch["teacher.input_ids"][b, tp : tp + resp_len]
        assert torch.equal(s_resp, t_resp)

        # Padding checks
        t_mask = res_batch["teacher.attention_mask"][b]
        t_len = int(t_mask.sum().item())
        if t_len < res_batch["teacher.input_ids"].shape[1]:
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
            assert (res_batch["teacher.input_ids"][b, t_len:] == pad_id).all()
            assert (res_batch["teacher.labels"][b, t_len:] == -100).all()


def test_self_live_equals_student(base_recipe, tiny_models):
    """With teacher.input=same and weight_policy=live, teacher.token_logp == student target_logp."""
    base_recipe["teacher"] = {
        "provider": {"type": "self", "weight_policy": "live", "position_map": "identity"},
        "input": "same",
    }
    rt = build_runtime(recipe_from_dict(base_recipe))

    sample = {
        "student.input_ids": [2, 3, 10, 11, 1],
        "student.attention_mask": [1, 1, 1, 1, 1],
        "student.labels": [-100, -100, 10, 11, 1],
        "student.prompt_len": 2,
    }
    b = rt.collator([sample])

    from lmkd.models.forward import ForwardRunner, SideRequest, side_inputs

    s_outs = ForwardRunner.run(
        rt.student_model, rt.student_adapter, side_inputs(b, "student"), SideRequest(logits="view"), grad=False
    )
    s_target_logp = s_outs[0].logits.target_logp(b["student.labels"])

    out_self = rt.teacher.signals(b, TeacherSignalSpec(kind="sampled_logp"), None, s_outs[0])
    assert out_self is not None
    assert torch.allclose(out_self.token_logp, s_target_logp, atol=1e-6)


def test_lora_disabled_equals_base(base_recipe, tiny_models):
    """Inside lora_disabled context, logits match base model; outside context, LoRA is restored."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base_path = tiny_models["student"]
    base_model = AutoModelForCausalLM.from_pretrained(base_path)
    base_model.eval()

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        init_lora_weights=False,  # Non-zero init so LoRA output differs from base
        target_modules=["c_attn"],
    )
    student_model = AutoModelForCausalLM.from_pretrained(base_path)
    peft_student = get_peft_model(student_model, lora_cfg)

    base_recipe["models"]["student"]["path"] = base_path
    rt = build_runtime(recipe_from_dict(base_recipe))
    rt.student_model = peft_student

    policy = LoraDisabled(LoraDisabled.Config())
    policy.setup(rt)

    inp = torch.tensor([[3, 4, 10, 11]])

    with torch.no_grad():
        base_out = base_model(inp).logits
        lora_out = peft_student(inp).logits

    # Verify LoRA produces different logits from base
    assert not torch.allclose(base_out, lora_out, atol=1e-4)

    # Within context, logits should match base model
    with policy.context(rt) as m:
        with torch.no_grad():
            ctx_out = m(inp).logits
        assert torch.allclose(ctx_out, base_out, atol=1e-5)

    # After context, logits should match LoRA model again
    with torch.no_grad():
        restored_out = peft_student(inp).logits
    assert torch.allclose(restored_out, lora_out, atol=1e-6)


def test_self_teacher_and_trust_region_with_lora_on_lm_head(base_recipe, tiny_models):
    """When LoRA targets lm_head, SelfTeacher and TrustRegion evaluate base head weights correctly."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from lmkd.models.forward import SideRequest
    from lmkd.teachers.providers import SelfTeacher

    base_path = tiny_models["student"]
    student_model = AutoModelForCausalLM.from_pretrained(base_path)

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        init_lora_weights=False,  # Non-zero init
        target_modules=["c_attn", "lm_head"],
    )
    peft_student = get_peft_model(student_model, lora_cfg)

    base_recipe["models"]["student"]["path"] = base_path
    rt = build_runtime(recipe_from_dict(base_recipe))
    rt.student_model = peft_student
    rt.student_module = peft_student

    # Test SelfTeacher with LoraDisabled
    teacher = SelfTeacher(SelfTeacher.Config(weight_policy="lora_disabled"))
    teacher.setup(rt)

    b = rt.collator(
        [
            {
                "student.input_ids": [3, 4, 10, 11],
                "student.attention_mask": [1, 1, 1, 1],
                "student.labels": [3, 4, 10, 11],
            }
        ]
    )
    out = teacher.signals(b, None, SideRequest(logits="view"), None)
    assert out is not None and out.logits is not None
    # Materializing out.logits outside the context should match base model logits
    base_m = AutoModelForCausalLM.from_pretrained(base_path).eval()
    peft_student.eval()
    base_logits = base_m(b["student.input_ids"], attention_mask=b["student.attention_mask"]).logits
    teacher_pos = out.logits.positional()
    assert torch.allclose(teacher_pos, base_logits, atol=1e-4)

    # Test TrustRegion with mix=0.5
    tr_teacher = SelfTeacher(SelfTeacher.Config(weight_policy={"type": "trust_region", "mix": 0.5}))
    tr_teacher.setup(rt)
    tr_out = tr_teacher.signals(b, None, SideRequest(logits="view"), None)
    assert tr_out is not None and tr_out.logits is not None
    live_logits = peft_student(b["student.input_ids"], attention_mask=b["student.attention_mask"]).logits
    expected_tr = 0.5 * base_logits + 0.5 * live_logits
    assert torch.allclose(tr_out.logits.positional(), expected_tr, atol=1e-4)


def test_ema_math():
    """EMA update math: lazy init with clone, then rate decay."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    m = DummyModel()
    rt = SimpleNamespace(student_model=m)
    policy = EMA(EMA.Config(rate=0.5))

    # Initial optimizer step: lazy initialization (no decay)
    policy.on_optimizer_step(rt)
    assert "p" in policy.ema
    assert torch.allclose(policy.ema["p"], torch.tensor([1.0]))

    # Modify student parameter to 3.0
    m.p.data = torch.tensor([3.0])
    # Next optimizer step: ema = (1 - 0.5) * 1.0 + 0.5 * 3.0 = 2.0
    policy.on_optimizer_step(rt)
    assert torch.allclose(policy.ema["p"], torch.tensor([2.0]))


def test_ema_context_restores_and_grad_correct():
    """Teacher forward inside ema context does not corrupt student backward gradient."""

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2, bias=False)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(42)
    net1 = SimpleNet()
    net2 = copy.deepcopy(net1)

    x = torch.randn(2, 4)
    loss1 = (net1(x) ** 2).sum()

    rt = SimpleNamespace(student_model=net1)
    policy = EMA(EMA.Config(rate=0.1))
    # Initialize EMA with different weights
    policy.ema = {name: p.data.clone() * 2.0 for name, p in net1.named_parameters()}

    orig_data = net1.linear.weight.data.clone()

    # In EMA context, forward teacher
    with policy.context(rt) as teacher_m:
        with torch.no_grad():
            _ = teacher_m(x)

    # Verify student parameter is restored
    assert torch.allclose(net1.linear.weight.data, orig_data)

    # Backward loss1
    loss1.backward()

    # Compare gradient with net2 (ran without EMA teacher forward)
    loss2 = (net2(x) ** 2).sum()
    loss2.backward()

    assert torch.allclose(net1.linear.weight.grad, net2.linear.weight.grad, atol=1e-6)


def test_periodic_sync():
    """PeriodicSync refreshes copy every N steps."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    m = DummyModel()
    rt = SimpleNamespace(student_model=m, state=SimpleNamespace(step=0))
    policy = PeriodicSync(PeriodicSync.Config(every=2))
    policy.setup(rt)

    # Initial copy has p=1.0
    assert torch.allclose(policy.copy.p, torch.tensor([1.0]))

    # Change student param
    m.p.data = torch.tensor([5.0])

    # Step 1: 1 % 2 != 0 -> copy keeps old weights (1.0)
    rt.state.step = 1
    policy.on_optimizer_step(rt)
    assert torch.allclose(policy.copy.p, torch.tensor([1.0]))

    # Step 2: 2 % 2 == 0 -> copy updated to student weights (5.0)
    rt.state.step = 2
    policy.on_optimizer_step(rt)
    assert torch.allclose(policy.copy.p, torch.tensor([5.0]))


def test_tinker_matches_original():
    """TinkerRKL loss and student gradients match original OPSD implementation."""
    torch.manual_seed(42)
    B, T, V = 2, 6, 16
    student_logits = torch.randn(B, T, V, requires_grad=True)
    teacher_token_logp = torch.randn(B, T)
    labels = torch.randint(0, V, (B, T))
    labels[:, :2] = -100

    mask = (labels != -100).float()
    s_view = LogitsView(logits=student_logits)
    student_target_logp = s_view.target_logp(labels)

    # Reference loss from OPSD reference implementation
    ref_loss = opsd_tinker_loss_ref(
        student_log_probs_sampled=student_target_logp,
        teacher_log_probs_sampled=teacher_token_logp,
        labels=labels,
    )

    batch = Batch({"student.labels": labels, "student.attention_mask": torch.ones(B, T, dtype=torch.long)})
    s_out = SimpleNamespace(logits=s_view)
    t_out = TeacherOutput(token_logp=teacher_token_logp)

    ctx = LossContext(
        batch=batch,
        student=s_out,
        student_views=[],
        teacher=t_out,
        modules=None,
        state=TrainState(),
        rt=None,
    )

    criterion = TinkerRKL(TinkerRKL.Config(temperature=1.0))
    loss_out = criterion(ctx)

    our_val = loss_out.terms["pg"].value / mask.sum()

    assert torch.allclose(our_val, ref_loss, atol=1e-6)

    # Check gradients w.r.t student_logits
    (ref_grad,) = torch.autograd.grad(ref_loss, student_logits, retain_graph=True)
    (our_grad,) = torch.autograd.grad(our_val, student_logits)

    assert torch.allclose(our_grad, ref_grad, atol=1e-6)


def test_reason_first_build(base_recipe, tiny_models):
    """ReasonFirst generates teacher reasoning via rollout and constructs privileged context."""
    from baselines.self_distill.opsd import ReasonFirst
    from lmkd.rollout.base import Rollout

    rt = build_runtime(recipe_from_dict(base_recipe))
    tok = rt.student_tokenizer

    # Mock rollout engine
    class MockRollout:
        def generate(self, prompts, spec, lora=None):
            return [[Rollout(prompt_ids=p, response_ids=[20, 21, tok.eos_token_id or 1])] for p in prompts]

    rt.rollout = MockRollout()

    s_resp = [10, 11, 1]
    s_ids = [2, 3] + s_resp
    batch = rt.collator(
        [
            {
                "student.input_ids": s_ids,
                "student.attention_mask": [1] * len(s_ids),
                "student.labels": make_labels(s_ids, 2).tolist(),
                "student.prompt_len": 2,
                "meta.problem": "math problem",
                "meta.solution": "math solution",
            }
        ]
    )

    builder = ReasonFirst(
        ReasonFirst.Config(
            reasoning_template="Problem: {problem}\nRef: {solution}\nExplain:",
            transition_template="\nNow derive:",
            fields=["solution"],
            use_chat_template=False,
        )
    )
    builder.setup(rt)
    res_batch = builder.build(batch, rt)

    sp = int(res_batch["student.prompt_len"][0].item())
    tp = int(res_batch["teacher.prompt_len"][0].item())
    s_mask = res_batch["student.attention_mask"][0]
    s_len = int(s_mask.sum().item())
    resp_len = s_len - sp

    # Teacher response tokens match student response tokens
    s_resp_t = res_batch["student.input_ids"][0, sp : sp + resp_len]
    t_resp_t = res_batch["teacher.input_ids"][0, tp : tp + resp_len]
    assert torch.equal(s_resp_t, t_resp_t)
    assert (res_batch["teacher.labels"][0, :tp] == -100).all()
