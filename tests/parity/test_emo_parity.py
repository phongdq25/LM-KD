"""Numerical parity tests for EMO against embedding-distill/emo-unsup-simcse.ipynb."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import editdistance
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.embedding.emo import (
    EMO,
    align_tokens,
    build_reciprocal_mapping_from_token_lists,
    compute_att_loss_2,
    compute_ot_loss,
    compute_token_importance,
    pairwise_attention_distance,
    project_importance,
    sinkhorn,
)
from lmkd.core.batch import Batch
from lmkd.losses.base import ModelPairInfo
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput

ROOT = Path(__file__).resolve().parents[3]


def _load_notebook_env():
    nb_path = ROOT / "embedding-distill/emo-unsup-simcse.ipynb"
    if not nb_path.exists():
        pytest.skip(f"EMO notebook not found: {nb_path}", allow_module_level=True)

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    src_13 = "".join(nb["cells"][13]["source"])
    src_14 = "".join(nb["cells"][14]["source"])
    src_14_fixed = src_14.replace(
        "proj_teacher_seq_f = proje",
        "proj_teacher_seq_f = projected_teacher_seq.to(device=device, dtype=torch.float32)",
    )

    env = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "editdistance": editdistance,
    }
    exec(src_13, env)
    exec(src_14_fixed, env)
    return env


@pytest.fixture(scope="module")
def nb_env():
    return _load_notebook_env()


class DummyTok:
    def convert_ids_to_tokens(self, ids):
        mapping = {0: "[CLS]", 1: "hello", 2: "world", 3: "distillation", 4: "optimal", 5: "transport", 6: "[SEP]", 7: "[PAD]"}
        return [mapping.get(i, f"tok_{i}") for i in ids]


@pytest.mark.parity
def test_align_tokens_parity(nb_env):
    t_tokens = ["<s>", "hello", "world", "distill", "ation", "</s>"]
    s_tokens = ["[CLS]", "hello", "world", "distillation", "[SEP]"]
    ref_res = nb_env["align_tokens"](t_tokens, s_tokens, "<s>", "[CLS]")
    my_res = align_tokens(t_tokens, s_tokens, "<s>", "[CLS]")
    assert ref_res == my_res


@pytest.mark.parity
def test_reciprocal_mapping_parity(nb_env):
    t_tokens = ["<s>", "hello", "world", "distill", "ation", "</s>"]
    s_tokens = ["[CLS]", "hello", "world", "distill", "[SEP]"]
    ref_res = nb_env["build_reciprocal_mapping_from_token_lists"](t_tokens, s_tokens, "<s>", "[CLS]")
    my_res = build_reciprocal_mapping_from_token_lists(t_tokens, s_tokens, "<s>", "[CLS]")
    assert ref_res == my_res


@pytest.mark.parity
def test_compute_token_importance_parity(nb_env):
    torch.manual_seed(42)
    att = torch.softmax(torch.randn(4, 8, 8), dim=-1)
    tokens = [f"t{i}" for i in range(8)]
    ref_imp = nb_env["compute_token_importance"](att, tokens)
    my_imp = compute_token_importance(att, tokens)
    diff = (ref_imp - my_imp).abs().max().item()
    assert diff <= 1e-7, f"Token importance diff: {diff}"


@pytest.mark.parity
def test_project_importance_parity(nb_env):
    torch.manual_seed(42)
    t_imp = torch.softmax(torch.randn(6), dim=0)
    t_tokens = ["[CLS]", "hello", "world", "foo", "bar", "[SEP]"]
    s_tokens = ["[CLS]", "hello", "world", "baz", "[SEP]"]
    mapping = {"[CLS]": "[CLS]", "hello": "hello", "world": "world", "[SEP]": "[SEP]"}
    ref_p = nb_env["project_importance"](t_imp, t_tokens, s_tokens, mapping)
    my_p = project_importance(t_imp, t_tokens, s_tokens, mapping)
    diff = (ref_p - my_p).abs().max().item()
    assert diff <= 1e-7, f"Project importance diff: {diff}"


@pytest.mark.parity
def test_pairwise_attention_distance_parity(nb_env):
    torch.manual_seed(42)
    x = torch.randn(5, 16)
    y = torch.randn(7, 16)
    ref_dist = nb_env["pairwise_attention_distance"](x, y)
    my_dist = pairwise_attention_distance(x, y)
    diff = (ref_dist - my_dist).abs().max().item()
    assert diff <= 1e-7, f"Pairwise attention distance diff: {diff}"


@pytest.mark.parity
def test_sinkhorn_parity(nb_env):
    torch.manual_seed(42)
    cost = torch.rand(6, 8)
    a = torch.softmax(torch.randn(6), dim=0)
    b = torch.softmax(torch.randn(8), dim=0)
    ref_loss, ref_P = nb_env["sinkhorn"](cost, a, b, alpha=0.1, max_iter=50)
    my_loss, my_P = sinkhorn(cost, a, b, alpha=0.1, max_iter=50)
    diff_loss = (ref_loss - my_loss).abs().max().item()
    diff_P = (ref_P - my_P).abs().max().item()
    assert diff_loss <= 1e-6, f"Sinkhorn loss diff: {diff_loss}"
    assert diff_P <= 1e-6, f"Sinkhorn P diff: {diff_P}"


@pytest.mark.parity
def test_compute_ot_loss_parity(nb_env):
    torch.manual_seed(42)
    B, L_s, L_t = 2, 6, 8
    d_s, d_t = 16, 24
    H = 2

    t_last = torch.randn(B, L_t, d_t)
    s_last = torch.randn(B, L_s, d_s)
    t_att_last = torch.softmax(torch.randn(B, H, L_t, L_t), dim=-1)
    t_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]])
    s_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    input_ids_t = torch.tensor([[0, 1, 2, 3, 5, 7, 7, 7], [0, 1, 4, 6, 7, 7, 7, 7]])
    input_ids_s = torch.tensor([[0, 1, 2, 6, 7, 7], [0, 1, 6, 7, 7, 7]])
    tok = DummyTok()

    proj = nn.Linear(d_t, d_s, bias=False)

    ref_ot = nb_env["compute_ot_loss"](
        teacher_last=t_last,
        student_last=s_last,
        teacher_att_last=t_att_last,
        attention_mask_teacher=t_mask,
        attention_mask_student=s_mask,
        input_ids_tea=input_ids_t,
        input_ids_stu=input_ids_s,
        tok_teacher=tok,
        tok_student=tok,
        projector=proj,
        teacher_special="[CLS]",
        student_special="[CLS]",
    )

    my_ot = compute_ot_loss(
        teacher_last=t_last,
        student_last=s_last,
        teacher_att_last=t_att_last,
        attention_mask_teacher=t_mask,
        attention_mask_student=s_mask,
        input_ids_tea=input_ids_t,
        input_ids_stu=input_ids_s,
        tok_teacher=tok,
        tok_student=tok,
        projector=proj,
        teacher_special="[CLS]",
        student_special="[CLS]",
    )

    diff = (ref_ot - my_ot).abs().max().item()
    assert diff <= 1e-6, f"Compute OT loss diff: {diff}"


@pytest.mark.parity
def test_compute_att_loss_2_parity(nb_env):
    torch.manual_seed(42)
    B, L_s, L_t = 2, 6, 8
    H = 2

    t_atts = [torch.softmax(torch.randn(B, H, L_t, L_t), dim=-1) for _ in range(4)]
    s_atts = [torch.softmax(torch.randn(B, H, L_s, L_s), dim=-1) for _ in range(2)]
    t_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]])
    s_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    input_ids_t = torch.tensor([[0, 1, 2, 3, 5, 7, 7, 7], [0, 1, 4, 6, 7, 7, 7, 7]])
    input_ids_s = torch.tensor([[0, 1, 2, 6, 7, 7], [0, 1, 6, 7, 7, 7]])
    tok = DummyTok()

    t_last_dev = torch.device("cpu")
    ref_att = nb_env["compute_att_loss_2"](
        teacher_atts=t_atts,
        student_atts=s_atts,
        input_ids_tea=input_ids_t,
        input_ids_stu=input_ids_s,
        attention_mask_tea=t_mask,
        attention_mask_stu=s_mask,
        tok_teacher=tok,
        tok_student=tok,
        k=1,
        device=t_last_dev,
        teacher_special="[CLS]",
        student_special="[CLS]",
    )

    my_att = compute_att_loss_2(
        teacher_atts=t_atts,
        student_atts=s_atts,
        input_ids_tea=input_ids_t,
        input_ids_stu=input_ids_s,
        attention_mask_tea=t_mask,
        attention_mask_stu=s_mask,
        tok_teacher=tok,
        tok_student=tok,
        k=1,
        device=t_last_dev,
        teacher_special="[CLS]",
        student_special="[CLS]",
    )

    diff = (ref_att - my_att).abs().max().item()
    assert diff <= 1e-6, f"Compute att loss diff: {diff}"


@pytest.mark.parity
def test_emo_criterion():
    torch.manual_seed(42)
    B, L_s, L_t = 2, 6, 8
    d_s, d_t = 16, 24
    H = 2

    criterion = EMO(EMO.Config(teacher_dim=d_t, w_att=0.1, w_ot=1.0, teacher_special="[CLS]", student_special="[CLS]"))
    criterion.teacher_tokenizer = DummyTok()
    criterion.student_tokenizer = DummyTok()

    info = ModelPairInfo(student_hidden=d_s, student_layers=2, student_vocab=100, teacher_hidden=d_t)
    modules = ModuleRegistry()
    criterion.build_modules(info, modules)
    modules.get("emo__projector")

    t_last = torch.randn(B, L_t, d_t)
    s_last = torch.randn(B, L_s, d_s)
    t_atts = [torch.softmax(torch.randn(B, H, L_t, L_t), dim=-1) for _ in range(4)]
    s_atts = [torch.softmax(torch.randn(B, H, L_s, L_s), dim=-1) for _ in range(2)]

    t_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]])
    s_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    input_ids_t = torch.tensor([[0, 1, 2, 3, 5, 7, 7, 7], [0, 1, 4, 6, 7, 7, 7, 7]])
    input_ids_s = torch.tensor([[0, 1, 2, 6, 7, 7], [0, 1, 6, 7, 7, 7]])

    student_side = SimpleNamespace(
        last_hidden=s_last,
        num_layers=2,
        attention=lambda layer: s_atts[layer - 1 if layer > 0 else layer],
        attention_mask=s_mask,
    )
    teacher_side = SimpleNamespace(
        last_hidden=t_last,
        num_layers=4,
        attention=lambda layer: t_atts[layer - 1 if layer > 0 else layer],
        attention_mask=t_mask,
    )

    batch = Batch({
        "student.input_ids": input_ids_s,
        "student.attention_mask": s_mask,
        "teacher.input_ids": input_ids_t,
        "teacher.attention_mask": t_mask,
    })

    ctx = SimpleNamespace(
        batch=batch,
        student=student_side,
        teacher=TeacherOutput(side=teacher_side),
        modules=modules,
        rt=SimpleNamespace(student_tokenizer=DummyTok(), teacher_tokenizer=DummyTok()),
    )

    out = criterion.forward(ctx)
    assert "emo" in out.terms
    assert out.terms["emo"].value.item() > 0.0
    assert "emo/ot_loss" in out.metrics
    assert "emo/att_loss" in out.metrics
