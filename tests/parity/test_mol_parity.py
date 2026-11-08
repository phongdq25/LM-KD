"""Numerical parity tests for MoL against embedding-distill/MoL.ipynb."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from baselines.embedding.mol import (
    LinearCKA,
    MoL,
    align_teacher_to_student,
    mol_kd_loss,
)
from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext, ModelPairInfo
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import ForwardRunner
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput

ROOT = Path(__file__).resolve().parents[3]


def _load_notebook_env():
    nb_path = ROOT / "embedding-distill/MoL.ipynb"
    if not nb_path.exists():
        pytest.skip(f"MoL notebook not found: {nb_path}", allow_module_level=True)

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    src_14 = "".join(nb["cells"][14]["source"])

    env = {
        "math": math,
        "torch": torch,
        "nn": nn,
    }
    exec(src_14, env)
    return env


@pytest.fixture(scope="module")
def nb_env():
    return _load_notebook_env()


@pytest.mark.parity
def test_linear_cka_parity(nb_env):
    torch.manual_seed(42)
    B, Ls, Ds, Dt = 3, 8, 16, 24

    student_h = torch.randn(B, Ls, Ds)
    teacher_h = torch.randn(B, Ls, Dt)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]])

    ref_cka = nb_env["LinearCKA"](eps=1e-8)
    actual_cka = LinearCKA(eps=1e-8)

    # Test with mask
    ref_sim = ref_cka(student_h, teacher_h, student_mask=mask)
    actual_sim = actual_cka(student_h, teacher_h, student_mask=mask)
    diff = (ref_sim - actual_sim).abs().item()
    assert diff <= 1e-6, f"Linear CKA diff with mask: {diff}"

    # Test without mask
    ref_sim_nomask = ref_cka(student_h, teacher_h, student_mask=None)
    actual_sim_nomask = actual_cka(student_h, teacher_h, student_mask=None)
    diff_nomask = (ref_sim_nomask - actual_sim_nomask).abs().item()
    assert diff_nomask <= 1e-6, f"Linear CKA diff without mask: {diff_nomask}"


@pytest.mark.parity
def test_align_teacher_to_student_parity(nb_env):
    torch.manual_seed(42)
    B, Ls, Lt = 2, 6, 8
    Ds, Dt = 16, 24

    ref_mol = nb_env["MinCKATokenKDLoss"](
        student_hidden_size=Ds,
        teacher_hidden_size=Dt,
        student_layers=[1, 2],
        teacher_window=1,
    )

    s_emb = torch.randn(B, Ls, Ds)
    s_last = torch.randn(B, Ls, Ds)
    t_emb = torch.randn(B, Lt, Dt)
    t_last = torch.randn(B, Lt, Dt)
    t_layer = torch.randn(B, Lt, Dt)
    t_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]])

    ref_aligned = ref_mol._align_teacher_to_student(
        student_embedding=s_emb,
        student_last_hidden=s_last,
        teacher_embedding=t_emb,
        teacher_last_hidden=t_last,
        teacher_layer_hidden=t_layer,
        teacher_attention_mask=t_mask,
    )

    actual_aligned = align_teacher_to_student(
        query_projector=ref_mol.query_projector,
        student_embedding=s_emb,
        student_last_hidden=s_last,
        teacher_embedding=t_emb,
        teacher_last_hidden=t_last,
        teacher_layer_hidden=t_layer,
        teacher_attention_mask=t_mask,
        teacher_hidden_size=Dt,
    )

    diff = (ref_aligned - actual_aligned).abs().max().item()
    assert diff <= 1e-6, f"Align teacher to student diff: {diff}"


@pytest.mark.parity
def test_mol_kd_loss_parity(nb_env):
    torch.manual_seed(42)
    B, Ls, Lt = 2, 6, 8
    Ds, Dt = 16, 24
    num_s_layers = 3
    num_t_layers = 4

    ref_mol = nb_env["MinCKATokenKDLoss"](
        student_hidden_size=Ds,
        teacher_hidden_size=Dt,
        student_layers=[2, 3],
        teacher_window=2,
    )

    s_hidden_states = [torch.randn(B, Ls, Ds) for _ in range(num_s_layers + 1)]
    t_hidden_states = [torch.randn(B, Lt, Dt) for _ in range(num_t_layers + 1)]
    s_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
    t_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]])

    ref_loss, ref_logs = ref_mol(
        student_hidden_states=s_hidden_states,
        teacher_hidden_states=t_hidden_states,
        student_attention_mask=s_mask,
        teacher_attention_mask=t_mask,
    )

    actual_loss, actual_logs = mol_kd_loss(
        query_projector=ref_mol.query_projector,
        student_hidden_states=s_hidden_states,
        teacher_hidden_states=t_hidden_states,
        student_attention_mask=s_mask,
        teacher_attention_mask=t_mask,
        student_layers=[2, 3],
        teacher_window=2,
        teacher_hidden_size=Dt,
    )

    diff = (ref_loss - actual_loss).abs().item()
    assert diff <= 1e-6, f"MoL KD loss diff: {diff}"
    assert (ref_logs["cka_kd_loss"] - actual_logs["cka_kd_loss"]).abs().item() <= 1e-6
    assert ref_logs["cka_num_pairs"] == actual_logs["cka_num_pairs"].item()


@pytest.mark.parity
def test_mol_criterion_parity(nb_env):
    torch.manual_seed(42)
    B, Ls, Lt = 2, 6, 8
    Ds, Dt = 16, 24
    num_s_layers = 2
    num_t_layers = 3

    cfg_s = BertConfig(
        vocab_size=100,
        hidden_size=Ds,
        num_hidden_layers=num_s_layers,
        num_attention_heads=2,
        intermediate_size=32,
    )
    student = BertModel(cfg_s).eval()
    adapter = HFAutoAdapter()

    input_ids_s = torch.randint(1, 100, (B, Ls))
    mask_s = torch.ones(B, Ls, dtype=torch.long)
    inputs_s = {"input_ids": input_ids_s, "attention_mask": mask_s}

    criterion = MoL(
        MoL.Config(
            student_layers=(1, 2),
            teacher_window=2,
            teacher_dim=Dt,
        )
    )

    info = ModelPairInfo(
        student_hidden=Ds,
        student_layers=num_s_layers,
        student_vocab=100,
        teacher_hidden=Dt,
        teacher_layers=num_t_layers,
    )
    modules = ModuleRegistry()
    criterion.build_modules(info, modules)
    proj = modules.get("mol__query_proj")

    req = criterion.features().student
    s_views = ForwardRunner.run(student, adapter, inputs_s, req, grad=False)

    t_hidden_states = [torch.randn(B, Lt, Dt) for _ in range(num_t_layers + 1)]
    mask_t = torch.ones(B, Lt, dtype=torch.long)
    input_ids_t = torch.randint(1, 100, (B, Lt))

    teacher_side = SimpleNamespace(
        hidden=lambda idx: t_hidden_states[idx if idx >= 0 else idx + num_t_layers + 1],
        last_hidden=t_hidden_states[-1],
        num_layers=num_t_layers,
        attention_mask=mask_t,
    )

    batch = Batch(
        {
            "student.input_ids": input_ids_s,
            "student.attention_mask": mask_s,
            "teacher.input_ids": input_ids_t,
            "teacher.attention_mask": mask_t,
        }
    )

    ctx = LossContext(
        batch=batch,
        student=s_views[0],
        student_views=s_views,
        teacher=TeacherOutput(side=teacher_side),
        modules=modules,
        state=SimpleNamespace(step=0),
        rt=None,
    )

    actual_out = criterion.forward(ctx)
    actual_loss = actual_out.terms["mol"].value

    # Compute reference loss using notebook's MinCKATokenKDLoss
    ref_mol = nb_env["MinCKATokenKDLoss"](
        student_hidden_size=Ds,
        teacher_hidden_size=Dt,
        student_layers=[1, 2],
        teacher_window=2,
    )
    # Copy projector weights
    ref_mol.query_projector.load_state_dict(proj.state_dict())

    s_hidden_states = [s_views[0].hidden(i) for i in range(num_s_layers + 1)]
    ref_loss, _ = ref_mol(
        student_hidden_states=s_hidden_states,
        teacher_hidden_states=t_hidden_states,
        student_attention_mask=mask_s,
        teacher_attention_mask=mask_t,
    )

    diff = (ref_loss - actual_loss).abs().item()
    assert diff <= 1e-6, f"MoL criterion parity diff {diff} exceeds atol 1e-6"
