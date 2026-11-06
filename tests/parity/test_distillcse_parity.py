"""Numerical parity tests for DistillCSE against embedding-distill/distillcse (2).ipynb."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from baselines.embedding.distillcse import (
    DistillCSE,
    distillcse_kd_loss,
)
from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext, ModelPairInfo
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import ForwardRunner
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput

ROOT = Path(__file__).resolve().parents[3]


def _load_notebook_functions():
    nb_path = ROOT / "embedding-distill/distillcse (2).ipynb"
    if not nb_path.exists():
        pytest.skip(f"DistillCSE notebook not found: {nb_path}", allow_module_level=True)

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # cell 18: distillcse_kd_loss
    src_18 = "".join(nb["cells"][18]["source"])

    env = {"torch": torch, "F": F}
    exec(src_18, env)
    return env["distillcse_kd_loss"]


ref_distillcse_kd = _load_notebook_functions()


@pytest.mark.parity
def test_distillcse_kd_loss_parity():
    torch.manual_seed(42)
    B, D_s, D_t = 6, 32, 64
    s1 = torch.randn(B, D_s, dtype=torch.float32)
    s2 = torch.randn(B, D_s, dtype=torch.float32)
    t1 = torch.randn(B, D_t, dtype=torch.float32)
    t2 = torch.randn(B, D_t, dtype=torch.float32)
    tau = 0.05

    ref_loss = ref_distillcse_kd(s1, s2, t1, t2, tau=tau, detach_teacher=True)
    actual_loss = distillcse_kd_loss(s1, s2, t1, t2, tau=tau, detach_teacher=True)

    diff = (ref_loss - actual_loss).abs().item()
    assert diff <= 1e-6, f"Max diff {diff} exceeds atol 1e-6"


@pytest.mark.parity
def test_distillcse_criterion_parity():
    torch.manual_seed(42)
    B, T, D_s, D_t = 4, 12, 32, 48
    L = 2

    cfg = BertConfig(
        vocab_size=100,
        hidden_size=D_s,
        num_hidden_layers=L,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=0.0,
    )
    student = BertModel(cfg).eval()
    adapter = HFAutoAdapter()

    input_ids = torch.randint(0, 100, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    mask[:, 10:] = 0
    student_inputs = {"input_ids": input_ids, "attention_mask": mask}

    teacher_embeddings = torch.randn(B, D_t)

    criterion = DistillCSE(DistillCSE.Config(tau=0.05, layer=-1, pooler="cls"))
    info = ModelPairInfo(student_hidden=D_s, student_layers=L, student_vocab=100, teacher_hidden=D_t)
    modules = ModuleRegistry()
    criterion.build_modules(info, modules)

    req = criterion.features().student
    views = ForwardRunner.run(student, adapter, student_inputs, req, grad=False)
    assert len(views) == 2

    batch = Batch({"student.input_ids": input_ids, "student.attention_mask": mask})
    ctx = LossContext(
        batch=batch,
        student=views[0],
        student_views=views,
        teacher=TeacherOutput(embedding=teacher_embeddings),
        modules=modules,
        state=SimpleNamespace(step=0),
        rt=None,
    )

    actual_output = criterion.forward(ctx)
    actual_loss = actual_output.terms["distillcse"].value

    s1 = views[0].pooled(-1, "cls")
    s2 = views[1].pooled(-1, "cls")
    ref_loss = ref_distillcse_kd(s1, s2, teacher_embeddings, teacher_embeddings, tau=0.05, detach_teacher=True)

    diff = (ref_loss - actual_loss).abs().item()
    assert diff <= 1e-6, f"DistillCSE criterion parity diff {diff} exceeds atol 1e-6"
