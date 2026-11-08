"""Numerical parity tests for Jasper against embedding-distill/jasper (1).ipynb."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from baselines.embedding.jasper import (
    Jasper,
    cosine_embedding_loss,
    get_score_diff,
    pair_inbatch_similarity_loss,
    pair_inbatch_triplet_loss,
)
from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext, ModelPairInfo
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import ForwardRunner
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput

ROOT = Path(__file__).resolve().parents[3]


def _load_notebook_functions():
    nb_path = ROOT / "embedding-distill/jasper (1).ipynb"
    if not nb_path.exists():
        pytest.skip(f"Jasper notebook not found: {nb_path}", allow_module_level=True)

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # cell 15: cosine_embedding_loss, pair_inbatch_similarity_loss, pair_inbatch_triplet_loss, get_score_diff
    src_15 = "".join(nb["cells"][15]["source"])

    env = {"torch": torch, "F": F}
    exec(src_15, env)
    return (
        env["cosine_embedding_loss"],
        env["pair_inbatch_similarity_loss"],
        env["pair_inbatch_triplet_loss"],
        env["get_score_diff"],
    )


ref_cos, ref_sim, ref_tri, ref_score_diff = _load_notebook_functions()


@pytest.mark.parity
def test_cosine_embedding_loss_parity():
    torch.manual_seed(42)
    B, D = 6, 32
    s = torch.randn(B, D, dtype=torch.float32)
    t = torch.randn(B, D, dtype=torch.float32)

    ref_loss = ref_cos(s, t)
    actual_loss = cosine_embedding_loss(s, t)

    diff = (ref_loss - actual_loss).abs().item()
    assert diff <= 1e-6, f"Cosine loss diff {diff} exceeds atol 1e-6"


@pytest.mark.parity
def test_pair_inbatch_similarity_loss_parity():
    torch.manual_seed(42)
    B, D_s, D_t = 6, 32, 64
    s = torch.randn(B, D_s, dtype=torch.float32)
    t = torch.randn(B, D_t, dtype=torch.float32)

    ref_loss = ref_sim(s, t)
    actual_loss = pair_inbatch_similarity_loss(s, t)

    diff = (ref_loss - actual_loss).abs().item()
    assert diff <= 1e-6, f"Similarity loss diff {diff} exceeds atol 1e-6"


@pytest.mark.parity
def test_get_score_diff_and_triplet_loss_parity():
    torch.manual_seed(42)
    B, D_s, D_t = 5, 32, 48
    s = F.normalize(torch.randn(B, D_s, dtype=torch.float32), p=2, dim=-1)
    t = F.normalize(torch.randn(B, D_t, dtype=torch.float32), p=2, dim=-1)

    ref_diff_t = ref_score_diff(t)
    actual_diff_t = get_score_diff(t)
    diff = (ref_diff_t - actual_diff_t).abs().max().item()
    assert diff <= 1e-6, f"Score diff diff {diff} exceeds atol 1e-6"

    ref_tri_loss = ref_tri(s, t, triplet_margin=0.015)
    actual_tri_loss = pair_inbatch_triplet_loss(s, t, triplet_margin=0.015)
    diff_tri = (ref_tri_loss - actual_tri_loss).abs().item()
    assert diff_tri <= 1e-6, f"Triplet loss diff {diff_tri} exceeds atol 1e-6"


@pytest.mark.parity
def test_jasper_criterion_parity():
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
    student_inputs = {"input_ids": input_ids, "attention_mask": mask}
    teacher_embeddings = torch.randn(B, D_t)

    criterion = Jasper(
        Jasper.Config(
            layer=-1,
            pooler="cls",
            teacher_dim=D_t,
            projector=True,
            w_cos=0.5,
            w_sim=2.0,
            w_tri=0.5,
            triplet_margin=0.015,
        )
    )
    info = ModelPairInfo(student_hidden=D_s, student_layers=L, student_vocab=100, teacher_hidden=D_t)
    modules = ModuleRegistry()
    criterion.build_modules(info, modules)

    # Initialize projector deterministically
    torch.nn.init.normal_(modules.get("jasper__proj").proj.weight, mean=0.0, std=1e-3)

    req = criterion.features().student
    views = ForwardRunner.run(student, adapter, student_inputs, req, grad=False)

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
    actual_loss = actual_output.terms["jasper"].value

    s = views[0].pooled(-1, "cls")
    s_emb = modules.get("jasper__proj")(s)
    s_emb_norm = F.normalize(s_emb, p=2, dim=-1)
    t_emb_norm = F.normalize(teacher_embeddings, p=2, dim=-1)
    expected_cos = ref_cos(s_emb_norm, t_emb_norm)
    expected_sim = ref_sim(s_emb_norm, t_emb_norm)
    expected_tri = ref_tri(s_emb_norm, t_emb_norm, triplet_margin=0.015)
    expected_loss = 0.5 * expected_cos + 2.0 * expected_sim + 0.5 * expected_tri

    diff = (expected_loss - actual_loss).abs().item()
    assert diff <= 1e-6, f"Jasper criterion parity diff {diff} exceeds atol 1e-6"
