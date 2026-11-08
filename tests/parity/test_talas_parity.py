import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from baselines.embedding.talas import LASD
from lmkd.core.batch import Batch
from lmkd.core.registry import POOLERS
from lmkd.losses.base import LossContext, ModelPairInfo
from lmkd.losses.embedding import CosineToTeacher, DimWeightedMSE, SimCSEUnsup
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import ForwardRunner
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput

ROOT = Path(__file__).resolve().parents[3]


def _load_talas_trainer():
    path = ROOT / "embedding-distill/TALAS-M-main/trainer.py"
    if not path.exists():
        pytest.skip(f"Original file not found: {path}", allow_module_level=True)
    spec = importlib.util.spec_from_file_location("ref_talas_trainer", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


talas_trainer = _load_talas_trainer()


@pytest.mark.parity
def test_cls_attention_parity():
    pooler = POOLERS.build("cls_attention")
    torch.manual_seed(42)

    B, T, D = 4, 16, 32
    hidden = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, 10:] = 0
    mask[1, 8:] = 0
    mask[2, 14:] = 0

    actual = pooler(hidden, mask)
    expected = talas_trainer.get_embedding(hidden, mask, cls_pooling=False)

    diff = (actual - expected).abs().max().item()
    assert diff <= 1e-6, f"Max diff {diff} exceeds atol 1e-6"


@pytest.mark.parity
def test_loss_components_parity():
    torch.manual_seed(42)
    B, T, D_s, D_t = 4, 12, 32, 48
    L = 3

    # Bert student with dropout 0 for deterministic parity
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

    # Reference trainer setup
    projectors = torch.nn.ModuleList([torch.nn.Linear(D_s, D_t, bias=False) for _ in range(L)])
    for proj in projectors:
        torch.nn.init.normal_(proj.weight, mean=0.0, std=1e-3)

    args = SimpleNamespace(
        cls_pooling=False,
        temperature=0.05,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
    )
    ref_trainer = talas_trainer.Trainer(student, None, projectors, args)
    teacher_embeddings = torch.randn(B, D_t)

    # 1. SimCSE parity (dropout 0 -> identical q and p)
    # Ref computation:
    q_out = student(**student_inputs, return_dict=True, output_hidden_states=True)
    p_out = student(**student_inputs, return_dict=True, output_hidden_states=True)
    q_emb = talas_trainer.get_embedding(q_out.last_hidden_state, mask, cls_pooling=False)
    p_emb = talas_trainer.get_embedding(p_out.last_hidden_state, mask, cls_pooling=False)
    q_norm = F.normalize(q_emb, dim=-1)
    p_norm = F.normalize(p_emb, dim=-1)
    scores = torch.matmul(q_norm, p_norm.t()) / args.temperature
    targets = torch.arange(B, dtype=torch.long)
    ref_simcse = F.cross_entropy(scores, targets, reduction="mean")

    # Framework SimCSEUnsup
    simcse = SimCSEUnsup(SimCSEUnsup.Config(temperature=0.05, layer=-1, pooler="cls_attention"))
    req = simcse.features().student
    views = ForwardRunner.run(student, adapter, student_inputs, req, grad=False)
    batch = Batch({"student.input_ids": input_ids, "student.attention_mask": mask})
    ctx = LossContext(
        batch=batch,
        student=views[0],
        student_views=views,
        teacher=None,
        modules=ModuleRegistry(),
        state=SimpleNamespace(step=0),
        rt=None,
    )
    lmkd_simcse = simcse.forward(ctx).terms["simcse_unsupervised"].value
    diff_simcse = (ref_simcse - lmkd_simcse).abs().item()
    assert diff_simcse <= 1e-5, f"SimCSE diff {diff_simcse} > 1e-5"

    # 2. TAMD parity
    # Ref TAMD
    ref_tamd, ref_lasd = ref_trainer.distill_loss(q_out, teacher_embeddings, mask)

    # Framework CosineToTeacher
    tamd = CosineToTeacher(CosineToTeacher.Config(layers=[-2, -1], pooler="cls_attention", teacher_dim=D_t))
    info = ModelPairInfo(student_hidden=D_s, student_layers=L, student_vocab=100, teacher_hidden=D_t)
    modules = ModuleRegistry()
    tamd.build_modules(info, modules)
    # Copy projector weights from ref_trainer (projectors 1 and 2, 0-indexed, corresponding to layers 2 and 3)
    modules.get("cosine_to_teacher__proj2").proj.weight.data.copy_(projectors[1].weight.data)
    modules.get("cosine_to_teacher__proj3").proj.weight.data.copy_(projectors[2].weight.data)

    req_tamd = tamd.features().student
    views_tamd = ForwardRunner.run(student, adapter, student_inputs, req_tamd, grad=False)
    ctx_tamd = LossContext(
        batch=batch,
        student=views_tamd[0],
        student_views=views_tamd,
        teacher=TeacherOutput(embedding=teacher_embeddings),
        modules=modules,
        state=SimpleNamespace(step=0),
        rt=None,
    )
    lmkd_tamd = tamd.forward(ctx_tamd).terms["cosine_to_teacher"].value
    diff_tamd = (ref_tamd - lmkd_tamd).abs().item()
    assert diff_tamd <= 1e-5, f"TAMD diff {diff_tamd} > 1e-5"

    # 3. LASD parity
    lasd = LASD(LASD.Config(pooler="cls_attention", start=1, detach_target=False))
    lasd.build_modules(info, modules)
    req_lasd = lasd.features().student
    views_lasd = ForwardRunner.run(student, adapter, student_inputs, req_lasd, grad=False)
    ctx_lasd = LossContext(
        batch=batch,
        student=views_lasd[0],
        student_views=views_lasd,
        teacher=None,
        modules=modules,
        state=SimpleNamespace(step=0),
        rt=None,
    )
    lmkd_lasd = lasd.forward(ctx_lasd).terms["lasd"].value
    diff_lasd = (ref_lasd - lmkd_lasd).abs().item()
    assert diff_lasd <= 1e-5, f"LASD diff {diff_lasd} > 1e-5"

    # 4. Dim-weighted MSE parity
    dim_w = DimWeightedMSE(DimWeightedMSE.Config(layer=-1, pooler="cls_attention", teacher_dim=D_t, power=1.0))
    dim_w.build_modules(info, modules)
    modules.get("dim_weighted_mse__proj").proj.weight.data.copy_(projectors[-1].weight.data)
    s_emb_proj = modules.get("dim_weighted_mse__proj")(q_emb)
    ref_dim_loss = ref_trainer.mse_dim_weight_loss(s_emb_proj, teacher_embeddings, power=1.0)
    ctx_dim = LossContext(
        batch=batch,
        student=views_lasd[0],
        student_views=views_lasd,
        teacher=TeacherOutput(embedding=teacher_embeddings),
        modules=modules,
        state=SimpleNamespace(step=0),
        rt=None,
    )
    lmkd_dim_loss = dim_w.forward(ctx_dim).terms["dim_weighted_mse"].value
    diff_dim = (ref_dim_loss - lmkd_dim_loss).abs().item()
    assert diff_dim <= 1e-5, f"DimWeightedMSE diff {diff_dim} > 1e-5"
