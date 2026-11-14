from types import SimpleNamespace

import pytest
import torch
from tests.parity.test_emo_parity import DummyTok
from transformers import BertConfig, BertModel

from baselines.embedding.distillcse import DistillCSE
from baselines.embedding.emo import EMO
from baselines.embedding.jasper import Jasper
from baselines.embedding.mol import MoL
from baselines.embedding.talas import LASD
from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext, ModelPairInfo
from lmkd.losses.embedding import CosineToTeacher, InfoNCE, SimCSEUnsup
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import ForwardRunner
from lmkd.models.modules import ModuleRegistry
from lmkd.plugins import load_builtin_plugins
from lmkd.teachers.signals import TeacherOutput

load_builtin_plugins()


def test_simcse_unsupervised_two_views_differ():
    torch.manual_seed(42)
    B, T, D = 4, 10, 32
    cfg = BertConfig(
        vocab_size=100,
        hidden_size=D,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    )
    student = BertModel(cfg).train()
    adapter = HFAutoAdapter()
    inputs = {
        "input_ids": torch.randint(0, 100, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
    }

    simcse = SimCSEUnsup(SimCSEUnsup.Config(temperature=0.05, layer=-1, pooler="cls_attention"))
    req = simcse.features().student
    views = ForwardRunner.run(student, adapter, inputs, req, grad=False)
    assert len(views) == 2

    emb0 = views[0].pooled(-1, "cls_attention")
    emb1 = views[1].pooled(-1, "cls_attention")
    assert not torch.allclose(emb0, emb1), "Expected different pooled representations due to dropout"


def test_infonce_teacher_contrastive():
    torch.manual_seed(42)
    B, T, D_s, D_t = 4, 10, 32, 48
    cfg = BertConfig(
        vocab_size=100,
        hidden_size=D_s,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    student = BertModel(cfg).eval()
    adapter = HFAutoAdapter()
    inputs = {
        "input_ids": torch.randint(0, 100, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
    }

    infonce = InfoNCE(InfoNCE.Config(temperature=0.07, layer=-1, pooler="cls_attention", teacher_dim=D_t))
    info = ModelPairInfo(student_hidden=D_s, student_layers=2, student_vocab=100, teacher_hidden=D_t)
    modules = ModuleRegistry()
    infonce.build_modules(info, modules)
    assert "infonce__proj" in modules

    req = infonce.features().student
    assert req.n_views == 1  # Standard InfoNCE only requires 1 view

    views = ForwardRunner.run(student, adapter, inputs, req, grad=False)
    assert len(views) == 1

    teacher_embeddings = torch.randn(B, D_t)

    ctx = LossContext(
        batch=Batch({"student.input_ids": inputs["input_ids"], "student.attention_mask": inputs["attention_mask"]}),
        student=views[0],
        student_views=views,
        teacher=TeacherOutput(embedding=teacher_embeddings),
        modules=modules,
        state=SimpleNamespace(step=0),
        rt=None,
    )
    loss = infonce.forward(ctx).terms["infonce"].value
    assert loss.ndim == 0 and loss.item() > 0


def test_tamd_projectors_registered():
    tamd = CosineToTeacher(CosineToTeacher.Config(layers=[-2, -1], pooler="cls_attention", teacher_dim=64))
    info = ModelPairInfo(student_hidden=32, student_layers=3, student_vocab=100, teacher_hidden=64)
    modules = ModuleRegistry()
    tamd.build_modules(info, modules)

    # In ModuleRegistry, '.' is mapped to '__', so cosine_to_teacher__proj2 and cosine_to_teacher__proj3
    assert "cosine_to_teacher__proj2" in modules
    assert "cosine_to_teacher__proj3" in modules
    assert modules.info["cosine_to_teacher__proj2"].group == "projector"
    assert modules.info["cosine_to_teacher__proj3"].group == "projector"

    # Missing teacher_dim
    tamd_no_dim = CosineToTeacher(CosineToTeacher.Config(layers=[-2, -1], pooler="cls_attention", teacher_dim=None))
    info_no_teacher = ModelPairInfo(student_hidden=32, student_layers=3, student_vocab=100, teacher_hidden=None)
    with pytest.raises(ValueError, match="set teacher_dim"):
        tamd_no_dim.build_modules(info_no_teacher, ModuleRegistry())


def test_lasd_no_detach():
    torch.manual_seed(42)
    B, T, D = 4, 8, 16
    cfg = BertConfig(
        vocab_size=50,
        hidden_size=D,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=32,
    )
    student = BertModel(cfg).train()
    adapter = HFAutoAdapter()
    inputs = {
        "input_ids": torch.randint(0, 50, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
    }

    lasd = LASD(LASD.Config(pooler="cls_attention", start=1, detach_target=False))
    info = ModelPairInfo(student_hidden=D, student_layers=3, student_vocab=50)
    modules = ModuleRegistry()
    lasd.build_modules(info, modules)

    req = lasd.features().student
    views = ForwardRunner.run(student, adapter, inputs, req, grad=True)
    batch = Batch({"student.input_ids": inputs["input_ids"], "student.attention_mask": inputs["attention_mask"]})
    ctx = LossContext(
        batch=batch,
        student=views[0],
        student_views=views,
        teacher=None,
        modules=modules,
        state=SimpleNamespace(step=0),
        rt=None,
    )

    out = lasd.forward(ctx)
    loss = out.terms["lasd"].value
    loss.backward()

    # Verify gradients flow back to student parameters in layer 0 and layer 1
    p0 = next(student.encoder.layer[0].parameters())
    p1 = next(student.encoder.layer[1].parameters())
    assert p0.grad is not None and p0.grad.abs().sum() > 0
    assert p1.grad is not None and p1.grad.abs().sum() > 0


def test_distillcse_unit():
    torch.manual_seed(42)
    B, T, D_s, D_t = 4, 8, 16, 24
    cfg = BertConfig(
        vocab_size=50,
        hidden_size=D_s,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    student = BertModel(cfg).train()
    adapter = HFAutoAdapter()
    inputs = {
        "input_ids": torch.randint(0, 50, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
    }

    criterion = DistillCSE(DistillCSE.Config(tau=0.05, layer=-1, pooler="cls", bidirectional=True))
    req = criterion.features()
    assert req.student.n_views == 2
    assert req.teacher_signal is not None and req.teacher_signal.kind == "embedding"

    views = ForwardRunner.run(student, adapter, inputs, req.student, grad=True)
    assert len(views) == 2

    teacher_embeddings = torch.randn(B, D_t)

    batch = Batch({"student.input_ids": inputs["input_ids"], "student.attention_mask": inputs["attention_mask"]})
    ctx = LossContext(
        batch=batch,
        student=views[0],
        student_views=views,
        teacher=TeacherOutput(embedding=teacher_embeddings),
        modules=ModuleRegistry(),
        state=SimpleNamespace(step=0),
        rt=None,
    )

    out = criterion.forward(ctx)
    loss = out.terms["distillcse"].value
    assert loss.ndim == 0 and loss.item() > 0
    loss.backward()

    # Gradient flows to student parameters
    p0 = next(student.encoder.layer[0].parameters())
    assert p0.grad is not None and p0.grad.abs().sum() > 0


def test_jasper_unit():
    torch.manual_seed(42)
    B, T, D_s, D_t = 4, 8, 16, 32
    cfg = BertConfig(
        vocab_size=50,
        hidden_size=D_s,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
    )
    student = BertModel(cfg).train()
    adapter = HFAutoAdapter()
    inputs = {
        "input_ids": torch.randint(0, 50, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
    }

    criterion = Jasper(
        Jasper.Config(
            layer=-1,
            pooler="cls",
            teacher_dim=D_t,
            projector=True,
            mrl_dims=(8,),
            w_cos=0.5,
            w_sim=2.0,
            w_tri=0.5,
        )
    )
    info = ModelPairInfo(student_hidden=D_s, student_layers=2, student_vocab=50, teacher_hidden=D_t)
    modules = ModuleRegistry()
    criterion.build_modules(info, modules)
    assert "jasper__proj" in modules
    assert "jasper__mrl_8" in modules

    req = criterion.features()
    assert req.student.n_views == 1
    assert req.teacher_signal is not None and req.teacher_signal.kind == "embedding"

    views = ForwardRunner.run(student, adapter, inputs, req.student, grad=True)
    teacher_embeddings = torch.randn(B, D_t)

    batch = Batch({"student.input_ids": inputs["input_ids"], "student.attention_mask": inputs["attention_mask"]})
    ctx = LossContext(
        batch=batch,
        student=views[0],
        student_views=views,
        teacher=TeacherOutput(embedding=teacher_embeddings),
        modules=modules,
        state=SimpleNamespace(step=0),
        rt=None,
    )

    out = criterion.forward(ctx)
    loss = out.terms["jasper"].value
    assert loss.ndim == 0 and loss.item() > 0
    assert "jasper/cos" in out.metrics
    assert "jasper/sim" in out.metrics
    assert "jasper/tri" in out.metrics

    loss.backward()
    p0 = next(student.encoder.layer[0].parameters())
    assert p0.grad is not None and p0.grad.abs().sum() > 0
    proj_w = modules.get("jasper__proj").proj.weight
    assert proj_w.grad is not None and proj_w.grad.abs().sum() > 0


def test_emo_unit():
    torch.manual_seed(42)
    B, T_s, T_t = 2, 6, 8
    D_s, D_t = 16, 24
    cfg_s = BertConfig(
        vocab_size=50,
        hidden_size=D_s,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
    )
    student = BertModel(cfg_s).train()
    adapter = HFAutoAdapter()
    inputs_s = {
        "input_ids": torch.randint(1, 50, (B, T_s)),
        "attention_mask": torch.ones(B, T_s, dtype=torch.long),
    }

    criterion = EMO(
        EMO.Config(
            teacher_dim=D_t,
            w_att=0.1,
            w_ot=1.0,
            teacher_special="[CLS]",
            student_special="[CLS]",
        )
    )
    criterion.student_tokenizer = DummyTok()
    criterion.teacher_tokenizer = DummyTok()

    info = ModelPairInfo(student_hidden=D_s, student_layers=2, student_vocab=50, teacher_hidden=D_t)
    modules = ModuleRegistry()
    criterion.build_modules(info, modules)
    assert "emo__projector" in modules

    req = criterion.features()
    assert req.student.attentions == "all"
    assert req.teacher.attentions == "all"

    s_views = ForwardRunner.run(student, adapter, inputs_s, req.student, grad=True)

    # Teacher mock side output
    t_last = torch.randn(B, T_t, D_t)
    t_atts = [torch.softmax(torch.randn(B, 2, T_t, T_t), dim=-1) for _ in range(4)]
    t_mask = torch.ones(B, T_t, dtype=torch.long)
    input_ids_t = torch.randint(1, 50, (B, T_t))

    teacher_side = SimpleNamespace(
        last_hidden=t_last,
        num_layers=4,
        attention=lambda layer: t_atts[layer - 1 if layer > 0 else layer],
        attention_mask=t_mask,
    )

    batch = Batch(
        {
            "student.input_ids": inputs_s["input_ids"],
            "student.attention_mask": inputs_s["attention_mask"],
            "teacher.input_ids": input_ids_t,
            "teacher.attention_mask": t_mask,
        }
    )

    ctx = LossContext(
        batch=batch,
        student=s_views[0],
        student_views=s_views,
        teacher=TeacherOutput(side=teacher_side),
        modules=modules,
        state=SimpleNamespace(step=0),
        rt=SimpleNamespace(student_tokenizer=DummyTok(), teacher_tokenizer=DummyTok()),
    )

    out = criterion.forward(ctx)
    loss = out.terms["emo"].value
    assert loss.ndim == 0 and loss.item() > 0
    assert "emo/ot_loss" in out.metrics
    assert "emo/att_loss" in out.metrics

    loss.backward()
    p0 = next(student.encoder.layer[0].parameters())
    assert p0.grad is not None and p0.grad.abs().sum() > 0
    proj_w = modules.get("emo__projector").weight
    assert proj_w.grad is not None and proj_w.grad.abs().sum() > 0

    # Test attn_implementation validation in prepare()
    sdpa_cfg = SimpleNamespace(
        models=SimpleNamespace(
            student=SimpleNamespace(attn_implementation="sdpa"),
            teacher=SimpleNamespace(attn_implementation="eager"),
        )
    )
    pctx = SimpleNamespace(
        rt=SimpleNamespace(config=sdpa_cfg, student_tokenizer=DummyTok(), teacher_tokenizer=DummyTok())
    )
    with pytest.raises(ValueError, match="is set to 'sdpa'"):
        criterion.prepare(pctx)


def test_mol_unit():
    torch.manual_seed(42)
    B, T_s, T_t = 2, 6, 8
    D_s, D_t = 16, 24
    cfg_s = BertConfig(
        vocab_size=50,
        hidden_size=D_s,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=32,
    )
    student = BertModel(cfg_s).train()
    adapter = HFAutoAdapter()
    inputs_s = {
        "input_ids": torch.randint(1, 50, (B, T_s)),
        "attention_mask": torch.ones(B, T_s, dtype=torch.long),
    }

    criterion = MoL(
        MoL.Config(
            teacher_dim=D_t,
            student_layers=(-2, -1),
            teacher_window=2,
        )
    )

    info = ModelPairInfo(student_hidden=D_s, student_layers=3, student_vocab=50, teacher_hidden=D_t)
    modules = ModuleRegistry()
    criterion.build_modules(info, modules)
    assert "mol__query_proj" in modules
    assert modules.info["mol__query_proj"].group == "projector"

    req = criterion.features()
    assert req.student.hidden == "all"
    assert req.teacher.hidden == "all"

    s_views = ForwardRunner.run(student, adapter, inputs_s, req.student, grad=True)

    # Teacher mock side output (3 layers -> 4 hidden states including embedding)
    t_hidden = [torch.randn(B, T_t, D_t) for _ in range(4)]
    t_mask = torch.ones(B, T_t, dtype=torch.long)
    input_ids_t = torch.randint(1, 50, (B, T_t))

    teacher_side = SimpleNamespace(
        hidden=lambda idx: t_hidden[idx if idx >= 0 else idx + 4],
        last_hidden=t_hidden[-1],
        num_layers=3,
        attention_mask=t_mask,
    )

    batch = Batch(
        {
            "student.input_ids": inputs_s["input_ids"],
            "student.attention_mask": inputs_s["attention_mask"],
            "teacher.input_ids": input_ids_t,
            "teacher.attention_mask": t_mask,
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

    out = criterion.forward(ctx)
    loss = out.terms["mol"].value
    assert loss.ndim == 0 and loss.item() > 0
    assert "mol/cka_loss" in out.metrics
    assert "mol/num_pairs" in out.metrics

    loss.backward()
    p0 = next(student.encoder.layer[0].parameters())
    assert p0.grad is not None and p0.grad.abs().sum() > 0
    proj_w = modules.get("mol__query_proj").weight
    assert proj_w.grad is not None and proj_w.grad.abs().sum() > 0

    # Missing teacher
    ctx_no_teacher = LossContext(
        batch=batch,
        student=s_views[0],
        student_views=s_views,
        teacher=None,
        modules=modules,
        state=SimpleNamespace(step=0),
        rt=None,
    )
    with pytest.raises(ValueError, match="requires teacher side outputs"):
        criterion.forward(ctx_no_teacher)
