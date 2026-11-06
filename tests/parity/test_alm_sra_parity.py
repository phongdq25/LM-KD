"""Parity against the positional baseline arithmetic on small random tensors."""

from types import SimpleNamespace

import pytest
import torch
from tests.parity._ref.alm_ref import alm_loss
from tests.parity._ref.sra_ref import components, pool

from baselines.cross_tokenizer.alm import ALM
from baselines.cross_tokenizer.sra import SRA
from lmkd.core.batch import Batch
from lmkd.losses.base import LossContext
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherOutput


def _alm_case(mode, chunk_dim="shared_length"):
    torch.manual_seed(141)
    s_logits = torch.randn(2, 6, 9, requires_grad=True)
    t_logits = torch.randn(2, 5, 11)
    s_ids = torch.tensor([[1, 2, 3, 4, 5, 0], [1, 3, 4, 2, 0, 0]])
    t_ids = torch.tensor([[1, 2, 4, 5, 0], [1, 4, 2, 0, 0]])
    s_am = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
    t_am = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    s_chunks = torch.tensor([[-1, 0, 0, 1, 2, -1], [-1, 0, 1, 2, -1, -1]])
    t_chunks = torch.tensor([[-1, 0, 1, 1, -1], [-1, 0, 1, -1, -1]])
    s_space = torch.tensor([0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.bool)
    t_space = torch.tensor([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    batch = Batch(
        {
            "student.input_ids": s_ids,
            "student.attention_mask": s_am,
            "student.prompt_len": torch.ones(2, dtype=torch.long),
            "teacher.input_ids": t_ids,
            "teacher.attention_mask": t_am,
            "teacher.prompt_len": torch.ones(2, dtype=torch.long),
            "align.chunk_s": s_chunks,
            "align.chunk_t": t_chunks,
        }
    )
    vm = SimpleNamespace(space_mask_s=s_space, space_mask_t=t_space)
    rt = SimpleNamespace(data_ctx=SimpleNamespace(vocab_map=vm))
    ctx = LossContext(
        batch,
        SimpleNamespace(logits=LogitsView(logits=s_logits)),
        [],
        TeacherOutput(logits=LogitsView(logits=t_logits)),
        None,
        None,
        rt,
    )
    crit = ALM(ALM.Config(mode=mode, chunk_dim=chunk_dim, chunk_size=2))
    result = crit(ctx).terms["alm"].value
    c = min(s_ids.shape[1], t_ids.shape[1])
    a = torch.nn.functional.one_hot(s_chunks.clamp_min(0), c).float() * (s_chunks >= 0)[..., None]
    b = torch.nn.functional.one_hot(t_chunks.clamp_min(0), c).float() * (t_chunks >= 0)[..., None]
    s_lm = ((torch.arange(s_ids.shape[1])[None, :] >= 1) & s_am.bool()).float()
    t_lm = ((torch.arange(t_ids.shape[1])[None, :] >= 1) & t_am.bool()).float()
    expected = alm_loss(s_logits, t_logits, s_ids, t_ids, a, b, s_lm, t_lm, s_space, t_space, mode)
    return result, expected


@pytest.mark.parity
@pytest.mark.parametrize("mode", ["", "merge_by_space_prob", "append_space", "merge_by_space_prob+append_space"])
def test_alm_parity(mode):
    result, expected = _alm_case(mode)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)


@pytest.mark.parity
def test_alm_c_invariance():
    shared, _ = _alm_case("merge_by_space_prob+append_space", "shared_length")
    maximum, _ = _alm_case("merge_by_space_prob+append_space", "max_chunk")
    assert abs(float((shared - maximum).detach())) < 1e-4


@pytest.mark.parity
@pytest.mark.parametrize("shared_vocab", ["exact", "none"])
def test_sra_parity(shared_vocab):
    torch.manual_seed(142)
    s_h = torch.randn(2, 5, 3, requires_grad=True)
    t_h = torch.randn(2, 6, 4)
    s_att = torch.softmax(torch.randn(2, 2, 5, 5), -1)
    t_att = torch.softmax(torch.randn(2, 2, 6, 6), -1)
    s_idx = torch.tensor([[[1, 2], [3, 4]], [[1, -1], [2, 3]]])
    t_idx = torch.tensor([[[1, 2], [3, 4]], [[1, 2], [3, -1]]])
    s_head = torch.nn.Linear(3, 7, bias=False)
    t_head = torch.nn.Linear(4, 7, bias=False)
    s_vocab = {"a": 0, "b": 2, "c": 4, "s": 6}
    t_vocab = {"a": 3, "b": 1, "c": 5, "t": 6}
    rt = SimpleNamespace(
        student_model=s_head,
        teacher_model=t_head,
        student_adapter=SimpleNamespace(lm_head=lambda model: model),
        teacher_adapter=SimpleNamespace(lm_head=lambda model: model),
        student_tokenizer=SimpleNamespace(get_vocab=lambda: s_vocab),
        teacher_tokenizer=SimpleNamespace(get_vocab=lambda: t_vocab),
        data_ctx=SimpleNamespace(vocab_map=SimpleNamespace(same_vocab=False, stu2tea=None)),
    )
    side_s = SimpleNamespace(hidden=lambda layer: s_h, attention_last_row=lambda layer: s_att.sum(1)[:, -1].detach())
    side_t = SimpleNamespace(hidden=lambda layer: t_h, attention_last_row=lambda layer: t_att.sum(1)[:, -1].detach())
    modules = ModuleRegistry()
    crit = SRA(SRA.Config(shared_vocab=shared_vocab, geom_weight=5.0))
    crit.build_modules(SimpleNamespace(student_hidden=3, teacher_hidden=4), modules)
    crit.prepare(SimpleNamespace(rt=rt))
    batch = Batch({"align.sra_s_idx": [s_idx[0], s_idx[1]], "align.sra_t_idx": [t_idx[0], t_idx[1]]})
    ctx = LossContext(batch, side_s, [], TeacherOutput(side=side_t), modules, None, rt)
    result = crit(ctx)
    s_pool, _ = pool(s_h, s_att, s_idx)
    t_pool, t_w = pool(t_h, t_att, t_idx)
    ids = None
    if shared_vocab == "exact":
        shared = sorted(set(s_vocab) & set(t_vocab))
        ids = ([s_vocab[k] for k in shared], [t_vocab[k] for k in shared])
    ref = components(s_pool, t_pool, t_w, modules.get("sra__sra_proj"), s_head, t_head, 2.0, 5.0, 1.0, ids)
    for key, expected in zip(("span", "geom", "kd"), ref):
        torch.testing.assert_close(result.terms[key].value / 2, expected, atol=1e-5, rtol=0)
