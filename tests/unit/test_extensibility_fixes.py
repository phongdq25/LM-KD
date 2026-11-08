"""Behavioral regressions for the new-method APIs (T32)."""

import copy
from types import SimpleNamespace

import pytest
import torch

import lmkd
from lmkd import Inputs, ScalarLoss, SeqLoss, TokenLoss
from lmkd.core.batch import Batch
from lmkd.core.registry import CRITERIONS, RegistryError
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import SideOutput
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherOutput


def _context(efficient=True):
    torch.manual_seed(17)
    head = torch.nn.Linear(4, 9, bias=True)
    teacher_head = copy.deepcopy(head).requires_grad_(False)
    hidden = torch.randn(2, 7, 4, requires_grad=True)
    teacher_hidden = torch.randn(2, 7, 4)

    def side(h, projection):
        out = SideOutput(torch.nn.Identity(), HFAutoAdapter(), None, None)
        out._out = SimpleNamespace(last_hidden_state=h, hidden_states=(h,))
        out.logits = LogitsView(h, projection) if efficient else LogitsView(logits=projection(h))
        return out

    s, t = side(hidden, head), side(teacher_hidden, teacher_head)
    labels = torch.tensor([[-100, 1, 2, -100, 3, 4, 5], [-100, 2, -100, 4, 5, 6, -100]])
    batch = Batch({"student.input_ids": labels.clamp_min(0), "student.labels": labels})
    ctx = LossContext(batch, s, [s], TeacherOutput(side=t, logits=t.logits), None, TrainState(), None)
    return ctx, hidden, head


@pytest.mark.parametrize("teacher", [False, True])
@pytest.mark.parametrize("efficient", [False, True])
def test_custom_mask_chunks_match_full_loss_logs_and_gradients(teacher, efficient):
    def objective(s, t=None):
        values = s.logits.square().mean(-1) if t is None else (s.logits - t.logits).square().mean(-1)
        custom = torch.ones_like(values)
        custom[:, ::2] = 0
        return TokenLoss(values, custom, values + 3)

    def student(s):
        return objective(s)

    def kd(s, t):
        return objective(s, t)

    fn = kd if teacher else student
    full = lmkd.loss(f"t32_full_{teacher}_{efficient}", teacher=teacher, chunked=False)(fn).criterion_class
    chunked = lmkd.loss(f"t32_chunk_{teacher}_{efficient}", teacher=teacher, chunk_size=2)(fn).criterion_class
    results = []
    for f in (full, chunked):
        ctx, hidden, head = _context(efficient)
        term = f()(ctx).terms["loss"]
        grads = torch.autograd.grad(term.value, (hidden, head.weight, head.bias))
        results.append((term.value.detach(), term.log_value, grads))
    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])
    for a, b in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(a, b)


def test_chunk_inputs_expose_offsets_mask_labels_and_lazy_logp():
    @lmkd.loss("t32_chunk_inputs", teacher=True, chunk_size=2)
    def objective(s: Inputs, t: Inputs) -> TokenLoss:
        assert s.end - s.start == s.logits.shape[1]
        assert s.labels.shape == s.mask.shape == t.mask.shape
        torch.testing.assert_close(s.labels, s.batch["student.labels"][:, s.start : s.end])
        return TokenLoss(-s.token_logp(s.labels), s.mask)

    ctx, hidden, _ = _context()
    term = objective.criterion_class()(ctx).terms["loss"]
    expected = -ctx.student.logits.target_logp(ctx.batch["student.labels"]).sum()
    torch.testing.assert_close(term.value, expected)
    term.value.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("kind", ["sequence", "scalar", "dict"])
def test_default_loss_selects_full_for_global_objectives(kind):
    @lmkd.loss(f"t32_global_{kind}")
    def objective(s):
        values = s.logits.square().mean((1, 2))
        if kind == "sequence":
            return SeqLoss(values)
        if kind == "scalar":
            return ScalarLoss(values.mean())
        return {"seq": SeqLoss(values), "regularizer": ScalarLoss(values.mean())}

    ctx, hidden, _ = _context()
    out = objective.criterion_class()(ctx)
    sum(term.value for term in out.terms.values()).backward()
    assert hidden.grad is not None


def test_hidden_only_objective_never_materializes_logits(monkeypatch):
    @lmkd.loss("t32_hidden_only", capture={"s": [0]})
    def objective(s):
        return TokenLoss(s.hidden[0].square().mean(-1))

    ctx, hidden, _ = _context()

    def forbid_full(*args):
        raise AssertionError("hidden-only loss materialized logits")

    monkeypatch.setattr(LogitsView, "full", forbid_full)
    objective.criterion_class()(ctx).terms["loss"].value.backward()
    assert hidden.grad is not None


def test_default_scalar_regularizer_runs_without_supervised_tokens():
    @lmkd.loss("t32_unsupervised_scalar", chunk_size=2)
    def objective(s):
        return ScalarLoss(s.logits.square().mean())

    ctx, hidden, _ = _context()
    ctx.batch["student.labels"].fill_(-100)
    term = objective.criterion_class()(ctx).terms["loss"]
    assert term.reduction == "mean" and term.value > 0
    term.value.backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_default_hidden_and_logit_objective_preserves_both_gradient_paths():
    def objective(s):
        return TokenLoss(s.logits.square().mean(-1) + s.last_hidden.square().mean(-1))

    full = lmkd.loss("t32_hidden_logit_full", chunked=False)(objective).criterion_class
    auto = lmkd.loss("t32_hidden_logit_auto", chunk_size=2)(objective).criterion_class
    gradients = []
    for fn in (full, auto):
        ctx, hidden, head = _context()
        gradients.append(torch.autograd.grad(fn()(ctx).terms["loss"].value, (hidden, head.weight)))
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b)


def test_duplicate_loss_names_preserve_unrelated_registration():
    @lmkd.loss("t32_duplicate")
    def first(s):
        return TokenLoss(s.logits.mean(-1))

    with pytest.raises(RegistryError, match="already registered"):

        @lmkd.loss("t32_duplicate")
        def unrelated(s):
            return TokenLoss(s.logits.mean(-1))

    assert CRITERIONS.get("t32_duplicate") is first.criterion_class


def test_quick_uses_registered_callable_name_and_rejects_unknown_inputs():
    from lmkd.api import _parse_quick_loss_terms

    @lmkd.loss("t32_callable_alias")
    def a_different_python_name(s):
        return TokenLoss(s.logits.mean(-1))

    assert _parse_quick_loss_terms(a_different_python_name) == [{"criterion": "t32_callable_alias"}]
    with pytest.raises(TypeError, match="loss"):
        _parse_quick_loss_terms([object()])


@pytest.mark.parametrize("bad", ["detached", "nan"])
def test_cli_try_rejects_detached_and_nonfinite_objectives(bad, capsys):
    from lmkd.cli import main

    @lmkd.loss(f"t32_try_{bad}", chunked=False)
    def objective(s):
        values = torch.ones_like(s.mask) if bad == "detached" else s.logits.mean(-1) * float("nan")
        return TokenLoss(values)

    assert main(["try", f"t32_try_{bad}"]) == 1
    assert "Success!" not in capsys.readouterr().out


def test_criterion_scaffold_runs_a_real_gradient_test(tmp_path):
    from lmkd.core.config import load_recipe
    from lmkd.core.validate import check_recipe
    from lmkd.scaffold import create_scaffold

    assert create_scaffold("criterion", "t32_scaffold", root=tmp_path) == 0
    cfg = load_recipe(tmp_path / "recipes/experimental/t32_scaffold.yaml")
    assert check_recipe(cfg).ok
    from lmkd.plugins import load_plugins

    load_plugins([str(tmp_path / "tests/unit/test_t32_scaffold.py")])
    import sys

    generated = next(
        m
        for m in sys.modules.values()
        if getattr(m, "__file__", None) == str(tmp_path / "tests/unit/test_t32_scaffold.py")
    )
    generated.test_t32_scaffold_gradient()


def test_criterion_aux_module_ergonomics_and_checkpoint_loading(tmp_path):
    from lmkd.core.plugin import BaseConfig
    from lmkd.losses.base import Criterion, ModelPairInfo, PrepareContext
    from lmkd.models.modules import ModuleRegistry

    class DummyCrit(Criterion):
        class Config(BaseConfig):
            projector_ckpt: str = ""

        cfg: Config

        def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
            self.register_aux_module(modules, "projector", torch.nn.Linear(info.student_hidden, 4, bias=False))
            self.register_aux_module(modules, "head", torch.nn.Linear(4, 2, bias=False))

    # Stage 1: build and save modules.pt in checkpoint-final/
    reg1 = ModuleRegistry()
    c1 = DummyCrit()
    c1.alias = "stage1_crit"
    c1.build_modules(ModelPairInfo(student_hidden=8, student_layers=2, student_vocab=16), reg1)

    # Direct attribute access on Criterion, ScopedModuleView, and ModuleRegistry
    assert c1.projector is reg1.projector
    assert c1.aux(reg1).projector is reg1.stage1_crit__projector
    assert reg1.for_owner("stage1_crit").head is c1.head

    with torch.no_grad():
        c1.projector.weight.fill_(3.25)
        c1.head.weight.fill_(-1.5)

    ckpt_dir = tmp_path / "run1" / "checkpoint-final"
    ckpt_dir.mkdir(parents=True)
    torch.save(reg1.saveable_state_dict(), ckpt_dir / "modules.pt")

    # Stage 2: load single module from stage 1 directory without manual prefix slicing
    reg2 = ModuleRegistry()
    c2 = DummyCrit(DummyCrit.Config(projector_ckpt=str(tmp_path / "run1")))
    c2.alias = "stage2_crit"
    c2.build_modules(ModelPairInfo(student_hidden=8, student_layers=2, student_vocab=16), reg2)
    pctx = PrepareContext(rt=SimpleNamespace(modules=reg2))
    c2.load_aux_checkpoint(pctx, "projector")
    torch.testing.assert_close(c2.projector.weight, torch.full_like(c2.projector.weight, 3.25))

    # Load all modules of an owner at once
    c2.load_aux_checkpoint(pctx, source_owner="stage1_crit")
    torch.testing.assert_close(c2.head.weight, torch.full_like(c2.head.weight, -1.5))

