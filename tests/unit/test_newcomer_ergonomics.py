"""Unit tests for newcomer integration ergonomics, simplified APIs, and CLI scaffolding."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import lmkd
from lmkd.cli import main
from lmkd.core.config import load_recipe
from lmkd.core.validate import check_recipe
from lmkd.plugins import _USER_LOADED
from lmkd.scaffold.probe import probe_gradient, synthetic_context


def test_top_level_exports_available():
    for attr in (
        "Criterion",
        "LossContext",
        "LossOutput",
        "LossTerm",
        "ModelPairInfo",
        "PrepareContext",
        "FeatureRequest",
        "SideRequest",
        "SideOutput",
        "TeacherSignalSpec",
        "TeacherOutput",
        "Batch",
        "FieldSpec",
        "FIELDS",
        "BaseConfig",
        "Plugin",
        "CRITERIONS",
        "TRANSFORMS",
        "ALIGNERS",
        "PRODUCERS",
        "TEACHERS",
        "TEACHER_INPUTS",
        "ALGORITHMS",
        "STEP_STRATEGIES",
        "CALLBACKS",
        "EVALUATORS",
        "STAGES",
    ):
        assert hasattr(lmkd, attr), f"lmkd.{attr} missing"


def test_loss_decorator_without_explicit_name_and_input_helpers():
    @lmkd.loss
    def ergo_bare_loss(s: lmkd.Inputs) -> lmkd.TokenLoss:
        assert s.input_ids is not None
        assert s.attention_mask is not None
        return lmkd.TokenLoss(s.logits.square().mean(-1))

    @lmkd.loss(teacher=True, chunked=False)
    def ergo_kw_teacher_loss(s: lmkd.Inputs, t: lmkd.Inputs) -> lmkd.TokenLoss:
        diff = (s.last_hidden - t.last_hidden).square().mean(-1)
        pos_diff = (s.positional_hidden(-1) - t.positional_hidden(-1)).square().mean(-1)
        return lmkd.TokenLoss(diff + 0.1 * pos_diff, s.mask)

    assert "ergo_bare_loss" in lmkd.CRITERIONS
    assert "ergo_kw_teacher_loss" in lmkd.CRITERIONS

    out1 = probe_gradient(lmkd.CRITERIONS.build("ergo_bare_loss"))
    assert torch.isfinite(out1.terms["loss"].value)

    out2 = probe_gradient(lmkd.CRITERIONS.build("ergo_kw_teacher_loss"))
    assert torch.isfinite(out2.terms["loss"].value)


def test_loss_context_shortcuts_and_declarative_criterion():
    class ErgoDeclCriterion(lmkd.Criterion):
        needs_teacher = True
        student_logits_mode = "view"
        teacher_logits_mode = "view"
        student_hidden_layers = (-2, -1)
        teacher_hidden_layers = (-2, -1)

        def build_modules(self, info: lmkd.ModelPairInfo, modules) -> None:
            t_dim = info.teacher_hidden or info.student_hidden
            self.register_aux_module(modules, "proj", nn.Linear(info.student_hidden, t_dim, bias=False))

        def forward(self, ctx: lmkd.LossContext) -> lmkd.LossOutput:
            s_logp = ctx.student_logits().target_logp(ctx.labels)
            t_logits = ctx.teacher_logits().full()
            s_h = self.proj(ctx.student_target_hidden(-2))
            t_h = ctx.teacher_target_hidden(-2).detach()
            h_loss = (s_h - t_h).square().mean(-1)
            val = (-(s_logp * 0.1) + h_loss + 0.0 * t_logits.sum(dim=-1)) * ctx.mask
            assert ctx.input_ids.shape == ctx.labels.shape
            assert ctx.attention_mask is not None
            return lmkd.LossOutput(terms={"ergo": lmkd.LossTerm(val.sum(), "token_sum")})

    crit = ErgoDeclCriterion()
    req = crit.features()
    assert req.student.logits == "view"
    assert req.student.hidden == (-2, -1)
    assert req.teacher is not None and req.teacher.logits == "view"
    assert req.teacher.hidden == (-2, -1)
    assert req.teacher_signal is not None and req.teacher_signal.kind == "full"

    out = probe_gradient(crit)
    assert torch.isfinite(out.terms["ergo"].value)

    # Also verify synthetic_context provides multi-layer hiddens and lm_head (e.g. for FDD)
    ctx, _ = synthetic_context()
    assert ctx.student_hidden(-4).shape == (2, 4, 16)
    assert getattr(ctx.student.model, "lm_head", None) is not None


def test_cli_try_fdd_and_init_minimal_auto_discover(tmp_path: Path, monkeypatch):
    # FDD requires multi-layer hidden states and lm_head resolution -> works cleanly with upgraded probe_gradient
    assert main(["try", "fdd"]) == 0

    monkeypatch.chdir(tmp_path)
    (tmp_path / "examples" / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "examples" / "data" / "train_sample.jsonl").write_text(
        '{"prompt": "hello", "response": "world"}\n',
        encoding="utf-8",
    )

    assert main(["init", "minimal_kd", "--minimal"]) == 0

    pkg_dir = tmp_path / "dev" / "minimal_kd"
    assert (pkg_dir / "__init__.py").is_file()
    assert (pkg_dir / "loss.py").is_file()
    assert (pkg_dir / "recipe.yaml").is_file()
    assert not (pkg_dir / "train.py").exists()
    assert not (pkg_dir / "data.py").exists()

    # Check recipe validity
    rep = check_recipe(load_recipe(pkg_dir / "recipe.yaml"))
    assert rep.ok, rep.render()

    # Auto-discovery of dev/minimal_kd by `lmkd try minimal_kd` without passing --plugins
    lmkd.CRITERIONS._items.pop("minimal_kd", None)
    assert main(["try", "minimal_kd"]) == 0


def test_standalone_dev_file_auto_discovery_without_imports(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "examples" / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "examples" / "data" / "train_sample.jsonl").write_text(
        '{"prompt": "hello", "response": "world"}\n',
        encoding="utf-8",
    )

    dev_dir = tmp_path / "dev"
    dev_dir.mkdir(parents=True, exist_ok=True)
    (dev_dir / "my_fkl.py").write_text(
        """import torch.nn.functional as F
import lmkd
from lmkd import Inputs, TokenLoss


@lmkd.loss("entropy_weighted_fkl", teacher=True)
def entropy_weighted_fkl(s: Inputs, t: Inputs, temperature: float = 1.0, alpha: float = 1.0) -> TokenLoss:
    t_probs = F.softmax(t.logits / temperature, dim=-1)
    t_logp = F.log_softmax(t.logits / temperature, dim=-1)
    s_logp = F.log_softmax(s.logits / temperature, dim=-1)
    kl = (t_probs * (t_logp - s_logp)).sum(dim=-1)
    entropy = -(t_probs * t_logp).sum(dim=-1).detach()
    return TokenLoss(kl * (1.0 + alpha * entropy) * (temperature ** 2), s.mask)
""",
        encoding="utf-8",
    )

    recipe_path = tmp_path / "recipe3.yaml"
    recipe_path.write_text(
        """name: gkd_entropy_fkl

models:
  student:
    path: Qwen/Qwen2.5-0.5B
    trainable: true
  teacher:
    path: Qwen/Qwen2.5-1.5B

data:
  train:
    type: jsonl
    path: examples/data/train_sample.jsonl
  transforms:
    - type: tokenize_sft
      use_chat_template: false

teacher:
  provider: hf_frozen
  input: same

producer:
  type: mixed
  lmbda: 0.5
  sampling:
    max_new_tokens: 128

rollout:
  type: hf

loss:
  terms:
    - criterion: nll
      weight: 0.2
    - criterion: entropy_weighted_fkl
      weight: 0.8
      params:
        temperature: 1.5
""",
        encoding="utf-8",
    )

    lmkd.CRITERIONS._items.pop("entropy_weighted_fkl", None)
    _USER_LOADED.discard(str((dev_dir / "my_fkl.py").resolve()))
    rep = check_recipe(load_recipe(recipe_path))
    assert rep.ok, rep.render()

    lmkd.CRITERIONS._items.pop("entropy_weighted_fkl", None)
    _USER_LOADED.discard(str((dev_dir / "my_fkl.py").resolve()))
    assert main(["try", "entropy_weighted_fkl"]) == 0
