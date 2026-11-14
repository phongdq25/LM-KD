"""Library mode (lmkd.api), loss tiers 0/1, plugin imports, injected objects, examples/."""

import copy
import importlib.util
import json
import textwrap
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lmkd
from lmkd.cli import main
from lmkd.core.config import deep_merge, load_yaml_with_bases, recipe_from_dict
from lmkd.core.registry import RegistryError
from lmkd.core.validate import check_recipe
from lmkd.losses.base import LossOutput, LossTerm
from lmkd.plugins import _discover, load_plugins

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _kd_recipe(base_recipe, tiny_models):
    r = copy.deepcopy(base_recipe)
    r["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    r["teacher"] = {"provider": "hf_frozen", "input": "same"}
    return r


def _params(rt):
    return torch.cat([p.detach().flatten().cpu() for p in rt.student_model.parameters()])


def _nll_token_sum(batch, rt):
    """Tier 0 written with the efficient API: identical to criterion `nll`."""
    s = rt.forward(batch)
    mask = (batch["student.labels"] != -100).float()
    logp = s.side.logits.target_logp(batch["student.labels"])
    return LossOutput({"nll": LossTerm(-(logp * mask).sum(), "token_sum")})


def test_hand_written_loop_with_loss_fn_equals_recipe(base_recipe):
    ref = lmkd.run(copy.deepcopy(base_recipe))  # criterion nll through Trainer.fit path

    r = copy.deepcopy(base_recipe)
    r["loss"] = {"terms": []}
    r["train"]["output_dir"] += "_lib"
    rt = lmkd.build(r, loss_fn=_nll_token_sum)
    rt.begin()
    while not rt.done:
        out = rt.train_step()
        assert "loss/loss_fn/nll" in out.metrics
    rt.end()
    assert rt.state.step == ref.state.step == 4
    torch.testing.assert_close(_params(rt), _params(ref), atol=1e-6, rtol=0)


def test_train_step_loss_fn_is_added_to_criteria(base_recipe):
    rt = lmkd.build(copy.deepcopy(base_recipe))
    rt.begin()
    out = rt.train_step(loss_fn=lambda b, rt_: {"zero": rt_.forward(b).logits.sum() * 0.0})
    assert {"loss/nll/nll", "loss/loss_fn/zero"} <= set(out.metrics)
    out2 = rt.train_step()  # override only lasts one step
    assert "loss/loss_fn/zero" not in out2.metrics


def test_tier1_register_loss_matches_builtin_token_kd(base_recipe, tiny_models):
    load_plugins([str(EXAMPLES / "my_losses.py")])
    r = _kd_recipe(base_recipe, tiny_models)
    r["loss"] = {
        "terms": [
            {"criterion": "token_kd", "alias": "builtin", "params": {"divergence": "fkl"}},
            {"criterion": "example_fkl", "alias": "mine"},
        ]
    }
    rt = lmkd.build(r)
    micro = rt.step_strategy.micro_batches(rt.algorithm.augment(rt.next_batch(), rt))[0]
    outs = {o.alias: o.out for o in rt.algorithm.loss(micro, rt)}
    a = next(iter(outs["builtin"].terms.values()))
    b = next(iter(outs["mine"].terms.values()))
    assert a.reduction == b.reduction == "token_sum"
    torch.testing.assert_close(a.value, b.value, atol=1e-4, rtol=1e-5)


def test_tier0_recipe_with_imports_via_cli(base_recipe, tiny_models, tmp_path):
    raw = load_yaml_with_bases(EXAMPLES / "recipes" / "tier0_compute_loss.yaml")
    assert raw["imports"] == [str((EXAMPLES / "my_losses.py").resolve())]  # relative to the YAML file
    tiny = _kd_recipe(base_recipe, tiny_models)
    tiny["train"]["output_dir"] = str(tmp_path / "out")
    r = deep_merge(raw, {k: tiny[k] for k in ("models", "data", "engine", "optim", "train")} | {"eval": []})
    r["optim"]["schedulers"] = [{"type": "constant"}]
    path = tmp_path / "r.yaml"
    path.write_text(json.dumps(r))
    assert main(["check", "-c", str(path)]) == 0
    assert main(["train", "-c", str(path)]) == 0
    rows = [json.loads(line) for line in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()]
    assert {"loss/loss_fn/ce", "loss/loss_fn/kd"} <= set(rows[-1])


def test_check_reports_unknown_compute_loss_and_empty_loss(base_recipe):
    r = copy.deepcopy(base_recipe)
    r["algorithm"] = {"type": "compute_loss", "fn": "does_not_exist"}
    assert any("does_not_exist" in e for e in check_recipe(recipe_from_dict(r)).errors)
    r = copy.deepcopy(base_recipe)
    r["loss"] = {"terms": []}
    assert any("loss.terms is empty" in e for e in check_recipe(recipe_from_dict(r)).errors)


def test_user_plugin_file_relative_import_and_cli_flag(base_recipe, tmp_path):
    (tmp_path / "plug").mkdir()
    (tmp_path / "plug" / "my_crit.py").write_text(
        textwrap.dedent('''
        from lmkd.core.registry import CRITERIONS
        from lmkd.losses.sft import NLL

        @CRITERIONS.register("t27_user_nll")
        class UserNLL(NLL):
            """NLL copy defined outside the package."""
    ''')
    )
    r = copy.deepcopy(base_recipe)
    r["loss"] = {"terms": [{"criterion": "t27_user_nll"}]}
    r["imports"] = ["plug/my_crit.py"]
    (tmp_path / "recipe.yaml").write_text(json.dumps(r))
    assert main(["check", "-c", str(tmp_path / "recipe.yaml")]) == 0  # resolved next to the YAML
    r.pop("imports")
    (tmp_path / "recipe2.yaml").write_text(json.dumps(r))
    assert (
        main(["check", "-c", str(tmp_path / "recipe2.yaml"), "--plugins", str(tmp_path / "plug" / "my_crit.py")]) == 0
    )
    assert main(["list", "criterions"]) == 0


def test_build_with_injected_objects_and_callable_evaluator(base_recipe, tiny_models, toy_jsonl):
    rows = [json.loads(line) for line in Path(toy_jsonl).read_text().splitlines()]
    r = copy.deepcopy(base_recipe)
    del r["models"], r["data"]["train"]
    rt = lmkd.build(
        r,
        student_model=AutoModelForCausalLM.from_pretrained(tiny_models["student"]),
        student_tokenizer=AutoTokenizer.from_pretrained(tiny_models["student"]),
        train_records=rows,
        evaluators=[lambda rt_: {"custom/x": 1.0}],
    )
    rt.fit()
    assert rt.state.step == 4 and len(rt.train_dataset) == len(rows)
    assert rt.evaluate() == {"custom/x": 1.0} and rt.state.extra["eval"]["custom/x"] == 1.0


def test_examples_custom_loop_runs(base_recipe, tiny_models):
    spec = importlib.util.spec_from_file_location("lmkd_example_custom_loop", EXAMPLES / "custom_loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rt = mod.main(_kd_recipe(base_recipe, tiny_models), eval_every=2)
    assert rt.state.step == 4 and "my_eval/token_acc" in rt.state.extra["eval"]


def test_builtin_plugins_are_auto_discovered():
    mods = _discover()
    assert "lmkd.losses.hf_compat" in mods and "lmkd.eval.embedding" in mods and "lmkd.cli" not in mods


def test_register_loss_rejects_name_clash_with_builtin():
    with pytest.raises(RegistryError, match="already registered"):

        @lmkd.register_loss("nll")
        def nll(s, t, batch):
            return s.logits.sum()
