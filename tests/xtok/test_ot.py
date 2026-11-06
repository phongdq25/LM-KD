"""MCW-KD auxiliary module ownership and cost-weight updates."""

import copy

import pytest
import torch

from lmkd.core.config import recipe_from_dict
from lmkd.losses.base import ModelPairInfo, PrepareContext
from lmkd.models.modules import ModuleRegistry
from lmkd.train.builder import build_runtime


def _recipe(base_recipe, xtok_models, tmp_path, compat=False, interval=1000):
    recipe = copy.deepcopy(base_recipe)
    recipe["models"]["student"] = {"path": xtok_models["student"], "dtype": "fp32"}
    recipe["models"]["teacher"] = {"path": xtok_models["teacher"], "dtype": "fp32"}
    recipe["teacher"] = {"provider": "hf_frozen", "input": "pretokenized"}
    recipe["data"]["transforms"] = [
        {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"},
    ]
    recipe["loss"] = {
        "terms": [
            {
                "criterion": "ot_logits",
                "params": {
                    "top_k_vocab": 8,
                    "compat_untrained_aux": compat,
                    "update_cost_weights": interval < 1000,
                    "update_interval": interval,
                },
            }
        ]
    }
    recipe["optim"]["groups"] = [
        {"name": "mcw_untrained", "match": {"group": "mcw_untrained"}, "lr": 0.0, "weight_decay": 0.0},
        {"name": "default", "match": {}},
    ]
    recipe["train"].update(output_dir=str(tmp_path / "ot"), max_steps=3, batch_size=2, micro_batch_size=2)
    return recipe


def test_aux_params_in_optimizer(base_recipe, xtok_models, tmp_path):
    rt = build_runtime(recipe_from_dict(_recipe(base_recipe, xtok_models, tmp_path)))
    names = [p.name for p in rt.trainable_params]
    assert any("mcw__salience" in n for n in names)
    assert any("mcw__ot" in n for n in names)
    before = {k: v.clone() for k, v in rt.modules.state_dict().items()}
    rt.begin()
    rt.train_step()
    rt.end()
    assert any(not torch.equal(before[k], v) for k, v in rt.modules.state_dict().items())


def test_compat_untrained_aux(base_recipe, xtok_models, tmp_path):
    rt = build_runtime(recipe_from_dict(_recipe(base_recipe, xtok_models, tmp_path, compat=True)))
    before = {k: v.clone() for k, v in rt.modules.state_dict().items()}
    rt.begin()
    for _ in range(3):
        rt.train_step()
    rt.end()
    assert all(torch.equal(before[k], v) for k, v in rt.modules.state_dict().items())


def test_qp_update(base_recipe, xtok_models, tmp_path):
    pytest.importorskip("cvxpy")
    rt = build_runtime(recipe_from_dict(_recipe(base_recipe, xtok_models, tmp_path, interval=2)))
    crit = rt.criteria[0].criterion
    before = crit.cost_w.clone()
    rt.begin()
    rt.train_step()
    rt.train_step()
    rt.end()
    assert not torch.equal(before, crit.cost_w)
    torch.testing.assert_close(crit.cost_w.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)
    assert (crit.cost_w >= 0.01 - 1e-6).all()


def test_missing_qp_dependency(monkeypatch):
    from baselines.cross_tokenizer.mcw_kd import OTLogits

    monkeypatch.setitem(__import__("sys").modules, "cvxpy", None)
    with pytest.raises(ImportError, match=r"pip install -e"):
        OTLogits().prepare(PrepareContext(rt=None))


def test_shared_module_once():
    from baselines.cross_tokenizer.mcw_kd import OTHidden, OTLogits

    mods = ModuleRegistry()
    info = ModelPairInfo(6, 1, 9, 8, 1, 11)
    a, b = OTLogits(), OTHidden()
    a.build_modules(info, mods)
    ot = mods.get("mcw__ot")
    b.build_modules(info, mods)
    assert mods.get("mcw__ot") is ot
    assert list(mods.mods).count("mcw__ot") == 1


def test_mcw_recipe_tiny_step(base_recipe, xtok_models, tmp_path):
    recipe = _recipe(base_recipe, xtok_models, tmp_path)
    recipe["data"]["transforms"] += [
        {"type": "load_vocab_map", "cache_path": str(tmp_path / "vocab.pt")},
        {"type": "align", "aligner": {"type": "dtw_edit", "full_sequence": True}},
    ]
    recipe["loss"] = {
        "aggregator": {"type": "weighted_sum"},
        "terms": [
            {"criterion": "nll"},
            {
                "criterion": "dskd_cma",
                "alias": "cma",
                "weight": 5.0,
                "params": {"temperature": 3.0, "kd_objective": "forward_kl"},
            },
            {"criterion": "ot_logits", "params": {"top_k_vocab": 8, "update_cost_weights": False}},
            {"criterion": "ot_hidden", "params": {"window": 1, "update_cost_weights": False}},
        ],
    }
    recipe["train"].update(max_steps=1, batch_size=2, micro_batch_size=2)
    rt = build_runtime(recipe_from_dict(recipe))
    rt.begin()
    result = rt.train_step()
    rt.end()
    assert rt.state.step == 1
    assert result is not None


def test_dual_tiny_step(base_recipe, xtok_models, tmp_path):
    recipe = _recipe(base_recipe, xtok_models, tmp_path)
    recipe["loss"]["terms"][0]["params"].update(dual=True, dual_hidden=16, inner_iters=2)
    recipe["train"]["max_steps"] = 1
    rt = build_runtime(recipe_from_dict(recipe))
    rt.begin()
    rt.train_step()
    rt.end()
    assert rt.state.step == 1
