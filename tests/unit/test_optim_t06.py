import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import transformers

from lmkd.core import registry as R
from lmkd.core.config import load_yaml_with_bases, recipe_from_dict
from lmkd.core.validate import check_recipe
from lmkd.train.optim import NamedParam, build_param_groups, talas_groups


def _p(name, shape):
    return NamedParam(name, torch.nn.Parameter(torch.randn(*shape)), "student")


def test_muon_rejects_1d():
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon not available in this PyTorch version")
    b = R.OPTIMIZERS.build("muon")
    # A group with 1-D parameter (e.g. bias)
    p1d = torch.nn.Parameter(torch.randn(10))
    groups = [{"params": [p1d], "names": ["bias"], "lr": 1e-3, "weight_decay": 0.0}]
    with pytest.raises(ValueError, match="2-D"):
        b.build(groups)


def test_muon_and_adamw_together():
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon not available in this PyTorch version")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = MLP()
    named = [NamedParam(n, p, "student") for n, p in model.named_parameters()]
    rules = [
        {"name": "2d", "match": {"ndim": 2}, "optimizer": "muon", "lr": 1e-3},
        {"name": "bias", "match": {"ndim_le": 1}, "optimizer": "adamw", "lr": 1e-4},
    ]
    groups = build_param_groups(named, rules, default_lr=1e-3, default_wd=0.01)
    assert set(groups.keys()) == {"muon", "adamw"}

    # Verify each parameter belongs to exactly one optimizer
    all_muon_params = [id(p) for g in groups["muon"] for p in g["params"]]
    all_adamw_params = [id(p) for g in groups["adamw"] for p in g["params"]]
    assert len(all_muon_params) == 2  # fc1.weight, fc2.weight
    assert len(all_adamw_params) == 2  # fc1.bias, fc2.bias
    assert set(all_muon_params).isdisjoint(set(all_adamw_params))

    # Build optimizers
    opt_muon = R.OPTIMIZERS.build("muon").build(groups["muon"])
    opt_adamw = R.OPTIMIZERS.build("adamw").build(groups["adamw"])

    # 1 train step changes all params
    initial_params = {n: p.clone() for n, p in model.named_parameters()}
    x = torch.randn(4, 8)
    loss = model(x).sum()
    loss.backward()

    opt_muon.step()
    opt_adamw.step()

    for n, p in model.named_parameters():
        assert not torch.equal(p, initial_params[n]), f"Parameter {n} was not updated"


def test_talas_groups_fixed_vs_compat():
    p = _p("encoder.layer.10.attention.self.query.weight", (4, 4))

    # Fixed version (compat=False)
    groups_fixed = build_param_groups([p], talas_groups(compat=False), default_lr=2e-5, default_wd=0.01)
    assert "muon" in groups_fixed
    muon_names = [n for g in groups_fixed["muon"] for n in g["names"]]
    assert muon_names == ["encoder.layer.10.attention.self.query.weight"]
    assert groups_fixed["muon"][0]["group_name"] == "muon_body"

    # Compat version (compat=True)
    groups_compat = build_param_groups([p], talas_groups(compat=True), default_lr=2e-5, default_wd=0.01)
    assert "default" in groups_compat
    early_names = [n for g in groups_compat["default"] for n in g["names"]]
    assert early_names == ["encoder.layer.10.attention.self.query.weight"]
    assert groups_compat["default"][0]["group_name"] == "early"


def test_polynomial_matches_hf():
    p = nn.Parameter(torch.zeros(1))
    opt_hf = torch.optim.SGD([p], lr=1.0)
    sch_hf = transformers.get_polynomial_decay_schedule_with_warmup(
        opt_hf, num_warmup_steps=0, num_training_steps=20, lr_end=0.0, power=0.5
    )

    opt_our = torch.optim.SGD([p], lr=1.0)
    b = R.SCHEDULERS.build({"type": "polynomial", "warmup_ratio": 0.0, "power": 0.5, "min_lr_ratio": 0.0})
    sch_our = b.build(opt_our, total_steps=20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(20):
            lr_hf = sch_hf.get_last_lr()[0]
            lr_our = sch_our.get_last_lr()[0]
            assert pytest.approx(lr_hf, rel=1e-6, abs=1e-6) == lr_our
            sch_hf.step()
            sch_our.step()


def test_cosine_min_ratio_per_group():
    p1 = nn.Parameter(torch.zeros(1))
    p2 = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD(
        [
            {"params": [p1], "lr": 1e-3, "initial_lr": 1e-3},
            {"params": [p2], "lr": 1e-4, "initial_lr": 1e-4},
        ]
    )
    b = R.SCHEDULERS.build({"type": "cosine", "warmup_ratio": 0.0, "min_lr_ratio": 0.1})
    sch = b.build(opt, total_steps=20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(20):
            sch.step()

    lrs = sch.get_last_lr()
    assert pytest.approx(1e-4, rel=1e-6) == lrs[0]
    assert pytest.approx(1e-5, rel=1e-6) == lrs[1]


def test_wsd_shape():
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    b = R.SCHEDULERS.build(
        {
            "type": "wsd",
            "warmup_ratio": 0.2,  # 4 steps warmup
            "decay_ratio": 0.3,  # 6 steps decay, 10 steps stable
            "min_lr_ratio": 0.0,
            "decay": "linear",
        }
    )
    sch = b.build(opt, total_steps=20)
    lrs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(20):
            lrs.append(sch.get_last_lr()[0])
            sch.step()

    # 1. Warmup (increasing): step 0 to 4
    for i in range(1, 4):
        assert lrs[i] > lrs[i - 1]
    assert pytest.approx(1.0, rel=1e-5) == lrs[4]

    # 2. Stable (constant ~ 1.0): step 4 to 14
    for i in range(4, 14):
        assert pytest.approx(1.0, rel=1e-5) == lrs[i]

    # 3. Decay (decreasing): step 14 to 19
    for i in range(15, 20):
        assert lrs[i] < lrs[i - 1]
    assert lrs[-1] < 0.2


def test_lion_builder():
    pytest.importorskip("pytorch_optimizer")
    b = R.OPTIMIZERS.build("lion")
    p = nn.Parameter(torch.randn(2, 2))
    opt = b.build([{"params": [p], "lr": 1e-4, "weight_decay": 0.01}])
    assert opt.__class__.__name__ == "Lion"


def test_optim_muon_talas_recipe(base_recipe):
    base_path = Path("recipes/_base_/optim_muon_talas.yaml")
    assert base_path.exists()
    optim_cfg = load_yaml_with_bases(base_path)
    base_recipe["optim"] = optim_cfg["optim"]
    cfg = recipe_from_dict(base_recipe)
    res = check_recipe(cfg)
    assert res.ok, res.render()
