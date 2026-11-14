import pytest
import torch

from lmkd.models.modules import LinearProjector, ModuleRegistry
from lmkd.train.optim import NamedParam, build_param_groups


def _p(name, shape):
    return NamedParam(name, torch.nn.Parameter(torch.zeros(*shape)), "student")


def test_layer_range_exact_index():
    ps = [
        _p("encoder.layer.1.attn.weight", (2, 2)),
        _p("encoder.layer.10.attn.weight", (2, 2)),
        _p("encoder.layer.1.attn.bias", (2,)),
    ]
    rules = [
        {"name": "nd", "match": {"ndim_le": 1}, "weight_decay": 0.0},
        {"name": "l1", "match": {"layer_range": [1, 1]}, "lr": 0.1},
        {"name": "rest", "match": {}},
    ]
    g = build_param_groups(ps, rules, default_lr=1.0, default_wd=0.01)["default"]
    names = {x["group_name"]: x["names"] for x in g}
    assert names["l1"] == ["encoder.layer.1.attn.weight"]  # layer.10 NOT matched (TALAS bug)
    assert names["rest"] == ["encoder.layer.10.attn.weight"]
    assert names["nd"] == ["encoder.layer.1.attn.bias"]


def test_unmatched_param_errors():
    with pytest.raises(ValueError, match="matched no"):
        build_param_groups([_p("x", (1,))], [{"name": "a", "match": {"ndim": 2}}], default_lr=1, default_wd=0)


def test_module_registry_dotted_names():
    reg = ModuleRegistry()
    reg.add("dskd.t2s", LinearProjector(LinearProjector.Config(in_dim=2, out_dim=3)), owner="dskd")
    reg.add("tmp", LinearProjector(LinearProjector.Config(in_dim=2, out_dim=2)), save=False)
    assert "dskd.t2s" in reg and reg.get("dskd.t2s") is reg.get("dskd__t2s")
    keys = list(reg.saveable_state_dict())
    assert keys == ["mods.dskd__t2s.proj.weight"]
    names = [n for n, _, _ in reg.named_params_with_group()]
    assert names[0] == "modules.dskd__t2s.proj.weight"
