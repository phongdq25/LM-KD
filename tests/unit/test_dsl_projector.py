"""Unit tests for DSLProjector."""

import pytest
import torch
import torch.nn as nn

from lmkd.core.registry import PROJECTORS
from lmkd.models.modules import DSLProjector


def test_parse_structures():
    # "t-s" -> 1 Linear(t, s)
    cfg1 = DSLProjector.Config(structure="t-s", s=32, t=48, bias=True)
    p1 = DSLProjector(cfg1)
    layers1 = list(p1.net.children())
    assert len(layers1) == 1
    assert isinstance(layers1[0], nn.Linear)
    assert layers1[0].in_features == 48
    assert layers1[0].out_features == 32
    assert layers1[0].bias is not None

    # Via registry build
    p1_reg = PROJECTORS.build({"type": "dsl", "structure": "t-s", "s": 32, "t": 48, "bias": True})
    assert isinstance(p1_reg, DSLProjector)

    # "2s-2t" with bias=False
    cfg2 = DSLProjector.Config(structure="2s-2t", s=16, t=24, bias=False)
    p2 = DSLProjector(cfg2)
    layers2 = list(p2.net.children())
    assert len(layers2) == 1
    assert isinstance(layers2[0], nn.Linear)
    assert layers2[0].in_features == 32
    assert layers2[0].out_features == 48
    assert layers2[0].bias is None

    # "s-p-relu-t" -> [Linear(s, p), ReLU, Linear(p, t)]
    cfg3 = DSLProjector.Config(structure="s-p-relu-t", s=10, t=20, p=15, bias=True)
    p3 = DSLProjector(cfg3)
    layers3 = list(p3.net.children())
    assert len(layers3) == 3
    assert isinstance(layers3[0], nn.Linear)
    assert layers3[0].in_features == 10
    assert layers3[0].out_features == 15
    assert isinstance(layers3[1], nn.ReLU)
    assert isinstance(layers3[2], nn.Linear)
    assert layers3[2].in_features == 15
    assert layers3[2].out_features == 20

    # Missing p -> ValueError
    cfg_bad = DSLProjector.Config(structure="s-p-relu-t", s=10, t=20, p=None)
    with pytest.raises(ValueError, match="symbol 'p'"):
        DSLProjector(cfg_bad)


def test_dsl_projector_forward():
    cfg = DSLProjector.Config(structure="2s-2t", s=8, t=12, bias=True)
    p = DSLProjector(cfg)
    x = torch.randn(2, 5, 16)
    out = p(x)
    assert out.shape == (2, 5, 24)


def test_module_registry_smart_lookup_and_debugging():
    from lmkd.losses.base import Criterion
    from lmkd.models.modules import ModuleRegistry

    modules = ModuleRegistry()
    lin = nn.Linear(8, 8)
    modules.add("projector", lin, owner="emo")

    # 1. Namespaced key is canonical
    assert "emo__projector" in modules.keys()
    assert modules.get("emo__projector") is lin

    # 2. Dotted syntax works
    assert modules.get("emo.projector") is lin
    assert "emo.projector" in modules

    # 3. Short name without '__' works automatically
    assert modules.get("projector") is lin
    assert "projector" in modules
    assert modules["projector"] is lin

    # 4. Unknown lookup raises informative KeyError
    with pytest.raises(KeyError, match="Available modules: \\['emo__projector'\\]"):
        _ = modules["non_existent"]

    # 5. Criterion helper methods
    crit = Criterion()
    crit.alias = "my_crit"
    aux = nn.Linear(4, 4)
    crit.register_aux_module(modules, "my_aux", aux)
    assert "my_crit__my_aux" in modules.keys()
    assert crit.get_module(modules, "my_aux") is aux
    assert crit.require_module(modules, "my_aux") is aux

    with pytest.raises(KeyError, match="could not find required module 'bad_mod'"):
        crit.require_module(modules, "bad_mod")

    # 6. Ambiguity handling
    other_aux = nn.Linear(4, 4)
    modules.add("my_aux", other_aux, owner="other_crit")
    # Short name "my_aux" is now ambiguous across my_crit and other_crit
    assert modules.get("my_aux") is None
    # But with owner or dotted syntax it resolves unambiguously
    assert modules.get("my_aux", owner="my_crit") is aux
    assert modules.get("my_crit.my_aux") is aux
    assert crit.get_module(modules, "my_aux") is aux
