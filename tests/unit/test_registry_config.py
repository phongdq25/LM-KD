import pytest

from lmkd.core.config import apply_overrides, deep_merge, load_recipe, resolve_interpolations
from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import Registry


def test_registry_build_and_strict():
    reg = Registry("t")

    @reg.register("foo")
    class Foo(Plugin):
        class Config(BaseConfig):
            a: int = 1

    assert reg.build({"type": "foo", "a": 3}).cfg.a == 3
    assert reg.build({"type": "foo", "params": {"a": 4}}).cfg.a == 4
    assert reg.build("foo").cfg.a == 1
    with pytest.raises(ValueError, match="invalid config"):
        reg.build({"type": "foo", "b": 2})  # unknown key -> error (no silent filtering)
    with pytest.raises(KeyError, match="Did you mean"):
        reg.build("fooo")


def test_registry_stub_points_to_task():
    reg = Registry("t")

    @reg.register("bar", status="stub", task="task_spec")
    class Bar(Plugin):
        pass

    with pytest.raises(NotImplementedError, match="task_spec"):
        reg.build("bar")


def test_merge_overrides_interp(tmp_path):
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}
    cfg = apply_overrides({"x": {"y": [{"z": 1}]}}, ["x.y.0.z=5", "lr=1e-4"])
    assert cfg["x"]["y"][0]["z"] == 5 and cfg["lr"] == 1e-4
    r = resolve_interpolations({"a": {"b": 2}, "c": "${a.b}", "d": "v${a.b}"})
    assert r["c"] == 2 and r["d"] == "v2"


def test_lazy_plugin_resolution():
    from lmkd.core.plugin_index import lookup
    from lmkd.core.registry import CRITERIONS

    # Static index resolves token_kd to its module
    mod = lookup("criterions", "token_kd")
    assert mod == "lmkd.losses.token_kd"

    # Registry lookup automatically imports and registers the plugin
    meta = CRITERIONS.meta("token_kd")
    assert meta.name == "token_kd"
    assert "token_kd" in CRITERIONS


def test_compat_probe_has_and_require():
    from lmkd import compat

    assert compat.has("torch")
    assert compat.require("torch") is not None

    assert not compat.has("nonexistent_package_xyz123")
    with pytest.raises(ImportError, match="Feature 'super_cool' requires package 'nonexistent_package_xyz123'"):
        compat.require("nonexistent_package_xyz123", feature="super_cool")


def test_base_inheritance(tmp_path, tiny_models, toy_jsonl):
    (tmp_path / "base.yaml").write_text(
        f"models: {{student: {{path: {tiny_models['student']}}}}}\n"
        f"data: {{train: {{type: jsonl, path: {toy_jsonl}}}}}\n"
        "loss: {terms: [{criterion: nll}]}\n"
        "train: {max_steps: 3}\n"
    )
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\ntrain: {max_steps: 7}\n")
    cfg = load_recipe(tmp_path / "child.yaml", ["train.seed=1"])
    assert cfg.train.max_steps == 7 and cfg.train.seed == 1
    assert cfg.models.student.path == tiny_models["student"]
