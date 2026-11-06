import pytest

from lmkd.core.config import apply_overrides, resolve_interpolations


def test_env_interp(monkeypatch):
    monkeypatch.setenv("X", "/p")
    cfg1 = {"val": "${env:X}"}
    out1 = resolve_interpolations(cfg1)
    assert out1["val"] == "/p"

    cfg2 = {"val": "${env:NOPE,3}"}
    out2 = resolve_interpolations(cfg2)
    assert out2["val"] == 3

    cfg3 = {"val": "${env:NOPE}"}
    with pytest.raises(KeyError) as exc_info:
        resolve_interpolations(cfg3)
    assert "NOPE" in str(exc_info.value)


def test_interp_error_path():
    cfg = {"a": {"b": "${zz.y}"}}
    with pytest.raises(KeyError) as exc_info:
        resolve_interpolations(cfg)
    assert "a.b" in str(exc_info.value)


def test_append_and_delete():
    base = {
        "loss": {"terms": [{"criterion": "nll"}]},
        "train": {"seed": 42},
        "a": {"foo": "bar"},
    }
    # Append
    out = apply_overrides(base, ["loss.terms+=[{criterion: x}]"])
    assert len(out["loss"]["terms"]) == 2
    assert out["loss"]["terms"][1] == {"criterion": "x"}

    # Delete
    out2 = apply_overrides(base, ["train.seed~"])
    assert "seed" not in out2["train"]

    # Append to non-list should raise ValueError
    with pytest.raises(ValueError):
        apply_overrides(base, ["a+=1"])


def test_override_yaml11_boolean_trap():
    base = {"engine": {"mixed_precision": "bf16"}}
    out = apply_overrides(base, ["engine.mixed_precision=no"])
    assert out["engine"]["mixed_precision"] == "no"
