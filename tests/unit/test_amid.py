"""Unit tests for AMID divergence validation and edge cases."""

from __future__ import annotations

import pytest
import torch

from lmkd.losses.functional.divergences import amid, get_divergence
from lmkd.losses.token_kd import TokenKD


def test_amid_divergence_lookup():
    fn = get_divergence("amid", alpha=0.5, lam=0.5, div_name="fkl", div_order="pr")
    s = torch.randn(2, 4, 10)
    t = torch.randn(2, 4, 10)
    out = fn(s, t)
    assert out.shape == (2, 4)


def test_amid_invalid_parameters():
    s = torch.randn(2, 4, 10)
    t = torch.randn(2, 4, 10)

    # Invalid div_name
    with pytest.raises(ValueError, match="Unknown div_name"):
        amid(s, t, div_name="invalid_name")

    # Invalid div_order in fkl
    with pytest.raises(ValueError, match="Unknown div_order"):
        amid(s, t, div_name="fkl", div_order="invalid_order")

    # Invalid div_order in ab
    with pytest.raises(ValueError, match="Unknown div_order"):
        amid(s, t, div_name="ab", div_order="invalid_order")

    # Invalid temperature
    with pytest.raises(ValueError, match="temperature must be positive"):
        amid(s, t, temperature=-0.5)

    # Invalid ab parameters
    with pytest.raises(ValueError, match="Invalid ab_alpha"):
        amid(s, t, div_name="ab", ab_alpha=0.0, ab_beta=1.0)


def test_amid_token_kd_config():
    cfg = TokenKD.Config(
        divergence="amid",
        amid_alpha=0.7,
        amid_lam=0.3,
        amid_div_name="ab",
        amid_div_order="qr",
        ab_alpha=0.2,
        ab_beta=0.7,
    )
    crit = TokenKD(cfg)
    assert crit.cfg.divergence == "amid"
    assert crit.cfg.amid_alpha == 0.7
    assert crit.cfg.amid_lam == 0.3
    assert crit.cfg.amid_div_name == "ab"
    assert crit.cfg.amid_div_order == "qr"
