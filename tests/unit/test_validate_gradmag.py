"""Static validation of GradMag's aggregator and step pairing."""

import pytest

from lmkd.core.config import recipe_from_dict
from lmkd.core.validate import check_recipe


@pytest.mark.parametrize(
    "step",
    [
        "standard",
        {"type": "standard"},
        {"type": "multi_objective_grad", "combine": "sum"},
    ],
)
def test_gradmag_requires_matching_step(base_recipe, step):
    base_recipe["loss"]["aggregator"] = "gradmag"
    base_recipe["step"] = step

    report = check_recipe(recipe_from_dict(base_recipe))

    assert not report.ok
    assert any("gradmag" in error and "multi_objective_grad" in error for error in report.errors)


@pytest.mark.parametrize(
    "step",
    [
        "multi_objective_grad",
        {"type": "multi_objective_grad", "params": {"combine": "gradmag"}},
    ],
)
def test_gradmag_matching_step_is_valid(base_recipe, step):
    base_recipe["loss"]["aggregator"] = {"type": "gradmag", "params": {"mode": "approx_gradmag"}}
    base_recipe["step"] = step

    report = check_recipe(recipe_from_dict(base_recipe))

    assert report.ok, report.render()


def test_multi_objective_grad_requires_gradmag_aggregator(base_recipe):
    base_recipe["step"] = {"type": "multi_objective_grad", "combine": "gradmag"}

    report = check_recipe(recipe_from_dict(base_recipe))

    assert not report.ok
    assert any("combine=gradmag" in error and "aggregator: gradmag" in error for error in report.errors)


def test_multi_objective_sum_accepts_weighted_sum(base_recipe):
    base_recipe["step"] = {"type": "multi_objective_grad", "combine": "sum"}

    report = check_recipe(recipe_from_dict(base_recipe))

    assert report.ok, report.render()
