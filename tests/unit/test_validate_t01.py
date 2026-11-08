from lmkd.core.config import recipe_from_dict
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS, TEACHERS
from lmkd.core.validate import check_recipe, preflight
from lmkd.losses.base import Criterion
from lmkd.teachers.providers import TeacherProvider


@TEACHERS.register("_test_mock_topk_provider", force=True)
class _MockTopkTeacher(TeacherProvider):
    capabilities = frozenset({"topk"})

    class Config(BaseConfig):
        pass

    def signals(self, batch, spec, req, student):
        return None


@CRITERIONS.register("_test_no_packing_crit", force=True)
class _NoPackingCrit(Criterion):
    supports_packing = False

    class Config(BaseConfig):
        pass

    def forward(self, ctx):
        pass


def test_teacher_capability_error(base_recipe, tiny_models):
    base_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    base_recipe["teacher"] = {"provider": "_test_mock_topk_provider", "input": "same"}
    base_recipe["loss"]["terms"] = [{"criterion": "token_kd", "alias": "my_kd", "params": {"divergence": "fkl"}}]
    cfg = recipe_from_dict(base_recipe)
    rep = check_recipe(cfg)
    assert not rep.ok
    assert any("cannot produce signal 'full'" in e and "my_kd" in e for e in rep.errors)


def test_warnings(base_recipe, tiny_models):
    base_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    base_recipe["teacher"] = {"provider": "none"}
    base_recipe["loss"]["terms"] = [
        {"criterion": "nll", "weight": 0.0, "alias": "zero_nll"},
    ]
    base_recipe["train"]["max_steps"] = 10
    base_recipe["train"]["epochs"] = 1.0
    cfg = recipe_from_dict(base_recipe)
    rep = check_recipe(cfg)
    assert any("teacher model loaded but no teacher provider (wasted memory)" in w for w in rep.warnings)
    assert any("term 'zero_nll' has weight 0 and is skipped" in w for w in rep.warnings)
    assert any("max_steps overrides epochs" in w for w in rep.warnings)
    # warnings should not cause ok to be False if there are no errors
    assert rep.ok


def test_packing_conflict(base_recipe):
    base_recipe["data"]["packing"] = True
    base_recipe["loss"]["terms"] = [{"criterion": "_test_no_packing_crit"}]
    cfg = recipe_from_dict(base_recipe)
    rep = check_recipe(cfg)
    assert not rep.ok
    assert any("packing" in e.lower() for e in rep.errors)


def test_preflight_ok(base_recipe):
    cfg = recipe_from_dict(base_recipe)
    rep = preflight(cfg)
    assert rep.ok, rep.render()
    info_str = "\n".join(rep.info)
    assert "student.input_ids" in info_str
    assert "loss/nll/nll" in info_str


def test_preflight_reports_error(base_recipe):
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    base_recipe["loss"]["terms"] = [{"criterion": "token_kd", "params": {"divergence": "fkl"}}]
    base_recipe["models"]["teacher"] = None
    cfg = recipe_from_dict(base_recipe)
    rep = preflight(cfg)
    assert not rep.ok
    assert any("preflight" in e.lower() for e in rep.errors)
