"""ResidualKD two-stage tiny CPU training."""

import copy
import os
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from baselines.cross_tokenizer.residualkd import ResidualKD
from lmkd.core.config import recipe_from_dict
from lmkd.core.state import TrainState
from lmkd.train.builder import build_runtime


def test_two_stage_e2e(base_recipe, tiny_models, tmp_path):
    stage1 = copy.deepcopy(base_recipe)
    stage1["models"]["student"] = {"path": tiny_models["teacher"], "dtype": "fp32", "trainable": False}
    stage1["loss"] = {"terms": [{"criterion": "residual_pretrain"}]}
    stage1["train"].update(output_dir=str(tmp_path / "stage1"), max_steps=5, batch_size=2, micro_batch_size=2)
    rt1 = build_runtime(recipe_from_dict(stage1))
    model_before = {k: v.clone() for k, v in rt1.student_model.state_dict().items()}
    ta_before = {k: v.clone() for k, v in rt1.modules.state_dict().items()}
    rt1.begin()
    for _ in range(5):
        rt1.train_step()
    rt1.end()
    assert all(torch.equal(model_before[k], v) for k, v in rt1.student_model.state_dict().items())
    assert any(not torch.equal(ta_before[k], v) for k, v in rt1.modules.state_dict().items())
    ckpt = Path(stage1["train"]["output_dir"]) / "checkpoint-final" / "modules.pt"
    assert "mods.residual_pretrain__ta.P_TA.weight" in torch.load(ckpt, weights_only=True)

    stage2 = copy.deepcopy(base_recipe)
    stage2["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    stage2["teacher"] = {"provider": "hf_frozen", "input": "same"}
    stage2["loss"] = {"terms": [{"criterion": "residualkd", "params": {"projector_ckpt": str(ckpt)}}]}
    stage2["train"].update(output_dir=str(tmp_path / "stage2"), max_steps=3, batch_size=2, micro_batch_size=2)
    rt2 = build_runtime(recipe_from_dict(stage2))
    before = {k: v.clone() for k, v in rt2.modules.state_dict().items()}
    assert torch.count_nonzero(before["mods.residualkd__as.P_AS.weight"]) == 0
    rt2.begin()
    for _ in range(3):
        rt2.train_step()
    rt2.end()
    after = rt2.modules.state_dict()
    assert torch.equal(before["mods.residualkd__ta.P_TA.weight"], after["mods.residualkd__ta.P_TA.weight"])


def test_missing_ckpt_raises():
    crit = ResidualKD()
    with pytest.raises(ValueError, match="stage 1"):
        crit.prepare(None)


def test_warmup_units():
    step = ResidualKD(ResidualKD.Config(lambda_res=0.5, lambda_warmup_steps=50))
    micro = ResidualKD(ResidualKD.Config(lambda_res=0.5, lambda_warmup_steps=50, warmup_unit="micro"))
    assert step.lambda_at(TrainState(step=0, micro_step=25)) == 0
    assert step.lambda_at(TrainState(step=50, micro_step=0)) == 0.5
    assert micro.lambda_at(TrainState(step=0, micro_step=25)) == 0.25


def test_script_two_stage_tiny(base_recipe, tiny_models, tmp_path):
    stage1 = copy.deepcopy(base_recipe)
    stage1["models"]["student"] = {"path": tiny_models["teacher"], "dtype": "fp32", "trainable": False}
    stage1["loss"] = {"terms": [{"criterion": "residual_pretrain"}]}
    stage1["train"].update(output_dir=str(tmp_path / "one"), max_steps=2, batch_size=2, micro_batch_size=2)
    stage2 = copy.deepcopy(base_recipe)
    stage2["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    stage2["teacher"] = {"provider": "hf_frozen", "input": "same"}
    stage2["loss"] = {"terms": [{"criterion": "residualkd", "params": {"d_bottleneck": 64}}]}
    stage2["train"].update(output_dir=str(tmp_path / "two"), max_steps=2, batch_size=2, micro_batch_size=2)
    one, two = tmp_path / "one.yaml", tmp_path / "two.yaml"
    one.write_text(yaml.safe_dump(stage1))
    two.write_text(yaml.safe_dump(stage2))
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PATH": str(root / ".venv" / "bin") + os.pathsep + os.environ["PATH"],
        "RESIDUALKD_STAGE1_RECIPE": str(one),
        "RESIDUALKD_STAGE2_RECIPE": str(two),
        "RESIDUALKD_PROJECTOR_CKPT": str(tmp_path / "one" / "checkpoint-final" / "modules.pt"),
    }
    subprocess.run(
        ["bash", str(root / "scripts" / "run_residualkd_full.sh")],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "two" / "checkpoint-final" / "COMPLETE").exists()
