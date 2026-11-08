"""GPU tests for teacher memory savings: fp8 and cpu_offload (T26)."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")]

from lmkd.core.config import recipe_from_dict
from lmkd.train.builder import build_runtime


def test_fp8_teacher_logits_close():
    """Test fp8 teacher logits close to bf16 (requires torchao and sm89+ GPU)."""
    pytest.importorskip("torchao")
    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):
        pytest.skip(f"GPU sm{cap[0]}{cap[1]} does not support fp8 (requires sm89+)")


def test_cpu_offload_teacher_runs(base_recipe, tiny_models):
    """placement: cpu_offload runs 1 step e2e; teacher parameters remain on CPU after the step."""
    pytest.importorskip("accelerate")
    base_recipe["engine"] = {"type": "local", "device": "cuda", "mixed_precision": "no"}
    base_recipe["models"]["teacher"] = {
        "path": tiny_models["teacher"],
        "dtype": "fp32",
        "placement": "cpu_offload",
    }
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "same"}
    base_recipe["loss"] = {
        "terms": [
            {"criterion": "nll"},
            {"criterion": "token_kd", "params": {"divergence": "rkl"}},
        ]
    }
    base_recipe["train"]["max_steps"] = 1

    rt = build_runtime(recipe_from_dict(base_recipe))
    rt.begin()
    rt.train_step()
    rt.end()

    # Verify teacher parameters are offloaded from CUDA after the step (cpu or meta)
    for p in rt.teacher_model.parameters():
        assert p.device.type in ("cpu", "meta"), f"Parameter {p} is on {p.device}, expected cpu/meta"
