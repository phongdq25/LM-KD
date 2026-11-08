"""SRA attention validation and tiny cross-tokenizer training."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.cross_tokenizer.sra import SRA
from lmkd.core.config import recipe_from_dict
from lmkd.train.builder import build_runtime


def test_attention_fallback_error():
    criterion = SRA(SRA.Config(shared_vocab="none"))
    model = SimpleNamespace()
    spec = SimpleNamespace(attn_implementation="sdpa")
    rt = SimpleNamespace(
        student_model=model,
        teacher_model=model,
        student_adapter=SimpleNamespace(),
        teacher_adapter=SimpleNamespace(),
        config=SimpleNamespace(models=SimpleNamespace(student=spec, teacher=spec)),
    )
    with pytest.raises(ValueError, match="eager"):
        criterion.prepare(SimpleNamespace(rt=rt))


def test_sra_runs(base_recipe, xtok_models, tmp_path):
    recipe = copy.deepcopy(base_recipe)
    recipe["models"] = {
        "student": {"path": xtok_models["student"], "dtype": "fp32", "attn_implementation": "eager"},
        "teacher": {"path": xtok_models["teacher"], "dtype": "fp32", "attn_implementation": "eager"},
    }
    recipe["teacher"] = {"provider": "hf_frozen", "input": "pretokenized"}
    recipe["data"]["transforms"] = [
        {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"},
        {"type": "token_offsets", "teacher": True},
        {"type": "align", "aligner": {"type": "lcs_span"}},
    ]
    recipe["loss"] = {
        "aggregator": {"type": "kd_mix", "kd_ratio": 0.5, "ce_alias": "nll"},
        "terms": [{"criterion": "nll"}, {"criterion": "sra", "alias": "sra"}],
    }
    recipe["train"].update(output_dir=str(tmp_path / "sra"), max_steps=3, batch_size=2, micro_batch_size=2, log_every=1)
    rt = build_runtime(recipe_from_dict(recipe))
    captured = []
    param = next(rt.modules.get("sra__sra_proj").parameters())
    hook = param.register_hook(lambda grad: captured.append(grad.detach().norm().item()))
    rt.begin()
    for _ in range(3):
        rt.train_step()
    rt.end()
    hook.remove()
    lines = (Path(recipe["train"]["output_dir"]) / "metrics.jsonl").read_text().splitlines()
    assert len(lines) >= 3
    assert math.isfinite(json.loads(lines[-1])["loss/sra/span"])
    assert captured and max(captured) > 0
