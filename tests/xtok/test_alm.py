"""ALM plus GradMag tiny training with T11's tokenkit alignment interface."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from baselines.cross_tokenizer.alm import TokenkitChunk
from lmkd.core.config import recipe_from_dict
from lmkd.train.builder import build_runtime


@pytest.mark.parametrize("require_tokenkit", [False, True])
def test_alm_runs_gradmag(base_recipe, xtok_models, tmp_path, monkeypatch, require_tokenkit):
    if require_tokenkit:
        pytest.importorskip("tokenkit")

    def one_chunk(self, s_ids, t_ids):
        return (np.ones((1, len(s_ids), 1), dtype=bool), np.ones((1, len(t_ids), 1), dtype=bool))

    monkeypatch.setattr(TokenkitChunk, "_matrices", one_chunk)
    recipe = copy.deepcopy(base_recipe)
    recipe["models"]["student"] = {"path": xtok_models["student"], "dtype": "fp32"}
    recipe["models"]["teacher"] = {"path": xtok_models["teacher"], "dtype": "fp32"}
    recipe["teacher"] = {"provider": "hf_frozen", "input": "pretokenized"}
    recipe["data"]["transforms"] = [
        {"type": "tokenize_sft", "use_chat_template": False, "teacher": True, "teacher_prompt_text": "same"},
        {"type": "load_vocab_map", "space_mask_mode": "space+tab+newline+special"},
        {"type": "align", "aligner": {"type": "tokenkit_chunk"}},
    ]
    recipe["loss"] = {"aggregator": {"type": "gradmag"}, "terms": [{"criterion": "nll"}, {"criterion": "alm"}]}
    recipe["step"] = {"type": "multi_objective_grad", "combine": "gradmag"}
    recipe["train"].update(output_dir=str(tmp_path / "alm"), max_steps=3, batch_size=2, micro_batch_size=2, log_every=1)
    rt = build_runtime(recipe_from_dict(recipe))
    rt.begin()
    for _ in range(3):
        rt.train_step()
    rt.end()
    lines = (Path(recipe["train"]["output_dir"]) / "metrics.jsonl").read_text().splitlines()
    assert len(lines) >= 3
    for line in lines:
        metrics = json.loads(line)
        assert abs(metrics["gradmag/weight_nll"] + metrics["gradmag/weight_alm"] - 1) < 1e-5
