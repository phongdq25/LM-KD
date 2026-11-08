"""torchrun worker for test_weight_sync_zero3.py. argv: <spec.json>."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from lmkd.core.config import recipe_from_dict
from lmkd.rollout.weight_sync import iter_student_weights
from lmkd.train.builder import build_runtime

spec_path = Path(sys.argv[1])
spec = json.loads(spec_path.read_text())
rt = build_runtime(recipe_from_dict(spec["recipe"]))
assert rt.engine.is_partitioned and rt.engine.zero_stage == 3
got = {n: t.detach().float().cpu() for n, t in iter_student_weights(rt, "full")}  # collective: all ranks iterate
from transformers import AutoModelForCausalLM

ref = AutoModelForCausalLM.from_pretrained(spec["student"], torch_dtype=torch.float32).state_dict()
ok = {k: bool(k in got and got[k].shape == v.shape and torch.allclose(got[k], v, atol=1e-6)) for k, v in ref.items()}
if rt.engine.is_main:
    (spec_path.parent / "result.json").write_text(json.dumps({"ok": ok, "extra": sorted(set(got) - set(ref))}))
rt.engine.barrier()
