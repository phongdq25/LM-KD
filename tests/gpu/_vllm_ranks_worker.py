"""torchrun worker: vllm_colocate with world_size 2 (external_launcher), ZeRO-1/2 trainer. argv: <spec.json>."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from lmkd.core.config import recipe_from_dict
from lmkd.models.logits_view import LogitsView
from lmkd.rollout.base import SamplingSpec, rollouts_to_samples
from lmkd.train.builder import build_runtime

spec_path = Path(sys.argv[1])
rt = build_runtime(recipe_from_dict(json.loads(spec_path.read_text())["recipe"]))
rank = rt.engine.rank
outs = rt.rollout.generate([[5, 6, 7 + rank]], SamplingSpec(n=3, max_new_tokens=6, seed=0))
rollouts = outs[0]
b = rt.collator(rollouts_to_samples(rollouts)).to(rt.engine.device)
m = rt.student_model
m.eval()
with torch.no_grad():
    h = m(
        input_ids=b["student.input_ids"], attention_mask=b["student.attention_mask"], output_hidden_states=True
    ).hidden_states[-1]
lp = LogitsView(h, rt.student_adapter.lm_head(m)).target_logp(b["student.labels"])
mask = b["student.labels"] != -100
diff = (lp - b["rollout.rollout_logp"]).abs()[mask].mean().item()
with torch.no_grad():
    m.lm_head.weight.zero_()
rt.state.step += 1
first = rt.rollout.generate([[5, 6, 7]], SamplingSpec(temperature=0.0, max_new_tokens=3, return_logprobs=False))[0][0]
(spec_path.parent / f"result{rank}.json").write_text(json.dumps({"diff": diff, "first": first.response_ids[0]}))
rt.engine.barrier()
