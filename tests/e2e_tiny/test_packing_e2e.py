"""Tiny packed SFT and FlashAttention sequence isolation checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lmkd.core.config import recipe_from_dict
from lmkd.data.collator import Collator
from lmkd.data.labels import make_labels, supervised_mask
from lmkd.data.packing import pack_samples
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import ForwardRunner, SideRequest, side_inputs
from lmkd.train.trainer import train


@pytest.mark.gpu
def test_packed_loss_matches_unpacked_flash():
    pytest.importorskip("flash_attn")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FlashAttention-2")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "flash_attention_2"
    model = Qwen2ForCausalLM(config).cuda().half().eval()
    adapter = HFAutoAdapter(HFAutoAdapter.Config())
    samples = []
    for ids in ([5, 6, 7, 8, 9], [10, 11, 12, 13]):
        samples.append(
            {
                "student.input_ids": list(ids),
                "student.attention_mask": [1] * len(ids),
                "student.labels": make_labels(ids, 1).tolist(),
            }
        )
    collator = Collator({"student": 0}, pad_to_multiple_of=None)
    unpacked = collator(samples).to("cuda")
    packed = collator([pack_samples(samples, max_tokens=16)]).to("cuda")

    def total_nll(batch):
        output = ForwardRunner.run(
            model, adapter, side_inputs(batch, "student"), SideRequest(logits="view"), grad=False
        )[0]
        labels = batch["student.labels"]
        return -(output.logits.target_logp(labels) * supervised_mask(labels)).sum()

    with torch.no_grad():
        assert torch.allclose(total_nll(packed).float(), total_nll(unpacked).float(), rtol=1e-3, atol=1e-3)


def test_packed_runs_cpu_unsafe(base_recipe):
    base_recipe["data"]["packing"] = True
    base_recipe["compat"] = {"allow_unsafe_packing": True}
    base_recipe["producer"] = {"type": "packed_dataset", "max_tokens": 64}
    base_recipe["train"].update(batch_size=2, micro_batch_size=1, max_steps=3, log_every=1)
    cfg = recipe_from_dict(base_recipe)
    rt = train(cfg)
    assert rt.state.step == 3
    assert rt.state.samples_seen > rt.state.step
    metrics = [json.loads(line) for line in (Path(cfg.train.output_dir) / "metrics.jsonl").read_text().splitlines()]
    assert metrics and "pack/efficiency" in metrics[-1]
    assert torch.isfinite(torch.tensor(metrics[-1]["loss/total"]))
