"""T25: verifier reuse through TRL and native PPO parity with actual bnpo compute_loss."""

import os
import subprocess
import sys

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from lmkd.core.batch import Batch
from lmkd.core.state import TrainState
from lmkd.losses.base import LossContext
from lmkd.losses.rl import PPOClip
from lmkd.models.forward import ForwardRunner, SideRequest, side_inputs
from lmkd.rl.rewards import MCQ, as_trl_reward


def _grpo_batch(inputs):
    """TRL GRPO ``inputs`` -> target-aligned Batch; only completion tokens are supervised."""
    p_ids, p_mask, c_ids, c_mask = (
        inputs[k] for k in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask")
    )
    ids = torch.cat([p_ids, c_ids], 1)
    supervised = torch.cat([torch.zeros_like(p_mask), c_mask], 1).bool()
    labels = ids.masked_fill(~supervised, -100)
    labels[:, 0] = -100
    return Batch(
        {
            "student.input_ids": ids,
            "student.attention_mask": torch.cat([p_mask, c_mask], 1),
            "student.labels": labels,
            "student.prompt_len": torch.full((ids.size(0),), p_ids.size(1), dtype=torch.long),
        }
    )


def test_ppo_clip_matches_trl_bnpo(tiny_models, tmp_path):
    # Earlier HF/DeepSpeed tests initialize Accelerate's singleton with bf16. Use a fresh process
    # for fp32 parity, without depending on private state-reset APIs or changing other tests.
    script = (
        "import sys; from pathlib import Path; "
        "from tests.unit.test_rl_trl import _ppo_clip_matches_trl_bnpo; "
        "_ppo_clip_matches_trl_bnpo({'student':sys.argv[1]}, Path(sys.argv[2]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, tiny_models["student"], str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "ACCELERATE_MIXED_PRECISION": "no"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _ppo_clip_matches_trl_bnpo(tiny_models, tmp_path):
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    model = AutoModelForCausalLM.from_pretrained(tiny_models["student"])
    cfg = GRPOConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        per_device_train_batch_size=4,
        num_generations=2,
        beta=0.0,
        loss_type="bnpo",
        epsilon=0.2,
        epsilon_high=0.28,
        temperature=1.0,
        delta=None,
        report_to="none",
        max_completion_length=3,
        bf16=False,
        fp16=False,
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        processing_class=tok,
        reward_funcs=[as_trl_reward(MCQ())],
        train_dataset=Dataset.from_list([{"prompt": "Q:", "answer": "A"}] * 4),
    )
    model.eval()  # Public compute_loss in eval mode uses a single token denominator, no accumulation state.
    inputs = {
        "prompt_ids": torch.tensor([[3, 4], [3, 5], [4, 5], [5, 6]]),
        "prompt_mask": torch.ones(4, 2, dtype=torch.long),
        "completion_ids": torch.tensor([[6, 7, 1], [7, 1, 0], [8, 9, 1], [8, 1, 0]]),
        "completion_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0]]),
        "advantages": torch.tensor([1.0, -1.0, 0.7, -0.7]),
    }
    batch = _grpo_batch(inputs)
    from lmkd.models.adapters import ADAPTERS

    adapter = ADAPTERS.build("hf_auto")
    so = ForwardRunner.run(model, adapter, side_inputs(batch, "student"), SideRequest(logits="view"), grad=True)[0]
    logp = so.logits.target_logp(batch["student.labels"])
    torch.manual_seed(25)
    old = logp.detach()[:, -3:] + torch.randn(4, 3) * 0.5
    inputs["old_per_token_logps"] = old
    batch["rollout.old_logp"] = torch.cat([torch.zeros(4, 2), old], 1)
    batch["rollout.advantage"] = inputs["advantages"][:, None].expand(4, 5)
    ctx = LossContext(batch, so, [so], None, None, TrainState(), None)
    actual = PPOClip()(ctx).terms["pg"].value / inputs["completion_mask"].sum()
    expected = trainer.compute_loss(model, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    params = [p for p in model.parameters() if p.requires_grad]
    actual_grads = torch.autograd.grad(actual, params)
    expected_grads = torch.autograd.grad(expected, params)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_as_trl_reward_trains_grpo(tiny_models, tmp_path):
    tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    model = AutoModelForCausalLM.from_pretrained(tiny_models["student"])
    cfg = GRPOConfig(
        output_dir=str(tmp_path / "trl"),
        use_cpu=True,
        max_steps=1,
        per_device_train_batch_size=4,
        num_generations=2,
        beta=0.0,
        max_completion_length=3,
        report_to="none",
        save_strategy="no",
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        processing_class=tok,
        reward_funcs=[as_trl_reward(MCQ())],
        train_dataset=Dataset.from_list([{"prompt": "Q: w0", "answer": "A"}] * 8),
    )
    assert trainer.train().global_step == 1
