"""T25: all native RL optimizer updates go through Runtime.train_step."""

from pathlib import Path

import pytest
import torch

import lmkd
from lmkd.core.registry import REWARDS
from lmkd.rl.rewards import Verifier


@REWARDS.register("toy_rl", force=True)
class ToyReward(Verifier):
    """Alternating group rewards, independent of the sampled text."""

    class Config(Verifier.Config):
        all_zero: bool = False

    def __call__(self, prompts, responses, meta):
        return [
            {"score": 0.0 if self.cfg.all_zero else float(i % 2 == 0), "acc": float(i % 2 == 0), "feedback": ""}
            for i in range(len(responses))
        ]


def rl_recipe(base, method="grpo", all_zero=False):
    base["data"]["transforms"] = [
        {"type": "tokenize_prompt", "template": "{prompt}", "use_chat_template": False, "keep_fields": ["response"]}
    ]
    base["producer"] = {
        "type": "rollout_group",
        "n": 2,
        "sampling": {"max_new_tokens": 4},
        "verifier": {"type": "toy_rl", "all_zero": all_zero},
    }
    base["rollout"] = {"type": "hf"}
    base["algorithm"] = {"type": "rl"}
    base["step"] = {"type": "ppo_epochs", "epochs": 2, "mini_batches": 2}
    base["train"].update(batch_size=4, micro_batch_size=1, max_steps=3, log_every=1)
    base["loss"] = {"terms": [{"criterion": "ppo_clip"}]}
    if method == "rlsd":
        base["teacher"] = {
            "provider": {"type": "self", "weight_policy": "live"},
            "input": {
                "type": "privileged_prompt",
                "template": "{prompt_text} {response}",
                "fields": ["response"],
                "use_chat_template": False,
            },
        }
        base["algorithm"]["adv_transforms"] = [{"type": "rlsd_reweight"}]
    elif method == "sdpo":
        base["teacher"] = {
            "provider": {"type": "self", "weight_policy": "ema"},
            "input": {
                "type": "sdpo_reprompt",
                "reprompt_template": "{prompt} {solution} {feedback}",
                "solution_template": "{successful_previous_attempt}",
                "max_reprompt_len": 32,
            },
        }
        base["loss"] = {"terms": [{"criterion": "sdpo_distill", "params": {"k": 5}}]}
        base["algorithm"]["rollout_is"] = "token"
    elif method == "luffy":
        base["producer"] = {
            "type": "luffy_group",
            "n": 2,
            "n_prefix": 1,
            "min_prefix_ratio": 1.0,
            "max_prefix_ratio": 1.0,
            "sampling": {"max_new_tokens": 4},
            "verifier": {"type": "toy_rl", "all_zero": all_zero},
        }
        base["algorithm"]["advantage"] = {"type": "grpo_split", "norm_by_std": False}
        base["algorithm"]["adv_transforms"] = [{"type": "luffy_prefix_weight", "alpha": 1.0, "beta": 1.0}]
        base["loss"] = {"terms": [{"criterion": "luffy", "params": {"off_policy_reshape": "p_div_p_0.1"}}]}
    elif method == "luffy_sft":
        base["producer"] = {
            "type": "luffy_group",
            "n": 2,
            "n_prefix": 1,
            "min_prefix_ratio": 1.0,
            "max_prefix_ratio": 1.0,
            "sampling": {"max_new_tokens": 4},
            "verifier": {"type": "toy_rl", "all_zero": all_zero},
        }
        base["algorithm"]["advantage"] = {"type": "grpo_split", "norm_by_std": False}
        base["algorithm"]["adv_transforms"] = [{"type": "luffy_prefix_weight", "alpha": 1.0, "beta": 1.0}]
        base["loss"] = {
            "terms": [{"criterion": "luffy", "params": {"use_sft_multitask_loss": True, "sft_loss_coef": 1.0}}]
        }
    return base


@pytest.mark.parametrize(
    "method,all_zero",
    [("grpo", False), ("rlsd", False), ("sdpo", False), ("sdpo", True), ("luffy", False), ("luffy_sft", False)],
)
def test_rl_tiny(base_recipe, method, all_zero):
    rt = lmkd.build(rl_recipe(base_recipe, method, all_zero))
    rt.begin()
    try:
        for _ in range(3):
            result = rt.train_step()
            assert torch.isfinite(result.metrics["loss/total"])
            if method == "sdpo" and all_zero:
                assert result.metrics["loss/total"].item() == 0.0
        assert rt.state.step == 3
        assert "rl/reward_mean" in (Path(rt.config.train.output_dir) / "metrics.jsonl").read_text()
    finally:
        rt.end()


@pytest.mark.parametrize("params", [{"k": None}, {"k": 0, "alpha": 1.0, "full_logit": False}])
def test_sdpo_full_and_sampled(base_recipe, params):
    recipe = rl_recipe(base_recipe, "sdpo")
    recipe["loss"]["terms"][0]["params"] = params
    rt = lmkd.build(recipe)
    rt.begin()
    try:
        assert torch.isfinite(rt.train_step().metrics["loss/total"])
    finally:
        rt.end()
