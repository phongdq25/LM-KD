"""Configuration schema for OPSD baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OPSDConfig:
    """Hyperparameter specification for On-Policy Self-Distillation (OPSD)."""

    beta: float = 0.0
    token_clip: float = 0.05
    clip_mode: str = "elementwise"
    temperature: float = 1.1

    # Teacher setup
    teacher_weight_policy: str = "lora_disabled"
    privileged_prompt_template: str = (
        "Problem: {problem}\n\nHere is a reference solution to this problem:\n"
        "=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

