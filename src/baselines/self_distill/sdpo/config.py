"""Configuration schemas for SDPO baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SDPOConfig:
    """Hyperparameter specification for Self-Distillation with Preference Optimization (SDPO)."""

    alpha: float = 0.5
    k: int = 100
    add_tail: bool = True
    is_clip: float = 2.0
    full_logit: bool = True
    temperature: float = 1.0

    # Reprompt configuration
    reprompt_template: str = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question."
    solution_template: str = "\nCorrect solution:\n{successful_previous_attempt}"
    feedback_template: str = "\nThe following is feedback from your unsuccessful earlier attempt:\n{feedback_raw}"
    success_threshold: float = 0.5
    dont_reprompt_on_self_success: bool = True
    include_feedback: bool = True
    max_reprompt_len: int = 10240
    remove_thinking_from_demonstration: bool = True
    feedback_only_without_solution: bool = True

    # Teacher weight policy
    weight_policy: str = "ema"
    ema_rate: float = 0.05
