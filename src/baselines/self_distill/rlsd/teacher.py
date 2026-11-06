"""RLSD Teacher specifications and configuration schemas.

Paper: Reinforcement Learning with Self-Distillation (2024)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLSDConfig:
    """Hyperparameter specification for Reinforcement Learning Self-Distillation (RLSD)."""

    # Advantage reweighting (STGCA)
    lam: float = 0.5
    clip: float = 0.2
    warmup_steps: int = 0
    decay_steps: int = 60
    negative_only: bool = False

    # Policy gradient clipping
    clip_low: float = 0.2
    clip_high: float = 0.28
    clip_c: float | None = None
    temperature: float = 1.0

    # Self-teacher setup
    periodic_sync_steps: int = 100
    privileged_prompt_template: str = "{prompt_text}\n\nReference answer: {answer}\n\nSolve the original question."
