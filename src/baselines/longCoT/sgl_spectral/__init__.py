"""SGL-Spectral (Spectral Guided Learning) Baseline Package."""

from __future__ import annotations

from baselines.longCoT.sgl_spectral.loss import (
    PerSourceObjective,
    TransitionLoss,
    TransitionPredictor,
    WeightedNLL,
    compute_transition_hidden_index,
    pool_step_targets,
    shuffled_targets,
    transition_loss_one,
)
from baselines.longCoT.sgl_spectral.pipeline import (
    AnswerGain,
    BuildWeights,
    SGLSpectralStage,
    SpectralCapture,
)
from baselines.longCoT.sgl_spectral.spectral import (
    analytic_gradients_and_entropies,
    effective_rank,
    step_strengths,
)
from baselines.longCoT.sgl_spectral.transforms import (
    AllocateRegion,
    Provenance,
    SelectEnergyTopP,
    StepTransitions,
    TailBoost,
    boost_tail,
    build_transition_fields,
    cot_char_span,
    merge_short_steps,
    region_budget,
    region_gain_ratio,
    region_stable_step_weights,
    select_steps_by_energy,
    split_cot_char_cuts,
    step_regions,
    step_token_spans_from_cuts,
)

__all__ = [
    "SGLSpectralStage",
    "SpectralCapture",
    "AnswerGain",
    "BuildWeights",
    "SelectEnergyTopP",
    "AllocateRegion",
    "TailBoost",
    "Provenance",
    "StepTransitions",
    "WeightedNLL",
    "PerSourceObjective",
    "TransitionLoss",
    "TransitionPredictor",
    "compute_transition_hidden_index",
    "pool_step_targets",
    "shuffled_targets",
    "transition_loss_one",
    "analytic_gradients_and_entropies",
    "effective_rank",
    "step_strengths",
    "select_steps_by_energy",
    "step_regions",
    "region_stable_step_weights",
    "region_gain_ratio",
    "region_budget",
    "boost_tail",
    "cot_char_span",
    "split_cot_char_cuts",
    "merge_short_steps",
    "step_token_spans_from_cuts",
    "build_transition_fields",
]
