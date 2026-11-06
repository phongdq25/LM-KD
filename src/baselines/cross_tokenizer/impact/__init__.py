"""IMPACT baseline: Boundary-preserving prefix increment alignment for cross-tokenizer distillation."""

from __future__ import annotations

from baselines.cross_tokenizer.impact.aligner import (
    BoundaryInterp,
    boundary_coeffs,
    boundary_table,
)
from baselines.cross_tokenizer.impact.layer_mappers import (
    ALPKD,
    BIRatio,
    First,
    Last,
    LayerMapper,
    MapperInputs,
    RailKD,
    Random,
    RatioOneToOne,
    TopKBI,
    TopKPPL,
    compute_bi_scores,
    compute_ppl_scores,
)
from baselines.cross_tokenizer.impact.loss import (
    Impact,
    vectorized_teacher_increments,
    vocab_projection,
)
from baselines.cross_tokenizer.impact.stages import ImpactCalibrate

# Aliases for convenience
TopKBiMapper = TopKBI
TopKPplMapper = TopKPPL
FirstMapper = First
LastMapper = Last
RandomMapper = Random
Ratio1to1Mapper = RatioOneToOne
BiRatioMapper = BIRatio
RailKdMapper = RailKD
AlpKdMapper = ALPKD

__all__ = [
    "Impact",
    "vocab_projection",
    "vectorized_teacher_increments",
    "BoundaryInterp",
    "boundary_coeffs",
    "boundary_table",
    "ImpactCalibrate",
    "LayerMapper",
    "MapperInputs",
    "compute_bi_scores",
    "compute_ppl_scores",
    "TopKBI",
    "TopKPPL",
    "First",
    "Last",
    "Random",
    "RatioOneToOne",
    "BIRatio",
    "RailKD",
    "ALPKD",
    "TopKBiMapper",
    "TopKPplMapper",
    "FirstMapper",
    "LastMapper",
    "RandomMapper",
    "Ratio1to1Mapper",
    "BiRatioMapper",
    "RailKdMapper",
    "AlpKdMapper",
]
