"""ALM baseline: chunk-level binarized probability matching for cross-tokenizer distillation."""

from __future__ import annotations

from baselines.cross_tokenizer.alm.aligner import (
    TokenkitChunk,
    chunk_ids,
    load_pair_data,
)
from baselines.cross_tokenizer.alm.loss import ALM
from lmkd.losses.aggregator import GradMag
from lmkd.train.multitask import gradmag_weights, task_grads

__all__ = [
    "ALM",
    "TokenkitChunk",
    "chunk_ids",
    "load_pair_data",
    "GradMag",
    "gradmag_weights",
    "task_grads",
]

