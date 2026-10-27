"""Poolers: (hidden [B,T,D], attention_mask [B,T]) -> [B,D]. Not normalized (losses normalize)."""
from __future__ import annotations

import math

import torch

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import POOLERS


class Pooler(Plugin):
    """Turns hidden states [B,T,D] and an attention mask into sentence vectors [B,D]."""
    def __call__(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError


@POOLERS.register("cls")
class ClsPooler(Pooler):
    """First-token (CLS) vector."""
    class Config(BaseConfig):
        pass

    def __call__(self, hidden, mask):
        return hidden[:, 0]


@POOLERS.register("mean")
class MeanPooler(Pooler):
    """Mask-weighted mean over tokens."""
    class Config(BaseConfig):
        pass

    def __call__(self, hidden, mask):
        if mask is None:
            return hidden.mean(1)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


@POOLERS.register("last")
class LastTokenPooler(Pooler):
    """Last non-pad token; works for right padding (and left padding: last column)."""

    class Config(BaseConfig):
        pass

    def __call__(self, hidden, mask):
        if mask is None:
            return hidden[:, -1]
        if bool((mask[:, -1] == 1).all()):  # left padding or no padding
            return hidden[:, -1]
        idx = mask.long().sum(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


@POOLERS.register(
    "cls_attention",
    source="embedding-distill/TALAS-M-main/trainer.py:6-25",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
)
class ClsAttentionPooler(Pooler):
    """Parameter-free CLS-query attention pooling (TALAS).

    Formula:
        cls = hidden[:, :1]
        s = torch.bmm(cls, hidden.transpose(1, 2)).squeeze(1) / sqrt(D)
        s = s + (1.0 - mask) * -1e9  (or torch.finfo(s.dtype).min / 2 for non-fp32 to prevent overflow)
        w = softmax(s, dim=-1)
        return (hidden * w.unsqueeze(-1)).sum(1)
    Source: embedding-distill/TALAS-M-main/trainer.py:6-25
    """

    class Config(BaseConfig):
        pass

    def __call__(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        D = hidden.size(-1)
        cls = hidden[:, :1]  # [B, 1, D]
        s = torch.bmm(cls, hidden.transpose(1, 2)).squeeze(1) / math.sqrt(D)  # [B, T]
        if mask is not None:
            penalty = -1e9 if s.dtype == torch.float32 else (torch.finfo(s.dtype).min / 2)
            s = s + (1.0 - mask.to(s.dtype)) * penalty
        w = torch.softmax(s, dim=-1)
        return (hidden * w.unsqueeze(-1)).sum(1)
