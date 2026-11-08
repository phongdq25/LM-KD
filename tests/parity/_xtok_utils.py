"""Helper utilities for cross-tokenizer parity tests."""

from __future__ import annotations

import torch


def pos_to_tgt_labels(labels_pos: torch.Tensor, pad_val: int = -100) -> torch.Tensor:
    """Prepend a dummy target position 0 (labeled -100) along sequence dimension T."""
    b = labels_pos.shape[0]
    dummy = torch.full((b, 1), pad_val, dtype=labels_pos.dtype, device=labels_pos.device)
    return torch.cat([dummy, labels_pos], dim=1)


def pad_pos(x: torch.Tensor) -> torch.Tensor:
    """Append one dummy step at the end so target index t (using pos t-1) maps to original pos t-1."""
    dummy = torch.zeros_like(x[:, :1])
    return torch.cat([x, dummy], dim=1)
