"""The ONLY place where labels are built (DESIGN §4.2, target-aligned convention).

``labels[t] = input_ids[t]`` if token t is supervised, else ``-100``; ``labels[0]`` is always
-100 (nothing predicts the first token). Logits are shifted exclusively by LogitsView.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

IGNORE_INDEX = -100


def make_labels(
    input_ids: Sequence[int] | torch.Tensor,
    prompt_len: int,
    *,
    valid_len: int | None = None,
) -> torch.Tensor:
    """Labels for one sequence.

    Args:
        input_ids: full sequence (prompt + response), WITHOUT padding.
        prompt_len: number of prompt tokens; they are not supervised.
        valid_len: if given, tokens at positions >= valid_len are not supervised.
    """
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    labels = ids.clone()
    labels[: max(prompt_len, 1)] = IGNORE_INDEX
    if valid_len is not None:
        labels[valid_len:] = IGNORE_INDEX
    return labels


def supervised_mask(labels: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Float mask of supervised target positions = (labels != -100) * loss_mask."""
    m = (labels != IGNORE_INDEX).float()
    if loss_mask is not None:
        m = m * loss_mask.float()
    return m


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Correct position ids for left- or mid-padded sequences."""
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.clamp(min=0)
