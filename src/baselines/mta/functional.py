"""Functional primitives for Multi-granularity Text Alignment (MTA) distillation.

This module provides low-level mathematical operations for MTA:
- Self-attention token importance weight computation
- Token-in-span incidence matrix construction and span representation aggregation
- Intra-batch relational span similarity MSE loss
- Teacher-weighted token-level cosine similarity loss
- Linguistic span filtering and extraction using spaCy (with heuristic fallback)
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

VERB_PHRASE_PATTERN = [
    {"POS": "AUX", "OP": "*"},
    {"POS": "ADV", "OP": "*"},
    {"POS": "VERB", "OP": "+"},
    {"POS": "ADV", "OP": "*"},
]


def compute_token_weights(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute self-attention importance weights across tokens for a given hidden state.

    Formula:
        Q = K = H / (std(H) + 1e-5)
        Scores = (Q @ K.T) / sqrt(D), masked on padding and diagonal
        Token_Weights = mean_queries(softmax(Scores))

    Args:
        hidden_state: [B, L, D]
        attention_mask: [B, L]
    Returns:
        [B, L] detached token weights
    """
    std = hidden_state.std(dim=-1, keepdim=True) + 1e-5
    Q = hidden_state / std
    K = hidden_state / std
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)

    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    diag_mask = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag_mask.unsqueeze(0), float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = attn_weights * mask
    denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-5)
    attn_weights = attn_weights / denom

    token_weights = attn_weights.mean(dim=1)  # [B, L]
    return token_weights.detach()


def aggregate_spans_for_model(
    hidden_states: torch.Tensor,
    layer_weights: torch.Tensor,
    attention_mask: torch.Tensor,
    offsets_mapping: torch.Tensor,
    spans_offsets: list[list[tuple[int, int]]],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Aggregate token representations into span representations using a token-in-span incidence matrix.

    Args:
        hidden_states: [B, SeqLen, D]
        layer_weights: [B, SeqLen]
        attention_mask: [B, SeqLen]
        offsets_mapping: [B, SeqLen, 2] character [start, end) offsets
        spans_offsets: list of spans per batch item, each span is (start_char, end_char)
    Returns:
        span_repr: [B, Max_Spans, D] or None
        weight_sum: [B, Max_Spans] or None
        valid_span_mask: [B, Max_Spans] (bool) or None
    """
    device = hidden_states.device
    B_size, SeqLen, _ = hidden_states.shape

    span_tensors = [
        torch.tensor(s, dtype=torch.long, device=device)
        if len(s) > 0
        else torch.empty((0, 2), dtype=torch.long, device=device)
        for s in spans_offsets
    ]
    padded_spans = pad_sequence(span_tensors, batch_first=True, padding_value=0)

    if padded_spans.numel() == 0 or padded_spans.size(1) == 0:
        return None, None, None

    max_spans = padded_spans.size(1)
    padded_span_starts = padded_spans[:, :, 0]
    padded_span_ends = padded_spans[:, :, 1]

    lengths = torch.tensor([len(s) for s in spans_offsets], device=device)
    col_indices = torch.arange(max_spans, device=device).unsqueeze(0)
    valid_span_mask = col_indices < lengths.unsqueeze(1)

    current_offsets = offsets_mapping[:, :SeqLen, :] if offsets_mapping.shape[1] != SeqLen else offsets_mapping
    offsets_start = current_offsets[..., 0].unsqueeze(2)
    offsets_end = current_offsets[..., 1].unsqueeze(2)

    span_starts_exp = padded_span_starts.unsqueeze(1)
    span_ends_exp = padded_span_ends.unsqueeze(1)

    token_in_span_map = (offsets_start + 1 >= span_starts_exp) & (offsets_end <= span_ends_exp)
    token_in_span_map = token_in_span_map & attention_mask.unsqueeze(2).bool()

    A = token_in_span_map.transpose(1, 2).to(hidden_states.dtype)

    weighted_hidden = hidden_states * layer_weights.unsqueeze(-1)
    span_sum = torch.bmm(A, weighted_hidden)
    weight_sum = torch.bmm(A, layer_weights.unsqueeze(-1)).squeeze(-1)

    span_repr = span_sum / weight_sum.unsqueeze(-1).clamp(min=1e-5)

    return span_repr, weight_sum, valid_span_mask


def compute_relational_span_loss(
    s_span_repr: torch.Tensor,
    t_span_repr: torch.Tensor,
    valid_span_mask: torch.Tensor,
    w_sum: torch.Tensor,
    projector: nn.Module | None = None,
) -> torch.Tensor:
    """Compute MSE loss between intra-batch span cosine similarity matrices.

    Args:
        s_span_repr: [B, Max_Spans, D_s]
        t_span_repr: [B, Max_Spans, D_t]
        valid_span_mask: [B, Max_Spans] (bool)
        w_sum: [B, Max_Spans]
        projector: optional linear projection module from D_s to D_t
    Returns:
        Scalar MSE loss over valid intra-batch span pairs
    """
    device = s_span_repr.device
    B_size, Max_Spans = valid_span_mask.shape

    if projector is not None:
        s_span_proj = projector(s_span_repr)
    else:
        s_span_proj = s_span_repr

    valid_s = s_span_proj[valid_span_mask]
    valid_t = t_span_repr[valid_span_mask]
    valid_w = w_sum[valid_span_mask]

    if valid_s.size(0) == 0:
        return torch.tensor(0.0, device=device)

    batch_indices = torch.arange(B_size, device=device).unsqueeze(1).expand(-1, Max_Spans)
    valid_batch_ids = batch_indices[valid_span_mask]

    S_normalized = F.normalize(valid_s.float(), p=2, dim=-1)
    T_normalized = F.normalize(valid_t.float(), p=2, dim=-1)

    S_sim_matrix = S_normalized @ S_normalized.T
    T_sim_matrix = T_normalized @ T_normalized.T

    Same_Batch_Mask = valid_batch_ids.unsqueeze(1) == valid_batch_ids.unsqueeze(0)
    Not_Self_Mask = ~torch.eye(valid_s.size(0), dtype=torch.bool, device=device)
    Final_Mask = Same_Batch_Mask & Not_Self_Mask

    if not Final_Mask.any():
        return torch.tensor(0.0, device=device)

    S_intra = torch.masked_select(S_sim_matrix, Final_Mask)
    T_intra = torch.masked_select(T_sim_matrix, Final_Mask)

    span_rel_loss = F.mse_loss(S_intra, T_intra, reduction="none")

    Pair_Weights_Matrix = valid_w.unsqueeze(1) * valid_w.unsqueeze(0)
    Valid_Pair_Weights = torch.masked_select(Pair_Weights_Matrix, Final_Mask)
    denom = Valid_Pair_Weights.sum().clamp(min=1e-5)
    return (span_rel_loss * Valid_Pair_Weights).sum() / denom


def compute_token_cosine_loss(
    s_hidden: torch.Tensor,
    t_hidden: torch.Tensor,
    projector: nn.Module | None,
    t_token_weights: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute token-level 1 - cosine similarity loss weighted by teacher token weights."""
    device = s_hidden.device
    if projector is not None:
        s_proj = projector(s_hidden)
    else:
        s_proj = s_hidden

    cos = F.cosine_similarity(s_proj.float(), t_hidden.float(), dim=-1, eps=1e-5)
    loss_per_tok = (1.0 - cos) * t_token_weights * token_mask.float()
    denom = (t_token_weights * token_mask.float()).sum().clamp(min=1e-5)
    if denom <= 1e-5:
        return torch.tensor(0.0, device=device)
    return loss_per_tok.sum() / denom


def filter_overlapping_spans(
    spans: list[tuple[int, int, Any]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Sort spans and filter nested/overlapping spans, returning (spans, words) character offsets."""
    sorted_spans = sorted(spans, key=lambda s: (s[0], -s[1]))
    filtered: list[tuple[int, int]] = []
    words: list[tuple[int, int]] = []
    if not sorted_spans:
        return filtered, words

    current_span = sorted_spans[0]
    for next_span in sorted_spans[1:]:
        _, current_end, p = current_span
        _, next_end, _ = next_span
        if next_end <= current_end:
            continue
        filtered.append((current_span[0], current_span[1]))

        n_token = len(p)
        words.extend([(p[idx - 1].idx, p[idx].idx) for idx in range(1, n_token)])
        words.append((p[n_token - 1].idx, p[n_token - 1].idx + len(p[n_token - 1])))

        current_span = next_span
    filtered.append((current_span[0], current_span[1]))

    p = current_span[2]
    n_token = len(p)
    words.extend([(p[idx - 1].idx, p[idx].idx) for idx in range(1, n_token)])
    words.append((p[n_token - 1].idx, p[n_token - 1].idx + len(p[n_token - 1])))

    return filtered, words


def _heuristic_spans_and_words(
    text: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Heuristic fallback for word and span extraction when spaCy is unavailable."""
    words: list[tuple[int, int]] = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    spans: list[tuple[int, int]] = []
    chunk_size = 3
    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]
        spans.append((chunk[0][0], chunk[-1][1]))
    return spans, words


def extract_spans_and_words(
    texts: list[str],
    nlp: Any = None,
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """Extract linguistic phrase spans and word spans from texts.

    Uses spaCy verb-phrase matcher and noun chunks if nlp is available;
    falls back to heuristic segmentation otherwise.
    """
    if nlp is None:
        try:
            import spacy
            from spacy.matcher import Matcher

            nlp = spacy.load("en_core_web_sm")
            matcher = Matcher(nlp.vocab)
            matcher.add("VERB_PHRASE", [VERB_PHRASE_PATTERN])
            nlp._mta_matcher = matcher
        except Exception:
            nlp = None

    if nlp is None:
        all_spans: list[list[tuple[int, int]]] = []
        all_words: list[list[tuple[int, int]]] = []
        for text in texts:
            s, w = _heuristic_spans_and_words(text)
            all_spans.append(s)
            all_words.append(w)
        return all_spans, all_words

    nlp_matcher: Any = getattr(nlp, "_mta_matcher", None)
    if nlp_matcher is None:
        from spacy.matcher import Matcher

        nlp_matcher = Matcher(nlp.vocab)
        nlp_matcher.add("VERB_PHRASE", [VERB_PHRASE_PATTERN])
        nlp._mta_matcher = nlp_matcher

    all_spans = []
    all_words = []
    disabled = ["ner", "lemmatizer"]
    docs = nlp.pipe(texts, disable=[d for d in disabled if d in nlp.pipe_names])
    for doc in docs:
        spans_with_offsets = []
        vps = nlp_matcher(doc)
        for _, start, end in vps:
            vp = doc[start:end]
            spans_with_offsets.append((vp.start_char, vp.end_char, vp))
        ncs = doc.noun_chunks
        spans_with_offsets.extend([(nc.start_char, nc.end_char, nc) for nc in ncs])

        unique_spans, unique_words = filter_overlapping_spans(spans_with_offsets)
        all_spans.append(unique_spans)
        all_words.append(unique_words)

    return all_spans, all_words

