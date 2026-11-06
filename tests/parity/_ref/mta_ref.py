"""Reference implementation of MTA from white-box-distillation/same-tokenizer/finetune.py."""
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


def ref_compute_token_weights(hidden_state, attention_mask):
    std = hidden_state.std(dim=-1, keepdim=True) + 1e-5
    Q = hidden_state / std
    K = hidden_state / std
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)

    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    diag_mask = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = attn_weights * mask
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

    token_weights = attn_weights.mean(dim=1)
    return token_weights.detach()


def ref_aggregate_spans_for_model(hidden_states, layer_weights, attention_mask, offsets_mapping, spans_offsets):
    device = hidden_states.device
    B_size, SeqLen, D = hidden_states.shape

    span_tensors = [
        torch.tensor(s, dtype=torch.long, device=device) if len(s) > 0
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


def ref_compute_hidden_span_loss(projector, s_span_repr, t_span_repr, valid_span_mask, w_sum):
    device = s_span_repr.device
    B_size, Max_Spans = valid_span_mask.shape

    s_span_proj = projector(s_span_repr) if projector is not None else s_span_repr

    valid_s = s_span_proj[valid_span_mask]
    valid_t = t_span_repr[valid_span_mask]
    valid_w = w_sum[valid_span_mask]

    if valid_s.size(0) == 0:
        return torch.tensor(0.0, device=device)

    batch_indices = torch.arange(B_size, device=device).unsqueeze(1).expand(-1, Max_Spans)
    valid_batch_ids = batch_indices[valid_span_mask]

    S_normalized = F.normalize(valid_s, p=2, dim=-1)
    T_normalized = F.normalize(valid_t, p=2, dim=-1)

    S_sim_matrix = S_normalized @ S_normalized.T
    T_sim_matrix = T_normalized @ T_normalized.T

    Same_Batch_Mask = (valid_batch_ids.unsqueeze(1) == valid_batch_ids.unsqueeze(0))
    Not_Self_Mask = ~torch.eye(valid_s.size(0), dtype=torch.bool, device=device)
    Final_Mask = Same_Batch_Mask & Not_Self_Mask

    S_intra_batch_similarities_flat = torch.masked_select(S_sim_matrix, Final_Mask)
    T_intra_batch_similarities_flat = torch.masked_select(T_sim_matrix, Final_Mask)

    span_rel_loss = F.mse_loss(S_intra_batch_similarities_flat,
                               T_intra_batch_similarities_flat, reduction='none')

    Pair_Weights_Matrix = valid_w.unsqueeze(1) * valid_w.unsqueeze(0)
    Valid_Pair_Weights = torch.masked_select(Pair_Weights_Matrix, Final_Mask)
    span_rel_loss = (span_rel_loss * Valid_Pair_Weights).sum() / Valid_Pair_Weights.sum().clamp(min=1e-5)

    return span_rel_loss

