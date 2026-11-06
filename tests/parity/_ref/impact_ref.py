"""Verbatim functions from white-box-distillation/cross-tokenizer/IMPACT/code/impact_utils.py:107-896.

Only log_rank import/calls remain inside their original guarded branches; parity tests keep log_selection=False.
"""
import math
import random
import torch
import torch.nn.functional as F


def _normalize_tokenizer_vocab(vocab):
    """Unify GPT-2 (Ġ) and SentencePiece (▁) space markers for overlap counting."""
    return {k.replace("Ġ", "▁"): v for k, v in vocab.items()}


def compute_vocab_projection(teacher_model, student_model, teacher_tokenizer,
                             student_tokenizer, lambda_reg=1.0, max_overlap=None):
    """Solve for W_{T->S} via ridge regression on overlapping vocabulary embeddings.

    W = (E_T^T E_T + lambda * I)^{-1} E_T^T E_S
    Returns W of shape [d_T, d_S].

    If max_overlap is set and smaller than |V_overlap|, keep only the
    max_overlap overlapping tokens with the lowest student token ids
    (BPE frequency proxy: lower id ≈ more frequent).
    """
    stu_norm = _normalize_tokenizer_vocab(student_tokenizer.get_vocab())
    tea_norm = _normalize_tokenizer_vocab(teacher_tokenizer.get_vocab())
    overlap = sorted(set(stu_norm.keys()) & set(tea_norm.keys()))

    if len(overlap) == 0:
        d_T = teacher_model.config.hidden_size
        d_S = student_model.config.hidden_size
        return torch.zeros(d_T, d_S)

    n_full = len(overlap)
    if max_overlap is not None and int(max_overlap) > 0 and n_full > int(max_overlap):
        # Rank by student token id (lower ≈ more frequent in BPE/Unigram vocabs).
        overlap = sorted(overlap, key=lambda t: stu_norm[t])[: int(max_overlap)]
        try:
            from utils import log_rank
            log_rank(
                "[IMPACT vocab projection] restricting overlap "
                "{} -> {} (lowest student token ids)".format(n_full, len(overlap))
            )
        except ImportError:
            pass

    stu_ids = [stu_norm[t] for t in overlap]
    tea_ids = [tea_norm[t] for t in overlap]

    stu_emb = student_model.get_input_embeddings()
    tea_emb = teacher_model.get_input_embeddings()

    E_S = stu_emb.weight.data[stu_ids].float()  # [V_overlap, d_S]
    E_T = tea_emb.weight.data[tea_ids].float()  # [V_overlap, d_T]

    d_T = E_T.size(1)
    A = E_T.T @ E_T + lambda_reg * torch.eye(d_T, device=E_T.device)
    B = E_T.T @ E_S
    W = torch.linalg.solve(A, B)  # [d_T, d_S]

    return W.detach()


def compute_bi_scores(hidden_states):
    """Compute Block Influence scores for each layer.

    BI(l) = 1 - mean_tokens(cosine_similarity(h^(l), h^(l-1)))
    Higher score = more influential layer.

    Args:
        hidden_states: list of [B, L, d] tensors (L_T+1 entries: embedding + L_T layers)

    Returns:
        bi_scores: tensor of shape [num_layers] (excludes embedding layer)
    """
    scores = []
    for l in range(1, len(hidden_states)):
        h_prev = hidden_states[l - 1].float()
        h_curr = hidden_states[l].float()
        cos = F.cosine_similarity(h_prev, h_curr, dim=-1, eps=1e-6)
        bi = 1.0 - cos.mean()
        scores.append(bi)
    return torch.stack(scores)


def select_top_k_layers(layer_scores, K):
    """Select top-K teacher layers by score. Returns sorted 0-indexed layer indices."""
    K = min(K, len(layer_scores))
    _, indices = layer_scores.topk(K)
    return indices.sort().values


def select_first_k_layers(K, L_T):
    """First K transformer layers (0-indexed: 0 .. K-1)."""
    K = min(K, L_T)
    return torch.arange(K, dtype=torch.long)


def select_last_k_layers(K, L_T):
    """Last K transformer layers (0-indexed: L_T-K .. L_T-1)."""
    K = min(K, L_T)
    return torch.arange(L_T - K, L_T, dtype=torch.long)


def select_random_k_layers(K, L_T, seed=None):
    """K distinct random layers; sorted 0-indexed indices."""
    K = min(K, L_T)
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(L_T), K))
    return torch.tensor(picks, dtype=torch.long)


def compute_ppl_layer_scores(teacher_hidden_states, teacher_model, labels, padding_id=-100):
    """Per-layer NLL via teacher lm_head (PPL ablation for layer selection)."""
    lm_head = teacher_model.get_output_embeddings()
    if lm_head is None:
        return torch.zeros(len(teacher_hidden_states) - 1)

    scores = []
    for l in range(1, len(teacher_hidden_states)):
        h = teacher_hidden_states[l]
        logits = lm_head(h)
        pad_mask = labels.ne(padding_id)
        target = labels.unsqueeze(-1)
        target = torch.where(target.eq(-100), torch.zeros_like(target), target)
        logits = logits.masked_fill_(logits.isnan() | logits.isinf(), 0.0)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        nll = -lprobs.gather(-1, target).squeeze(-1)
        layer_nll = (nll * pad_mask.float()).sum() / pad_mask.float().sum().clamp(min=1)
        scores.append(layer_nll)

    return torch.stack(scores)


def select_teacher_layers_and_weights(
    choose_layer,
    teacher_hidden_states,
    K,
    L_T,
    bi_tau=1.0,
    teacher_model=None,
    teacher_labels=None,
    padding_id=-100,
    random_seed=None,
    precomputed_layer_scores=None,
):
    """Pick teacher layer indices and aggregation weights alpha.

    choose_layer: BI | random | first | last | PPL
    - BI / PPL: top-K by score, alpha = softmax(score/tau) on selected layers
    - random / first / last: fixed index sets, alpha = uniform 1/K

    If ``precomputed_layer_scores`` is set (offline BI/PPL calibration), reuse it
    instead of recomputing from the current batch.
    """
    K = min(K, L_T)
    mode = (choose_layer or "BI").upper()

    if mode == "FIRST":
        selected = select_first_k_layers(K, L_T)
        layer_scores = None
    elif mode == "LAST":
        selected = select_last_k_layers(K, L_T)
        layer_scores = None
    elif mode == "RANDOM":
        selected = select_random_k_layers(K, L_T, seed=random_seed)
        layer_scores = None
    elif mode == "PPL":
        if precomputed_layer_scores is not None:
            layer_scores = precomputed_layer_scores
        else:
            if teacher_model is None or teacher_labels is None:
                raise ValueError("PPL layer selection requires teacher_model and teacher_labels")
            layer_scores = compute_ppl_layer_scores(
                teacher_hidden_states, teacher_model, teacher_labels, padding_id=padding_id
            )
        selected = select_top_k_layers(layer_scores, K)
    else:
        if precomputed_layer_scores is not None:
            layer_scores = precomputed_layer_scores
        else:
            layer_scores = compute_bi_scores(teacher_hidden_states)
        selected = select_top_k_layers(layer_scores, K)
        mode = "BI"

    device = teacher_hidden_states[1].device
    selected = selected.to(device)

    if layer_scores is not None and mode in ("BI", "PPL"):
        alpha = compute_layer_weights(layer_scores.to(device), selected, tau=bi_tau)
    else:
        alpha = torch.ones(len(selected), device=device) / len(selected)

    return selected, alpha, mode, layer_scores


def compute_layer_weights(layer_scores, selected_indices, tau=1.0):
    """Compute softmax-normalized layer weights from BI scores.

    alpha_l = exp(BI(l)/tau) / sum_r exp(BI(r)/tau)
    """
    selected_scores = layer_scores[selected_indices]
    return F.softmax(selected_scores / tau, dim=0)


def map_teacher_to_student_layer(teacher_layer_idx, L_T, L_S):
    """Map teacher layer to student layer via relative depth: g(l) = ceil(l * L_S / L_T).

    teacher_layer_idx is 1-indexed (layer 1 to L_T).
    Returns 1-indexed student layer.
    """
    return math.ceil(teacher_layer_idx * L_S / L_T)


def map_student_to_teacher_layer(student_layer_idx, L_T, L_S):
    """Map student layer to teacher layer via relative depth: l = ceil(m * L_T / L_S).

    student_layer_idx is 1-indexed (layer 1 to L_S).
    Returns 1-indexed teacher layer.
    """
    return math.ceil(student_layer_idx * L_T / L_S)


def normalize_choose_align(choose_align):
    """Normalize CLI value to BI | 1-1 | 1-all | 1-random."""
    s = (choose_align or "BI").strip()
    upper = s.upper()
    if upper in ("1-1", "1_1", "11", "DEPTH", "DEPTH-RATIO", "DEPTH_RATIO"):
        return "1-1"
    if upper in ("1-ALL", "1_ALL", "ALP", "ALP-KD", "ALPKD"):
        return "1-all"
    if upper in ("1-RANDOM", "1_RANDOM", "RAIL", "RAIL-KD", "RAIL_KD", "RAILKD"):
        return "1-random"
    return "BI"


def sample_rail_kd_teacher_layers(
    L_T,
    L_S,
    seed=None,
    no_random=False,
    has_embed=False,
    has_final=False,
):
    """GLMKD ``RAIL_KD.get_teacher_hook`` teacher layer indices (HF hidden_states index).

    Matches ``distill_model.RAIL_KD``:
      - ``random.sample(range(1, teacher_num_layers), num_layers - 1)`` then ``sort()``
      - ``teacher_num_layers`` = # teacher transformer blocks (= ``L_T`` here)
      - pool is ``{1, ..., L_T - 1}`` (upper bound exclusive, same as GLMKD)
      - count = ``L_S - 1`` (not ``L_S``); optional embed inserts ``0``, optional final adds output hook

    Returns sorted teacher indices used for intermediate ``layernorm_output`` hooks.
    """
    if L_T < 1 or L_S < 1:
        return [1]

    teacher_num_layers = L_T
    num_layers = L_S
    n_sample = max(num_layers - 1, 1)

    if no_random:
        layers_per_block = max(int(teacher_num_layers / max(num_layers, 1)), 1)
        layers = list(range(0, teacher_num_layers + 1, layers_per_block))[1:-1]
        layers = [int(i) for i in layers if 0 < i < teacher_num_layers]
        if len(layers) < n_sample:
            pad = layers[-1] if layers else 1
            layers = layers + [pad] * (n_sample - len(layers))
        layers = layers[:n_sample]
    else:
        pool = list(range(1, teacher_num_layers))
        if not pool:
            pool = [1]
        rng = random.Random(seed)
        k = min(n_sample, len(pool))
        layers = sorted(rng.sample(pool, k))

    if has_embed:
        layers = [0] + layers
    if has_final:
        layers = layers + [teacher_num_layers]
    return layers


def rail_kd_num_pairs(L_S, has_final=False):
    """Number of (student, teacher) reps zipped in GLMKD ``inter_loss`` (layer-wise mode)."""
    if has_final:
        return L_S
    return max(L_S - 1, 1)


def build_rail_kd_alignment_pairs(
    L_T,
    L_S,
    seed=None,
    no_random=False,
    has_embed=False,
    has_final=False,
):
    """Build BPIA pairs for ``choose_align=1-random`` using GLMKD RAIL-KD layer matching.

    GLMKD pairs the first ``L_S - 1`` student ``layernorm_output`` hooks (student layers
    ``1 .. L_S-1`` in HF ``hidden_states``) with the sorted random teacher hooks — not all
    ``L_S`` student blocks unless ``rail_kd_has_final`` is enabled.
    """
    teacher_layers = sample_rail_kd_teacher_layers(
        L_T,
        L_S,
        seed=seed,
        no_random=no_random,
        has_embed=has_embed,
        has_final=has_final,
    )
    n_pairs = rail_kd_num_pairs(L_S, has_final=has_final)
    t_start = 1 if has_embed else 0
    weight = 1.0 / max(n_pairs, 1)
    pairs = []
    for i in range(n_pairs):
        student_m = i + 1
        teacher_l = teacher_layers[t_start + i]
        pairs.append((teacher_l, student_m, weight))
    return pairs, "1-random", "RAIL_KD"


def compute_alp_kd_attention_weights(student_hidden_states, teacher_hidden_states, W_proj, L_T, L_S):
    """ALP-KD layer attention (GLMKD ``ALP_KD.inter_loss``).

    For each student layer, softmax over all teacher layers using dot-product
    scores between the first-token (CLS) hidden states. Teacher CLS is projected
    to student dim via ``W_proj`` (cross-tokenizer), matching GLMKD when hidden
    sizes differ.

    Returns:
        alpha: [B, L_S, L_T] with sum over L_T = 1 per student layer.
    """
    device = student_hidden_states[1].device
    W = W_proj.to(device=device, dtype=torch.float32)
    s_cls = torch.stack(
        [student_hidden_states[m][:, 0, :].float() for m in range(1, L_S + 1)],
        dim=1,
    )
    t_cls = torch.stack(
        [(teacher_hidden_states[l][:, 0, :].float() @ W) for l in range(1, L_T + 1)],
        dim=1,
    )
    scores = torch.bmm(s_cls, t_cls.transpose(1, 2))
    return F.softmax(scores, dim=-1)


def fuse_teacher_hidden_by_attention(teacher_hidden_states, alpha_bt, L_T):
    """Fused teacher hidden H^C = sum_l alpha_l * H_T^l (ALP-KD Eq. 5–6 style).

    Args:
        alpha_bt: [B, L_T] weights for one student layer.
    """
    fused = None
    for l in range(1, L_T + 1):
        h = teacher_hidden_states[l]
        w = alpha_bt[:, l - 1].view(-1, 1, 1).to(device=h.device, dtype=h.dtype)
        fused = h * w if fused is None else fused + h * w
    return fused


def build_impact_alignment_pairs(
    choose_align,
    choose_layer,
    teacher_hidden_states,
    top_k,
    L_T,
    L_S,
    bi_tau=1.0,
    teacher_model=None,
    teacher_labels=None,
    padding_id=-100,
    random_seed=None,
    rail_teacher_layers=None,
    rail_kd_no_random=False,
    rail_kd_has_embed=False,
    rail_kd_has_final=False,
    precomputed_layer_scores=None,
):
    """Build (teacher_layer, student_layer, weight) triples for BPIA aggregation.

    choose_align:
      - BI: top-K teacher layers (choose_layer), BI/PPL softmax weights, g(l) depth map
      - 1-1: every student layer m paired with teacher ceil(m*L_T/L_S), uniform 1/L_S
      - 1-all: not used here — handled in ``compute_impact_loss`` via ALP-KD attention
      - 1-random: RAIL_KD (GLMKD) sorted random teacher layers, 1:1 with student depth
    """
    align_mode = normalize_choose_align(choose_align)
    pairs = []

    if align_mode == "1-random":
        if rail_teacher_layers is not None:
            n_pairs = rail_kd_num_pairs(L_S, has_final=rail_kd_has_final)
            t_start = 1 if rail_kd_has_embed else 0
            weight = 1.0 / max(n_pairs, 1)
            layers = list(rail_teacher_layers)
            pairs = [
                (layers[t_start + i], i + 1, weight) for i in range(n_pairs)
            ]
            return pairs, align_mode, "RAIL_KD"
        return build_rail_kd_alignment_pairs(
            L_T,
            L_S,
            seed=random_seed,
            no_random=rail_kd_no_random,
            has_embed=rail_kd_has_embed,
            has_final=rail_kd_has_final,
        )

    if align_mode == "1-1":
        weight = 1.0 / max(L_S, 1)
        for m in range(1, L_S + 1):
            l = map_student_to_teacher_layer(m, L_T, L_S)
            pairs.append((l, m, weight))
        return pairs, align_mode, None

    if align_mode == "1-all":
        raise RuntimeError(
            "choose_align=1-all uses ALP-KD fused teacher states in compute_impact_loss; "
            "do not call build_impact_alignment_pairs for 1-all."
        )

    selected, alpha, layer_mode, _ = select_teacher_layers_and_weights(
        choose_layer,
        teacher_hidden_states,
        top_k,
        L_T,
        bi_tau=bi_tau,
        teacher_model=teacher_model,
        teacher_labels=teacher_labels,
        padding_id=padding_id,
        random_seed=random_seed,
        precomputed_layer_scores=precomputed_layer_scores,
    )
    for sel_idx, w in zip(selected, alpha):
        l = sel_idx.item() + 1
        m = map_teacher_to_student_layer(l, L_T, L_S)
        pairs.append((l, m, w.item() if torch.is_tensor(w) else float(w)))
    return pairs, align_mode, layer_mode


def _get_teacher_boundary_state(c, teacher_offsets, teacher_prefix_states, mode="linear"):
    """Get the virtual teacher state at text boundary c.

    teacher_offsets: list of (a_j, b_j) for each teacher token (0-indexed)
    teacher_prefix_states: [L_T+1, d] where index 0 is the initial state (zeros),
                           and index j+1 is the state after processing token j
    mode:
      - linear: interpolate between u_{j-1} and u_j by character offset (default)
      - nearest: snap to closer of the two token boundaries
      - containing: use state after consuming the teacher token that contains c
    """
    num_tokens = len(teacher_offsets)
    if num_tokens == 0:
        return teacher_prefix_states[0]

    if c <= teacher_offsets[0][0]:
        return teacher_prefix_states[0]
    if c >= teacher_offsets[-1][1]:
        return teacher_prefix_states[num_tokens]

    mode = (mode or "linear").lower()
    for j in range(num_tokens):
        a_j, b_j = teacher_offsets[j]
        if c == a_j:
            return teacher_prefix_states[j]
        if c == b_j:
            return teacher_prefix_states[j + 1]
        if a_j < c < b_j:
            if mode == "containing":
                return teacher_prefix_states[j + 1]
            if mode == "nearest":
                if (c - a_j) <= (b_j - c):
                    return teacher_prefix_states[j]
                return teacher_prefix_states[j + 1]
            # linear (default)
            span_len = b_j - a_j
            if span_len == 0:
                return teacher_prefix_states[j]
            alpha = (c - a_j) / span_len
            return (1.0 - alpha) * teacher_prefix_states[j] + alpha * teacher_prefix_states[j + 1]

    return teacher_prefix_states[num_tokens]


def compute_teacher_increments_batch(student_offsets_batch, teacher_offsets_batch,
                                     teacher_hidden, W_proj, student_mask, teacher_mask,
                                     boundary_mode="linear"):
    """Teacher increments in student space; shape [B, S_len, d_S]."""
    B, S_len = student_mask.shape
    device = teacher_hidden.device
    dtype = teacher_hidden.dtype

    t_projected = teacher_hidden.float() @ W_proj.to(device).float()  # [B, T_len, d_S]
    d_S = t_projected.size(-1)

    zero_state = torch.zeros(1, d_S, device=device, dtype=torch.float32)
    result = torch.zeros(B, S_len, d_S, device=device, dtype=torch.float32)

    for b in range(B):
        s_offsets = student_offsets_batch[b]
        t_offsets = teacher_offsets_batch[b]
        t_len = int(teacher_mask[b].sum().item())
        s_len = int(student_mask[b].sum().item())

        t_states = t_projected[b, :t_len]  # [t_len, d_S]
        t_prefix = torch.cat([zero_state, t_states], dim=0)  # [t_len+1, d_S]

        t_off = t_offsets[:t_len] if len(t_offsets) >= t_len else t_offsets

        num_s = min(s_len, len(s_offsets))
        for i in range(num_s):
            a_i, b_i = s_offsets[i]
            state_at_b = _get_teacher_boundary_state(
                b_i, t_off, t_prefix, mode=boundary_mode
            )
            state_at_a = _get_teacher_boundary_state(
                a_i, t_off, t_prefix, mode=boundary_mode
            )
            result[b, i] = state_at_b - state_at_a

    return result.to(dtype)


def compute_student_increments(student_hidden, student_mask):
    """Student prefix increments Δ_S[i] = h_S[i] - h_S[i-1]; shape [B, S_len, d_S]."""
    B, S_len, d_S = student_hidden.shape
    h_prev = torch.zeros(B, 1, d_S, device=student_hidden.device, dtype=student_hidden.dtype)
    h_shifted = torch.cat([h_prev, student_hidden[:, :-1]], dim=1)
    increments = student_hidden - h_shifted
    return increments


def _log_alp_kd_alignment(alpha, L_S, L_T):
    """Log ALP-KD attention (sample student layers, batch 0)."""
    try:
        from utils import log_rank
    except ImportError:
        return
    b0 = 0
    samples = sorted({1, max(1, L_S // 2), L_S})
    parts = []
    for m in samples:
        w = alpha[b0, m - 1].detach().float().cpu().tolist()
        ranked = sorted(enumerate(w, start=1), key=lambda x: -x[1])
        top5 = [(l, round(a, 4)) for l, a in ranked[:5]]
        parts.append(f"s{m}:{top5}")
    log_rank(
        "[IMPACT align] mode=1-all (ALP-KD, GLMKD ALP_KD) L_S={} L_T={} num_student_layers={} "
        "attn_softmax_over_teacher batch0_samples: {}".format(
            L_S, L_T, L_S, "; ".join(parts)
        )
    )


def compute_impact_loss(student_hidden_states, teacher_hidden_states,
                       student_offsets_batch, teacher_offsets_batch,
                       W_proj, student_mask, teacher_mask,
                       top_k=4, bi_tau=1.0, L_T=None, L_S=None,
                       choose_layer="BI", choose_align="BI",
                       teacher_model=None, teacher_labels=None,
                       padding_id=-100, random_seed=None, log_selection=False,
                       rail_teacher_layers=None, rail_kd_no_random=False,
                       rail_kd_has_embed=False, rail_kd_has_final=False,
                       boundary_mode="linear",
                       precomputed_layer_scores=None):
    """IMPACT / BPIA loss: weighted mean (1 - cos) over aligned prefix increments."""
    if L_T is None:
        L_T = len(teacher_hidden_states) - 1
    if L_S is None:
        L_S = len(student_hidden_states) - 1

    align_mode = normalize_choose_align(choose_align)
    if align_mode == "1-all":
        if W_proj is None:
            raise ValueError("ALP-KD (choose_align=1-all) requires vocabulary projection W_proj")
        alpha = compute_alp_kd_attention_weights(
            student_hidden_states, teacher_hidden_states, W_proj, L_T, L_S
        )
        if log_selection:
            _log_alp_kd_alignment(alpha, L_S, L_T)

        device = student_hidden_states[0].device
        total_loss = torch.tensor(0.0, device=device)
        mask_float = student_mask.float()
        N = mask_float.sum(dim=1).clamp(min=1)
        layer_weight = 1.0 / max(L_S, 1)

        for m in range(1, L_S + 1):
            s_hidden = student_hidden_states[m]
            t_fused = fuse_teacher_hidden_by_attention(
                teacher_hidden_states, alpha[:, m - 1, :], L_T
            )
            delta_T = compute_teacher_increments_batch(
                student_offsets_batch, teacher_offsets_batch,
                t_fused, W_proj, student_mask, teacher_mask,
                boundary_mode=boundary_mode,
            )
            delta_S = compute_student_increments(s_hidden, student_mask)
            cos_sim = F.cosine_similarity(
                delta_S.float(), delta_T.float(), dim=-1, eps=1e-6
            )
            cos_loss = (1.0 - cos_sim) * mask_float
            layer_loss = (cos_loss.sum(dim=1) / N).mean()
            total_loss = total_loss + layer_weight * layer_loss
        return total_loss

    pairs, align_mode, layer_mode = build_impact_alignment_pairs(
        choose_align,
        choose_layer,
        teacher_hidden_states,
        top_k,
        L_T,
        L_S,
        bi_tau=bi_tau,
        teacher_model=teacher_model,
        teacher_labels=teacher_labels,
        padding_id=padding_id,
        random_seed=random_seed,
        rail_teacher_layers=rail_teacher_layers,
        rail_kd_no_random=rail_kd_no_random,
        rail_kd_has_embed=rail_kd_has_embed,
        rail_kd_has_final=rail_kd_has_final,
        precomputed_layer_scores=precomputed_layer_scores,
    )

    if log_selection:
        try:
            from utils import log_rank
            t_layers = [p[0] for p in pairs]
            s_layers = [p[1] for p in pairs]
            weights = [round(p[2], 4) if not torch.is_tensor(p[2]) else round(p[2].item(), 4)
                       for p in pairs]
            msg = (
                "[IMPACT align] mode={} num_pairs={} "
                "teacher_layers(1-based)={} student_layers(1-based)={} weights={}"
            ).format(align_mode, len(pairs), t_layers, s_layers, weights)
            if layer_mode is not None:
                msg += " layer_select={}".format(layer_mode)
            if align_mode == "1-random" and rail_teacher_layers is not None:
                msg += " rail_sample={}".format(list(rail_teacher_layers)[:L_S])
            log_rank(msg)
        except ImportError:
            pass

    device = student_hidden_states[0].device
    total_loss = torch.tensor(0.0, device=device)
    mask_float = student_mask.float()
    N = mask_float.sum(dim=1).clamp(min=1)

    for teacher_layer, student_layer, weight in pairs:
        t_hidden = teacher_hidden_states[teacher_layer]
        s_hidden = student_hidden_states[min(student_layer, L_S)]
        w = weight if torch.is_tensor(weight) else torch.tensor(
            weight, device=device, dtype=total_loss.dtype
        )

        delta_T = compute_teacher_increments_batch(
            student_offsets_batch, teacher_offsets_batch,
            t_hidden, W_proj, student_mask, teacher_mask,
            boundary_mode=boundary_mode,
        )
        delta_S = compute_student_increments(s_hidden, student_mask)

        cos_sim = F.cosine_similarity(
            delta_S.float(), delta_T.float(), dim=-1, eps=1e-6
        )
        cos_loss = (1.0 - cos_sim) * mask_float
        layer_loss = (cos_loss.sum(dim=1) / N).mean()

        total_loss = total_loss + w * layer_loss

    return total_loss
