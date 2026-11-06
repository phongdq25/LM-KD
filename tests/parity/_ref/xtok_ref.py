"""Verbatim copies of the original cross-tokenizer helpers (parity references for T11).
Each block states its source file and lines. Do not edit."""
try:
    import editdistance
except ImportError:  # parity tests that need it importorskip it
    editdistance = None
import numpy as np
import torch
from torch.nn.functional import pad

N_SPAN = 1024


# --- DSKDv2-main/code/criterions/dual_space_kd_v2_with_exact_token_alignment.py:8-41  (align_sequences)
def align_sequences(tea_seq, stu_seq, student_tokenizer, teacher_tokenizer):
    i, j = 0, 0
    t2s_align, s2t_align = [], []
    history_tea_seq, history_stu_seq = "", ""

    tea_seq = [token.replace('▁', '').replace('Ġ', '') for token in tea_seq]
    stu_seq = [token.replace('▁', '').replace('Ġ', '') for token in stu_seq]

    while i < len(tea_seq) and j < len(stu_seq):
        if history_tea_seq == history_stu_seq and (
            tea_seq[i] == stu_seq[j] or (
                tea_seq[i] == teacher_tokenizer.eos_token and \
                stu_seq[j] == student_tokenizer.eos_token
            )
        ):
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            t2s_align.append(i)
            s2t_align.append(j)
            i += 1
            j += 1
        elif len(history_tea_seq) > len(history_stu_seq):
            history_stu_seq += stu_seq[j]
            j += 1
        elif len(history_tea_seq) < len(history_stu_seq):
            history_tea_seq += tea_seq[i]
            i += 1
        else:
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            i += 1
            j += 1

    return t2s_align, s2t_align


# --- IMPACT/code/criterions/E_SCOT.py:283-302  (_span_align)
def _span_align(s_spans, t_spans):
    """Algorithm 1 of paper 15849 — linear two-pointer sweep for character-span overlap.

    Returns: list of length |s_spans|; entry i is the list of teacher indices whose span
             [t_start, t_end) has non-empty intersection with student span [s_start, s_end).
    """
    neighborhoods = [[] for _ in range(len(s_spans))]
    if not t_spans:
        return neighborhoods
    j0 = 0
    for i, (st_i, ed_i) in enumerate(s_spans):
        # Advance j0 past teacher tokens that fully precede student token i.
        while j0 < len(t_spans) and t_spans[j0][1] <= st_i:
            j0 += 1
        # Collect all teacher tokens whose start is before student i's end.
        k = j0
        while k < len(t_spans) and t_spans[k][0] < ed_i:
            neighborhoods[i].append(k)
            k += 1
    return neighborhoods


# --- IMPACT/code/sra_span_utils.py:15-200  (LCS / pooler v2)
def longest_common_subsequence(a, b, s_i=0, s_j=0):
    """Greedy matching of character-boundary positions between two tokenizations.

    Compares end-character offsets (column 1 of offset_mapping) to identify
    positions where both tokenizers share a token boundary.  Handles word-
    boundary resets (offset == 0) by fast-forwarding past the current word.

    Args:
        a: student offset_mapping, array-like of shape [N, 2]
        b: teacher offset_mapping, array-like of shape [M, 2]
        s_i: starting student token index (typically prompt token count)
        s_j: starting teacher token index (typically prompt token count)

    Returns:
        List of (student_idx+1, teacher_idx+1) boundary pairs (1-indexed).
    """
    if hasattr(a, "numpy"):
        a = a.numpy()
    if hasattr(b, "numpy"):
        b = b.numpy()

    m, n = len(a), len(b)
    i, j = s_i, s_j
    result = []

    while i < m and j < n:
        if a[i][1] == 0:
            i += 1
            continue
        if b[j][1] == 0:
            j += 1
            continue

        if a[i][1] == b[j][1]:
            result.append((i + 1, j + 1))
            i += 1
            j += 1
        elif a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

        if i + 1 < m and j + 1 < n:
            if (a[i][1] == 0 and a[i + 1][1] > 0) or (
                b[j][1] == 0 and b[j + 1][1] > 0
            ):
                while j < n and b[j][1] > 0:
                    j += 1
                while i < m and a[i][1] > 0:
                    i += 1
                i += 1
                j += 1
                result.append((i, j))

    size = len(result)
    if size > N_SPAN:
        step = size / N_SPAN
        return [result[int((k + 1) * step) - 1] for k in range(N_SPAN)]

    return result


def get_pooler_tensor(segments_idxs):
    """Pad and stack per-sample segment indices into batched tensors.

    Args:
        segments_idxs: list of ``(seg_idx_list, max_len)`` per sample, where
            ``seg_idx_list`` is a list of 1-D LongTensors (token indices for
            each span).

    Returns:
        dict with ``safe_idx`` [B, max_seg, max_tok] and ``mask`` of same shape.
    """
    padded_idx_batch = []
    max_seg, max_len_all = 0, 0

    for seg_idx, max_len in segments_idxs:
        max_len_all = max(max_len_all, max_len)
        max_seg = max(max_seg, len(seg_idx))

        padded = torch.stack(
            [pad(x, (0, max_len - len(x)), value=-1) for x in seg_idx]
        )
        padded_idx_batch.append(padded)

    def pad2d(t, h, w):
        return pad(t, (0, w - t.size(1), 0, h - t.size(0)), value=-1)

    padded_idx_batch = torch.stack(
        [pad2d(p, max_seg, max_len_all) for p in padded_idx_batch]
    )

    mask = padded_idx_batch != -1
    safe_idx = padded_idx_batch.masked_fill(~mask, 0)

    return {"safe_idx": safe_idx, "mask": mask}


def prepare_pooler_v2(
    student_starts,
    student_offset_mapping,
    teacher_starts,
    teacher_offset_mapping,
    student_max_pos=None,
    teacher_max_pos=None,
):
    """Compute LCS span alignment and build pooler tensors for a batch.

    Args:
        student_starts: list[int] – per-sample student prompt token counts.
        student_offset_mapping: list of array-like [N_i, 2] per sample.
        teacher_starts: list[int] – per-sample teacher prompt token counts.
        teacher_offset_mapping: list of array-like [M_i, 2] per sample.
        student_max_pos: optional list[int] – exclusive upper bound for student
            span indices (used to clip to hidden-state length).
        teacher_max_pos: optional list[int] – same for teacher.

    Returns:
        ``(student_pooler_dict, teacher_pooler_dict)`` each with keys
        ``safe_idx`` and ``mask``.
    """
    student_seg_idxs, teacher_seg_idxs = [], []

    for idx, (s_start, s_offset, t_start, t_offset) in enumerate(
        zip(
            student_starts,
            student_offset_mapping,
            teacher_starts,
            teacher_offset_mapping,
        )
    ):
        if isinstance(s_start, torch.Tensor):
            s_start = s_start.item()
        if isinstance(t_start, torch.Tensor):
            t_start = t_start.item()

        token_offset_start = [(s_start, t_start)]
        lcs = longest_common_subsequence(s_offset, t_offset, s_start, t_start)
        longest_common_offset = token_offset_start + lcs

        student_seg_idx = []
        teacher_seg_idx = []
        student_max_len, teacher_max_len = 1, 1

        s_clip = (
            student_max_pos[idx]
            if student_max_pos is not None
            else len(s_offset)
        )
        t_clip = (
            teacher_max_pos[idx]
            if teacher_max_pos is not None
            else len(t_offset)
        )

        for k in range(1, len(longest_common_offset)):
            s_lo = longest_common_offset[k - 1][0]
            s_hi = min(longest_common_offset[k][0], s_clip)
            t_lo = longest_common_offset[k - 1][1]
            t_hi = min(longest_common_offset[k][1], t_clip)

            s_range = torch.arange(s_lo, s_hi, dtype=torch.long)
            t_range = torch.arange(t_lo, t_hi, dtype=torch.long)

            if len(s_range) > 0 and len(t_range) > 0:
                student_seg_idx.append(s_range)
                teacher_seg_idx.append(t_range)
                student_max_len = max(student_max_len, s_range.size(0))
                teacher_max_len = max(teacher_max_len, t_range.size(0))

        if len(student_seg_idx) == 0:
            student_seg_idx.append(
                torch.tensor([max(s_start, 0)], dtype=torch.long)
            )
            teacher_seg_idx.append(
                torch.tensor([max(t_start, 0)], dtype=torch.long)
            )
            student_max_len = 1
            teacher_max_len = 1

        student_seg_idxs.append((student_seg_idx, student_max_len))
        teacher_seg_idxs.append((teacher_seg_idx, teacher_max_len))

    return get_pooler_tensor(student_seg_idxs), get_pooler_tensor(
        teacher_seg_idxs
    )


# --- IMPACT/code/impact_utils.py:665-706  (_get_teacher_boundary_state)
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


# --- IMPACT/code/criterions/MinED.py:368-411  (dtw (MinED, method -> function))
def dtw_min_ed(series_1, series_2, norm_func=np.linalg.norm):
    """Use dynamic time wrapping to align to tokenizers, modified from:
    https://github.com/talcs/simpledtw/blob/master/simpledtw.py"""
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
            )
    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1 = [list() for v in range(matrix.shape[0])]
    mappings_series_2 = [list() for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix


# --- IMPACT/code/criterions/MCW_KD.py:433-464  (dtw_alignment (MCW))
def dist_fn_edit(a, b):
    return editdistance.eval(a, b)

def dtw_alignment(series_1, series_2, norm_func=dist_fn_edit):
    """Simple DTW based on FUSELLM"""
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
            )
    matrix = matrix[1:, 1:]
    i, j = len(series_1) - 1, len(series_2) - 1
    aligned = []
    while i > 0 or j > 0:
        aligned.append((i, j))
        options = [
            matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
            matrix[i - 1, j] if i > 0 else np.inf,
            matrix[i, j - 1] if j > 0 else np.inf,
        ]
        move = np.argmin(options)
        if move == 0: i -= 1; j -= 1
        elif move == 1: i -= 1
        else: j -= 1
    aligned.append((0, 0))
    return aligned

