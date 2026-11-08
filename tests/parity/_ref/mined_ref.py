"""Reference implementation of MinED alignment and loss."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tests.parity._ref.various_divergence_ref import VariousDivergenceRef


def dtw(
    series_1: list[Any], series_2: list[Any], norm_func: Any = np.linalg.norm
) -> tuple[list[tuple[int, int]], float, list[list[int]], list[list[int]], np.ndarray]:
    """Dynamic time warping to align token sequences."""
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
    mappings_series_1: list[list[int]] = [list() for _ in range(matrix.shape[0])]
    mappings_series_2: list[list[int]] = [list() for _ in range(matrix.shape[1])]
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

    return matches, float(matrix[-1, -1]), mappings_series_1, mappings_series_2, matrix


def transform_step_logits_fast_ref(
    base_model_tokens: list[str],
    blending_model_tokens: list[str],
    base_model_input_ids: torch.LongTensor,
    base_model_per_step_logits: torch.FloatTensor,
    base_model_target_ids: torch.LongTensor,
    blending_model_input_ids: torch.LongTensor,
    blending_model_per_step_logits: torch.FloatTensor,
    blending_to_base_mapping: torch.LongTensor,
    base_to_blending_mapping_blending_ids: torch.LongTensor,
    base_to_blending_mapping_base_ids: torch.LongTensor,
    base_model_special_token: str = "Ġ",
    blending_model_special_token: str = "Ġ",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (blending_to_base_logits, unalign_mask)."""
    import editdistance

    def dist_fn(a: str, b: str) -> int:
        aa = a.replace(blending_model_special_token, "")
        bb = b.replace(base_model_special_token, "")
        return editdistance.eval(aa, bb)

    _, _, _, base_to_blending, _ = dtw(
        blending_model_tokens, base_model_tokens, norm_func=dist_fn
    )
    unalign_mask = [1 if len(a) == 1 else 0 for a in base_to_blending]
    unalign_mask_t = torch.tensor(unalign_mask, device=base_model_input_ids.device, dtype=torch.bool)

    base_to_blending_first = [a[0] if len(a) == 1 else 0 for a in base_to_blending]
    base_to_blending_t = torch.tensor(base_to_blending_first, device=base_model_input_ids.device, dtype=torch.long)

    # 1:1 mapped check
    unalign_mask_t = unalign_mask_t & base_model_input_ids.eq(
        blending_to_base_mapping[blending_model_input_ids[base_to_blending_t]]
    )

    blending_per_step = blending_model_per_step_logits[base_to_blending_t]
    if base_to_blending_mapping_blending_ids.ndim == 1:
        tea_ids = base_to_blending_mapping_blending_ids
        blending_mapped = blending_per_step[:, tea_ids]
    else:
        blending_per_step = blending_per_step[:, base_to_blending_mapping_blending_ids.view(-1)]
        blending_mapped = blending_per_step.view(
            -1,
            base_to_blending_mapping_blending_ids.shape[0],
            base_to_blending_mapping_blending_ids.shape[1],
        ).max(-1)[0]

    blending_to_base_logits = torch.ones_like(base_model_per_step_logits) * (-100000.0)
    blending_to_base_logits[:, base_to_blending_mapping_base_ids] = blending_mapped

    unalign_mask_t = unalign_mask_t & (blending_to_base_logits.max(-1)[0] != -100000.0)

    one_hot_logits = F.one_hot(base_model_target_ids, num_classes=base_model_per_step_logits.shape[-1]).float()
    one_hot_logits = (1.0 - one_hot_logits) * (-100000.0) + one_hot_logits * 100.0

    blending_to_base_logits = torch.where(
        unalign_mask_t.unsqueeze(-1),
        blending_to_base_logits,
        one_hot_logits,
    )
    return blending_to_base_logits, unalign_mask_t


def get_aligned_teacher_logits_ref(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_input_ids: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    student_target: torch.Tensor,
    teacher_target: torch.Tensor,
    s_tokens: list[str],
    t_tokens: list[str],
    blending_to_base_mapping: torch.Tensor,
    map_s: torch.Tensor,
    map_t: torch.Tensor,
    padding_id: int = -100,
) -> torch.Tensor:
    """Rebuilds aligned teacher logits over the batch."""
    bsz = student_target.shape[0]
    aligned_tea_logits = []
    for i in range(bsz):
        stu_content_idx = torch.nonzero(student_target[i].ne(padding_id)).view(-1)
        tea_content_idx = torch.nonzero(teacher_target[i].ne(padding_id)).view(-1)

        stu_input = student_input_ids[i, stu_content_idx]
        stu_target_ids = student_target[i, stu_content_idx]
        tea_input = teacher_input_ids[i, tea_content_idx]

        stu_per_step = student_logits[i, stu_content_idx, :].float()
        tea_per_step = teacher_logits[i, tea_content_idx, :].float()

        stu_toks = [s_tokens[idx] for idx in stu_content_idx.tolist()]
        tea_toks = [t_tokens[idx] for idx in tea_content_idx.tolist()]

        aligned_content, _ = transform_step_logits_fast_ref(
            base_model_tokens=stu_toks,
            blending_model_tokens=tea_toks,
            base_model_input_ids=stu_input,
            base_model_per_step_logits=stu_per_step,
            base_model_target_ids=stu_target_ids,
            blending_model_input_ids=tea_input,
            blending_model_per_step_logits=tea_per_step,
            blending_to_base_mapping=blending_to_base_mapping,
            base_to_blending_mapping_blending_ids=map_t,
            base_to_blending_mapping_base_ids=map_s,
        )

        aligned_per_step = student_logits[i].float().detach().clone()
        aligned_per_step[stu_content_idx] = aligned_content
        aligned_tea_logits.append(aligned_per_step)

    return torch.stack(aligned_tea_logits, dim=0)


def compute_mined_loss_ref(
    student_logits: torch.Tensor,
    aligned_teacher_logits: torch.Tensor,
    student_labels: torch.Tensor,
    kd_temp: float = 1.0,
    padding_id: int = -100,
) -> torch.Tensor:
    ref = VariousDivergenceRef(kd_temp=kd_temp, padding_id=padding_id)
    return ref.compute_forward_kl_divergence(
        student_logits, aligned_teacher_logits, student_labels, reduction="sum"
    )
