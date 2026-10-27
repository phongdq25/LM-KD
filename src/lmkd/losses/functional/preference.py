"""Pure PyTorch preference distillation losses and utilities.

Formulas match Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/:
  - distillation_losses.py (VPD, PPD, DCKD compressed KL, TVKD, ADPA, CTPD, PBSD)
  - preference/losses.py (DPO, DP3O)
"""
from __future__ import annotations

from itertools import permutations

import torch
import torch.nn.functional as F


def _score_matrix(scores: torch.Tensor, name: str) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError(f"{name} must have shape [batch, responses] with at least two responses")
    if not torch.isfinite(scores).all():
        raise ValueError(f"{name} contains a non-finite value")
    return scores


def plackett_luce_log_probability(scores: torch.Tensor, rankings: torch.Tensor) -> torch.Tensor:
    """Return the log probability of each ranking under Plackett-Luce scores."""
    _score_matrix(scores, "scores")
    if rankings.shape != scores.shape:
        raise ValueError("rankings must have the same shape as scores")
    ordered = scores.gather(1, rankings.long())
    denominators = torch.logcumsumexp(ordered.flip(-1), dim=-1).flip(-1)
    return (ordered - denominators).sum(dim=-1)


def vpd_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor, *, beta: float = 1.0) -> torch.Tensor:
    """Vanilla preference distillation via teacher-induced ranking likelihood."""
    _score_matrix(student_scores, "student_scores")
    _score_matrix(teacher_scores, "teacher_scores")
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student_scores and teacher_scores must have the same shape")
    rankings = teacher_scores.argsort(dim=-1, descending=True)
    return -plackett_luce_log_probability(beta * student_scores, rankings)


def ppd_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    beta: float = 1.0,
    exact_ranking_limit: int = 8,
) -> torch.Tensor:
    """Probabilistic preference distillation using exact ranking-distribution JSD."""
    _score_matrix(student_scores, "student_scores")
    _score_matrix(teacher_scores, "teacher_scores")
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student_scores and teacher_scores must have the same shape")
    batch_size, response_count = student_scores.shape
    if response_count > exact_ranking_limit:
        raise ValueError(
            f"Exact PPD enumerates rankings and supports at most {exact_ranking_limit} responses; received {response_count}"
        )

    ranking_table = torch.tensor(
        tuple(permutations(range(response_count))),
        device=student_scores.device,
        dtype=torch.long,
    )
    ranking_count = ranking_table.shape[0]
    expanded_rankings = ranking_table.unsqueeze(0).expand(batch_size, -1, -1)

    def _distribution(scores: torch.Tensor) -> torch.Tensor:
        flat_scores = scores[:, None, :].expand(-1, ranking_count, -1).reshape(-1, response_count)
        flat_rankings = expanded_rankings.reshape(-1, response_count)
        log_probs = plackett_luce_log_probability(flat_scores, flat_rankings)
        log_probs = log_probs.reshape(batch_size, ranking_count)
        return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)

    student_log_probs = _distribution(beta * student_scores)
    teacher_log_probs = _distribution(beta * teacher_scores)
    mixture_log_probs = torch.logaddexp(student_log_probs, teacher_log_probs) - torch.log(
        student_scores.new_tensor(2.0)
    )
    student_kl = (student_log_probs.exp() * (student_log_probs - mixture_log_probs)).sum(dim=-1)
    teacher_kl = (teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)
    return 0.5 * (student_kl + teacher_kl)


def compressed_projection_kl(
    student_probs: torch.Tensor,
    ids: torch.Tensor,
    p: torch.Tensor,
    tail_p: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Token-level KL(teacher || student) using compressed top-k teacher distribution."""
    if student_probs.ndim != 2:
        raise ValueError("student_probs must have shape [tokens, vocab]")
    v_s = student_probs.shape[-1]
    in_range = (ids >= 0) & (ids < v_s)
    res_prime = tail_p + (p * (~in_range).to(p.dtype)).sum(dim=-1)
    safe_ids = ids.clamp(min=0, max=v_s - 1)
    q_sel = student_probs.gather(-1, safe_ids) * in_range.to(student_probs.dtype)
    q_res = (1.0 - q_sel.sum(dim=-1, keepdim=True)).clamp_min(0.0)

    keep = torch.cat([in_range, in_range.new_ones(*in_range.shape[:-1], 1)], dim=-1)
    concat_s = torch.cat([q_sel, q_res], dim=-1)
    concat_p = torch.cat([p * in_range.to(p.dtype), res_prime.unsqueeze(-1)], dim=-1)

    s = torch.where(keep, concat_s.clamp_min(eps), concat_s.new_zeros(1))
    s = s / s.sum(dim=-1, keepdim=True)
    p_proj = torch.where(keep, concat_p.clamp_min(0.0), concat_p.new_zeros(1))
    p_sum = p_proj.sum(dim=-1, keepdim=True)
    if (p_sum <= 0).any():
        raise ValueError("Compressed teacher distribution has no positive probability mass")
    p_proj = p_proj / p_sum

    return torch.where(
        keep & (p_proj > 0),
        p_proj * (p_proj.clamp_min(eps).log() - s.log()),
        torch.zeros_like(p_proj),
    ).sum(dim=-1)


def _project_support(
    flat_probs: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_p: torch.Tensor,
    flat_tail: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    v_s = flat_probs.shape[-1]
    in_range = (flat_ids >= 0) & (flat_ids < v_s)
    res_prime = flat_tail + (flat_p * (~in_range).to(flat_p.dtype)).sum(dim=-1)
    safe_ids = flat_ids.clamp(min=0, max=v_s - 1)
    q_sel = flat_probs.gather(-1, safe_ids) * in_range.to(flat_probs.dtype)
    q_res = (1.0 - q_sel.sum(dim=-1, keepdim=True)).clamp_min(0.0)

    keep = torch.cat([in_range, in_range.new_ones(*in_range.shape[:-1], 1)], dim=-1)
    concat_s = torch.cat([q_sel, q_res], dim=-1)
    concat_p = torch.cat([flat_p * in_range.to(flat_p.dtype), res_prime.unsqueeze(-1)], dim=-1)

    s = torch.where(keep, concat_s.clamp_min(eps), concat_s.new_zeros(1))
    s = s / s.sum(dim=-1, keepdim=True)
    p_proj = torch.where(keep, concat_p.clamp_min(0.0), concat_p.new_zeros(1))
    p_sum = p_proj.sum(dim=-1, keepdim=True)
    if (p_sum <= 0).any():
        raise ValueError("Compressed teacher distribution has no positive probability mass")
    p_proj = p_proj / p_sum
    return s, p_proj, keep, in_range


def _tvkd_values(
    s: torch.Tensor,
    p: torch.Tensor,
    keep: torch.Tensor,
    func: str,
    temp: float,
    s_wt: float,
    t_wt: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    s_log = s.clamp_min(eps).log()
    p_log = p.clamp_min(eps).log()
    if func == "entropy":
        values = -(torch.where(keep, s * s_log, torch.zeros_like(s))).sum(dim=-1)
    elif func == "soft":
        comb = (s_wt * s_log + t_wt * p_log) / temp
        comb = torch.where(keep, comb, torch.full_like(comb, -1e9))
        values = temp * torch.logsumexp(comb, dim=-1)
    else:
        raise ValueError(f"Unknown TVKD value function: {func}")
    return values, s_log


def tvkd_scores(
    student_probs: torch.Tensor,
    labels: torch.Tensor,
    ids: torch.Tensor,
    p: torch.Tensor,
    tail_p: torch.Tensor,
    *,
    value_function: str = "entropy",
    value_discount: float = 1.0,
    value_temperature: float = 1.0,
    student_value_weight: float = 0.0,
    teacher_value_weight: float = 0.7,
    eps: float = 1e-8,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Vectorized sequence score for TVKD: sum_t (action_t - discount * next_value_t)."""
    mask = labels.ne(ignore_index).clone()
    mask[:, 0] = False
    if not mask.any(dim=-1).all():
        raise ValueError("TVKD teacher probabilities do not align with response labels")

    b, t, v_s = student_probs.shape
    k = ids.shape[-1]
    s, p_proj, keep, in_range = _project_support(
        student_probs.reshape(-1, v_s), ids.reshape(-1, k), p.reshape(-1, k), tail_p.reshape(-1), eps
    )
    values, s_log = _tvkd_values(
        s, p_proj, keep, value_function, value_temperature, student_value_weight, teacher_value_weight, eps
    )

    flat_labels = labels.reshape(-1)
    matched = (ids.reshape(-1, k) == flat_labels.unsqueeze(-1)) & in_range
    action_index = torch.where(matched.any(dim=-1), matched.long().argmax(dim=-1), k)
    actions = s_log.gather(-1, action_index.unsqueeze(-1)).squeeze(-1)

    actions = torch.where(mask, actions.reshape(b, t), torch.zeros_like(mask, dtype=actions.dtype))
    values = torch.where(mask, values.reshape(b, t), torch.zeros_like(mask, dtype=values.dtype))

    first_idx = mask.long().argmax(dim=-1)
    first_values = values.gather(-1, first_idx.unsqueeze(-1)).squeeze(-1)
    next_value_sums = values.sum(dim=-1) - first_values
    return actions.sum(dim=-1) - value_discount * next_value_sums


def tvkd_loss(
    chosen_student_probs: torch.Tensor,
    rejected_student_probs: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    chosen_ids: torch.Tensor,
    chosen_p: torch.Tensor,
    chosen_tail_p: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_p: torch.Tensor,
    rejected_tail_p: torch.Tensor,
    *,
    value_function: str = "entropy",
    value_discount: float = 1.0,
    value_temperature: float = 1.0,
    student_value_weight: float = 0.0,
    teacher_value_weight: float = 0.7,
    eps: float = 1e-8,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute TVKD preference distillation loss from chosen and rejected sequences."""
    chosen_scores = tvkd_scores(
        chosen_student_probs,
        chosen_labels,
        chosen_ids,
        chosen_p,
        chosen_tail_p,
        value_function=value_function,
        value_discount=value_discount,
        value_temperature=value_temperature,
        student_value_weight=student_value_weight,
        teacher_value_weight=teacher_value_weight,
        eps=eps,
        ignore_index=ignore_index,
    )
    rejected_scores = tvkd_scores(
        rejected_student_probs,
        rejected_labels,
        rejected_ids,
        rejected_p,
        rejected_tail_p,
        value_function=value_function,
        value_discount=value_discount,
        value_temperature=value_temperature,
        student_value_weight=student_value_weight,
        teacher_value_weight=teacher_value_weight,
        eps=eps,
        ignore_index=ignore_index,
    )
    return -F.logsigmoid(chosen_scores - rejected_scores)


def advantage_expectation_loss(
    student_probs: torch.Tensor,
    labels: torch.Tensor,
    adv_ids: torch.Tensor,
    adv_val: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Maximize the expected teacher-reference token advantage used by ADPA."""
    mask = labels.ne(ignore_index).clone()
    mask[:, 0] = False
    v_s = student_probs.shape[-1]
    valid = (adv_ids >= 0) & (adv_ids < v_s)
    has_valid = valid.any(dim=-1)
    keep_mask = mask & has_valid
    if not keep_mask.any():
        raise ValueError("No ADPA advantages align with non-masked labels")

    safe_ids = adv_ids.clamp(min=0, max=v_s - 1)
    q = student_probs.gather(-1, safe_ids)
    term = (q * adv_val * valid.to(student_probs.dtype)).sum(dim=-1)
    return -term[keep_mask].mean()


def _parent_sums_tensor(tok: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    b, g, _ = groups.shape
    valid_idx = (groups >= 0) & (groups < tok.shape[-1])
    safe_groups = groups.clamp(min=0, max=tok.shape[-1] - 1)
    gathered = tok.unsqueeze(1).expand(-1, g, -1).gather(-1, safe_groups)
    return (gathered * valid_idx.to(tok.dtype)).sum(dim=-1)


def ctpd_loss(
    policy_chosen_token_logps: torch.Tensor,
    policy_rejected_token_logps: torch.Tensor,
    reference_chosen_token_logps: torch.Tensor,
    reference_rejected_token_logps: torch.Tensor,
    chosen_parent_groups: torch.Tensor,
    rejected_parent_groups: torch.Tensor,
    chosen_weights: torch.Tensor,
    rejected_weights: torch.Tensor,
    *,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute weighted parent-token KD-TIS-DPO losses for CTPD."""
    p_c = _parent_sums_tensor(policy_chosen_token_logps, chosen_parent_groups)
    r_c = _parent_sums_tensor(reference_chosen_token_logps, chosen_parent_groups)
    p_r = _parent_sums_tensor(policy_rejected_token_logps, rejected_parent_groups)
    r_r = _parent_sums_tensor(reference_rejected_token_logps, rejected_parent_groups)

    chosen_count = p_c.shape[1]
    rejected_count = p_r.shape[1]
    if chosen_weights.shape[1] < chosen_count or rejected_weights.shape[1] < rejected_count:
        raise ValueError("CTPD weight count does not match its parent-token groups")

    chosen_margin = ((p_c - r_c) * chosen_weights[:, :chosen_count]).sum(dim=-1)
    rejected_margin = ((p_r - r_r) * rejected_weights[:, :rejected_count]).sum(dim=-1)
    return -F.logsigmoid(beta * (chosen_margin - rejected_margin))


def sigmoid_dpo_loss(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """TRL's sigmoid DPO loss on summed policy-minus-reference log ratios."""
    logits = beta * (chosen - rejected)
    return -(1 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)


def dpo_loss(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    *,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    """DPO family objective supporting sigmoid, ipo, hinge, and simpo variants."""
    if loss_type == "sigmoid":
        return sigmoid_dpo_loss(chosen, rejected, beta=beta, label_smoothing=label_smoothing)
    if loss_type == "simpo":
        logits = beta * (chosen - rejected) - gamma
        return -F.logsigmoid(logits)
    if loss_type == "ipo":
        return (chosen - rejected - 1.0 / (2.0 * beta)) ** 2
    if loss_type == "hinge":
        return F.relu(1.0 - beta * (chosen - rejected))
    raise ValueError(f"Unknown DPO loss_type: {loss_type}")


def dp3o_loss(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    reward_diff: torch.Tensor | None = None,
    *,
    beta: float = 0.1,
    alpha: float = 0.2,
) -> torch.Tensor:
    """DP3O objective: alpha * hard_DPO + (1 - alpha) * soft_DPO."""
    logits = beta * (chosen - rejected)
    hard = -F.logsigmoid(logits)
    if alpha == 1.0:
        return hard
    if reward_diff is None or reward_diff.shape != chosen.shape:
        raise ValueError("DP3O with dp3o_alpha < 1 requires one reward difference per pair")
    target = torch.sigmoid(reward_diff.to(logits.dtype))
    soft = -F.logsigmoid(logits) * target - F.logsigmoid(-logits) * (1 - target)
    return alpha * hard + (1 - alpha) * soft


def pbsd_loss(
    teacher_on_teacher: torch.Tensor,
    student_on_teacher: torch.Tensor,
    teacher_on_student: torch.Tensor,
    student_on_student: torch.Tensor,
    *,
    beta: float = 0.1,
    loss_type: str = "logistic_gap",
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preference-Based Self-Distillation objective (Yu et al., 2026)."""
    delta = (teacher_on_teacher - student_on_teacher) - (teacher_on_student - student_on_student)
    if loss_type == "logistic_gap":
        losses = -F.logsigmoid(-beta * delta)
    elif loss_type == "mse":
        losses = (torch.sigmoid(delta) - 0.5) ** 2
    elif loss_type == "mse_to_zero":
        losses = delta * delta
    else:
        raise ValueError(f"Unknown PBSD loss type: {loss_type}")
    if gate is None:
        gate = torch.ones_like(delta)
    return (losses * gate).sum() / gate.sum().clamp(min=1.0)
