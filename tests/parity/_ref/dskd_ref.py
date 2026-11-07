"""Reference implementations for DSKD family (T12).
Reference implementations for DSKD baselines.
Do not edit unless marking `# FIX(original bug): ...`.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from tests.parity._ref.various_divergence_ref import VariousDivergenceRef
from tests.parity._ref.xtok_ref import align_sequences


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    padding_id: int = -100,
    reduction: str = "sum",
) -> tuple[torch.Tensor, Any]:
    pad_mask = target.ne(padding_id)
    target_clean = torch.where(target.eq(padding_id), torch.zeros_like(target), target)
    logits_clean = logits.masked_fill(logits.isnan() | logits.isinf(), 0.0)
    lprobs = torch.log_softmax(logits_clean, -1, dtype=torch.float32)
    nll_loss = -lprobs.gather(-1, target_clean.unsqueeze(-1)).squeeze(-1)
    if reduction == "sum":
        loss = (nll_loss * pad_mask).sum()
    elif reduction == "mean":
        loss = (nll_loss * pad_mask).sum() / pad_mask.sum().clamp(min=1)
    else:
        loss = nll_loss * pad_mask
    return loss, None


# --- DSKD reference
def compute_dual_space_kd_loss_ref(
    outputs: Any,
    teacher_outputs: Any,
    batch: dict[str, Any],
    distiller: Any,
    args: Any,
    div_ref: VariousDivergenceRef,
    padding_id: int = -100,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target = batch["label_batch"]["label"]
    pad_mask = target.ne(padding_id)
    teacher_target = batch["teacher_label_batch"]["label"]
    teacher_pad_mask = teacher_target.ne(padding_id)

    hiddens = outputs.hidden_states[-1]
    teacher_hiddens = teacher_outputs.hidden_states[-1]

    # student space
    t2s_hiddens = distiller.t2s_projector(teacher_hiddens)
    t2s_logits = t2s_hiddens.matmul(
        distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
    )

    t_preds = teacher_outputs.logits.argmax(-1)
    t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)
    t2s_ce_loss = compute_cross_entropy_loss(
        t2s_logits, target, padding_id=padding_id, reduction="sum"
    )[0] / batch["label_batch"]["loss_denom"]

    t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds) * pad_mask
    t2s_agreement = t2s_agreement_mask.sum() / batch["label_batch"]["loss_denom"]

    t2s_kd_loss = div_ref.dist_func(
        outputs.logits, t2s_logits.detach(), target, reduction="none"
    )
    t2s_kd_loss = (t2s_kd_loss * pad_mask).sum() / batch["label_batch"]["loss_denom"]

    # teacher space
    s2t_hiddens = distiller.s2t_projector(hiddens)
    s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)

    s2t_kd_loss = div_ref.dist_func(
        s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
    )
    s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / batch["label_batch"]["loss_denom"]

    if getattr(args, "only_stu_kd", False):
        kd_loss = t2s_kd_loss + t2s_ce_loss
    elif getattr(args, "only_tea_kd", False):
        kd_loss = s2t_kd_loss
    else:
        kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss

    s2t_agreement = (s2t_logits.argmax(-1).eq(outputs.logits.argmax(-1)) * pad_mask).sum() / batch["label_batch"]["loss_denom"]
    t_acc = (teacher_outputs.logits.argmax(-1).eq(target) * pad_mask).sum() / batch["label_batch"]["loss_denom"]
    t2s_acc = (t2s_logits.argmax(-1).eq(target) * pad_mask).sum() / batch["label_batch"]["loss_denom"]

    log = {
        "t2s_ce_loss": t2s_ce_loss,
        "t2s_kd_loss": t2s_kd_loss,
        "s2t_kd_loss": s2t_kd_loss,
        "t2s_agreement": t2s_agreement,
        "s2t_agreement": s2t_agreement,
        "t_acc": t_acc,
        "t2s_acc": t2s_acc,
        "kd_loss": kd_loss,
    }
    return kd_loss, log


# --- DSKDv2 reference
def compute_dual_space_kd_v2_loss_ref(
    outputs: Any,
    teacher_outputs: Any,
    batch: dict[str, Any],
    distiller: Any,
    args: Any,
    div_ref: VariousDivergenceRef,
    padding_id: int = -100,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target = batch["label_batch"]["label"]
    pad_mask = target.ne(padding_id)
    teacher_target = batch["teacher_label_batch"]["label"]
    teacher_pad_mask = teacher_target.ne(padding_id)

    hiddens = outputs.hidden_states[-1]
    teacher_hiddens = teacher_outputs.hidden_states[-1]

    # student space
    t2s_hiddens = distiller.t2s_projector(teacher_hiddens)
    t2s_logits = t2s_hiddens.matmul(
        distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
    )

    t_preds = teacher_outputs.logits.argmax(-1)
    t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)
    t2s_ce_loss = compute_cross_entropy_loss(
        t2s_logits, t_preds, padding_id=padding_id, reduction="sum"
    )[0] / batch["label_batch"]["loss_denom"]

    t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds) * pad_mask
    t2s_agreement = t2s_agreement_mask.sum() / batch["label_batch"]["loss_denom"]

    t2s_kd_loss = div_ref.dist_func(
        outputs.logits, t2s_logits.detach(), target, reduction="none"
    )

    if t2s_agreement <= distiller.args.t2s_agreement:
        t2s_kd_loss = (t2s_kd_loss * t2s_agreement_mask).sum() / max(1e-3, float(t2s_agreement_mask.sum()))
    else:
        t2s_kd_loss = (t2s_kd_loss * pad_mask).sum() / batch["label_batch"]["loss_denom"]

    # teacher space
    if getattr(args, "init_s2t_projector", False):
        stu_lm_head = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
        # FIX(original bug): when vocabularies differ, use overlap_s_part
        if hasattr(distiller, "student_overlap_token_ids") and distiller.student_overlap_token_ids is not None:
            stu_lm_head = stu_lm_head[:, distiller.student_overlap_token_ids]
        if getattr(args, "topk_vocab", -1) != -1:
            stu_lm_head = stu_lm_head[:, :args.topk_vocab]
        s2t_projector = stu_lm_head @ distiller.part_teacher_head_pinv
        s2t_hiddens = hiddens @ s2t_projector
    else:
        s2t_hiddens = distiller.s2t_projector(hiddens)

    s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)

    s2t_kd_loss = div_ref.dist_func(
        s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
    )
    s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / batch["label_batch"]["loss_denom"]

    if getattr(args, "only_stu_kd", False):
        kd_loss = t2s_kd_loss + t2s_ce_loss
    elif getattr(args, "only_tea_kd", False):
        kd_loss = s2t_kd_loss
    else:
        kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss

    s2t_agreement = (s2t_logits.argmax(-1).eq(outputs.logits.argmax(-1)) * pad_mask).sum() / batch["label_batch"]["loss_denom"]
    t_acc = (teacher_outputs.logits.argmax(-1).eq(target) * pad_mask).sum() / batch["label_batch"]["loss_denom"]
    t2s_acc = (t2s_logits.argmax(-1).eq(target) * pad_mask).sum() / batch["label_batch"]["loss_denom"]

    log = {
        "t2s_ce_loss": t2s_ce_loss,
        "t2s_kd_loss": t2s_kd_loss,
        "s2t_kd_loss": s2t_kd_loss,
        "t2s_agreement": t2s_agreement,
        "s2t_agreement": s2t_agreement,
        "t_acc": t_acc,
        "t2s_acc": t2s_acc,
        "kd_loss": kd_loss,
    }
    return kd_loss, log


# --- DSKD-CMA reference
def compute_dual_space_kd_loss_with_cma_ref(
    outputs: Any,
    teacher_outputs: Any,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    distiller: Any,
    args: Any,
    div_ref: VariousDivergenceRef,
    padding_id: int = -100,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target = output_data["label"]
    teacher_target = output_data["teacher_label"]
    pad_mask = target.ne(padding_id)
    teacher_pad_mask = teacher_target.ne(padding_id)

    hiddens = outputs.hidden_states[-1]
    teacher_hiddens = teacher_outputs.hidden_states[-1]

    stu_embed_tokens = distiller.student_embed_tokens
    tea_embed_tokens = distiller.teacher_embed_tokens

    formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
    formal_input = torch.where(pad_mask, input_data["input_ids"], torch.zeros_like(target))
    stu_input_embeds = stu_embed_tokens(formal_input).detach()
    stu_target_embeds = stu_embed_tokens(formal_target).detach()

    formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
    formal_teacher_input = torch.where(teacher_pad_mask, input_data["teacher_input_ids"], torch.zeros_like(teacher_target))
    tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
    tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

    stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
    tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

    norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
    norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
    norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

    stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float()
    tea_k_hiddens = norm_tea_index_embeds.float()

    stu_v_hiddens = distiller.projectors["s2t"](hiddens).float()
    tea_v_hiddens = distiller.projectors["t2s"](
        norm_teacher_hiddens + norm_tea_target_embeds
    ).float()

    align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
    align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
    align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
    align = align + (1.0 - align_mask) * (-100000)

    t2s_weight = torch.softmax(align, -1)
    t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)
    t2s_logits = t2s_hiddens.matmul(
        distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
    )

    t2s_ce_loss = compute_cross_entropy_loss(t2s_logits, target, padding_id=padding_id, reduction="sum")[0]
    t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
    t2s_acc = (t2s_acc_mask * pad_mask).sum()
    max_probs = (t2s_logits.softmax(-1).max(-1)[0] * pad_mask).sum()

    log = {"t2s_ce_loss": t2s_ce_loss, "t2s_acc": t2s_acc, "max_t2s_prob": max_probs}

    if not getattr(args, "only_save_projector", False):
        t2s_kd_loss = div_ref.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none", use_tea_temp=True
        )
        t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum()

        s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)
        s2t_logits = s2t_hiddens.matmul(
            distiller.teacher_model.lm_head.weight.detach().transpose(-1, -2)
        )

        s2t_kd_loss = div_ref.compute_forward_kl_divergence(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum()
        s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum().clamp(min=1)

        kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["s2t_acc"] = s2t_acc
    else:
        kd_loss = t2s_ce_loss

    log["kd_loss"] = kd_loss
    return kd_loss, log


# --- DSKDv2 with ETA reference
def compute_dual_space_kd_loss_with_eta_ref(
    outputs: Any,
    teacher_outputs: Any,
    batch: dict[str, Any],
    distiller: Any,
    args: Any,
    div_ref: VariousDivergenceRef,
    padding_id: int = -100,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target = batch["label_batch"]["label"]
    teacher_target = batch["teacher_label_batch"]["label"]

    pad_mask = target.ne(padding_id)
    teacher_pad_mask = teacher_target.ne(padding_id)

    hiddens = outputs.hidden_states[-1]
    teacher_hiddens = teacher_outputs.hidden_states[-1]

    # teacher space
    if getattr(args, "init_s2t_projector", False):
        stu_lm_head = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
        stu_lm_head = stu_lm_head[:, distiller.student_overlap_token_ids]
        if getattr(args, "topk_vocab", -1) != -1:
            stu_lm_head = stu_lm_head[:, :args.topk_vocab]
        s2t_projector = stu_lm_head @ distiller.part_teacher_head_pinv
        stu_v_hiddens = hiddens @ s2t_projector
    else:
        stu_v_hiddens = distiller.s2t_projector(hiddens)

    tea_v_hiddens = distiller.t2s_projector(teacher_hiddens)

    t_preds = teacher_outputs.logits.argmax(-1)
    t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)

    t_preds_as_label = []
    t2s_hiddens_align = torch.zeros_like(tea_v_hiddens).to(tea_v_hiddens.device).to(tea_v_hiddens)
    s2t_hiddens_align = torch.zeros_like(stu_v_hiddens).to(stu_v_hiddens.device).to(stu_v_hiddens)
    align_ratio = []
    for i in range(t_preds.shape[0]):
        indices_t_preds = torch.where(t_preds[i] != -100)[0]
        indices_t_target = torch.where(teacher_target[i] != -100)[0]
        indices_s_target = torch.where(target[i] != -100)[0]

        if indices_t_preds.shape[0] == 0:
            t_preds_as_label.append(torch.tensor([-100] * t_preds.shape[-1], device=tea_v_hiddens.device))
            align_ratio.append(1.0)
            continue

        cur_t_preds = t_preds[i][indices_t_preds[0]: indices_t_preds[-1] + 1]
        cur_t_target = teacher_target[i][indices_t_target[0]: indices_t_target[-1] + 1]
        cur_t_target_tokens = distiller.teacher_tokenizer.convert_ids_to_tokens(cur_t_target.tolist())

        cur_s_target = target[i][indices_s_target[0]: indices_s_target[-1] + 1]
        cur_s_target_tokens = distiller.student_tokenizer.convert_ids_to_tokens(cur_s_target.tolist())

        align_t_idx, align_s_idx = align_sequences(
            cur_t_target_tokens,
            cur_s_target_tokens,
            distiller.student_tokenizer,
            distiller.teacher_tokenizer,
        )
        if align_t_idx == [] and align_s_idx == []:
            t_preds_as_label.append(torch.tensor([-100] * t_preds.shape[-1], device=tea_v_hiddens.device))
            align_ratio.append(1.0)
            continue
        cur_align_ratio = len(align_s_idx) / len(cur_s_target)
        align_ratio.append(cur_align_ratio)

        cur_t_preds_as_label_1 = target[i][:indices_s_target[0]].cpu().tolist()
        cur_t_preds_as_label_2 = [-100] * len(cur_s_target)
        for _t_idx, _s_idx in zip(align_t_idx, align_s_idx):
            tmp_t_token = distiller.teacher_tokenizer.convert_ids_to_tokens([int(cur_t_preds[_t_idx])])
            if cur_t_preds[_t_idx] == distiller.teacher_tokenizer.eos_token_id:
                cur_t_preds_as_label_2[_s_idx] = distiller.student_tokenizer.eos_token_id
                s2t_hiddens_align[i, _t_idx + indices_t_target[0], :] = stu_v_hiddens[i, _s_idx + indices_s_target[0]]
                t2s_hiddens_align[i, _s_idx + indices_s_target[0], :] = tea_v_hiddens[i, _t_idx + indices_t_target[0]]
            else:
                try:
                    tmp = distiller.student_tokenizer.convert_tokens_to_ids(tmp_t_token)
                    if len(tmp) == 1 and tmp[0] is not None:
                        cur_t_preds_as_label_2[_s_idx] = tmp[0]
                        s2t_hiddens_align[i, _t_idx + indices_t_target[0], :] = stu_v_hiddens[i, _s_idx + indices_s_target[0]]
                        t2s_hiddens_align[i, _s_idx + indices_s_target[0], :] = tea_v_hiddens[i, _t_idx + indices_t_target[0]]
                    else:
                        cur_t_preds_as_label_2[_s_idx] = -100
                except Exception:
                    cur_t_preds_as_label_2[_s_idx] = -100

        assert len(cur_t_preds_as_label_2) == len(cur_s_target)
        cur_t_preds_as_label_3 = target[i][indices_s_target[-1] + 1:].cpu().tolist()

        cur_t_preds_as_label_full = cur_t_preds_as_label_1 + cur_t_preds_as_label_2 + cur_t_preds_as_label_3
        cur_t_preds_as_label_full = cur_t_preds_as_label_full[:teacher_target.shape[1]]
        t_preds_as_label.append(torch.tensor(cur_t_preds_as_label_full, device=teacher_target.device))

    t_preds_as_label_tensor = torch.cat(t_preds_as_label, dim=0).reshape(-1, teacher_target.shape[1])
    log = {"align_ratio": torch.tensor(align_ratio, device=teacher_target.device).mean()}

    t2s_logits = t2s_hiddens_align.matmul(
        distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
    )

    stu_align_token_num = max(1e-3, float(t_preds_as_label_tensor.ne(-100).sum()))

    t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds_as_label_tensor)
    stu_agreement_num = max(1e-3, float(t2s_agreement_mask.sum()))

    t2s_agreement = (t2s_agreement_mask * t_preds_as_label_tensor.ne(-100)).sum() / stu_align_token_num
    t2s_agreement_ratio = t2s_agreement_mask.sum() / pad_mask.sum().clamp(min=1)
    log["t2s_agreement"] = t2s_agreement
    log["t2s_agreement_ratio"] = t2s_agreement_ratio

    t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
    t2s_acc = (t2s_acc_mask * pad_mask).sum() / pad_mask.sum().clamp(min=1)
    t2s_acc_ratio = t2s_acc_mask.sum() / pad_mask.sum().clamp(min=1)
    log["t2s_acc"] = t2s_acc
    log["t2s_acc_ratio"] = t2s_acc_ratio
    log["t_preds_as_label_acc"] = (t_preds_as_label_tensor.eq(target) * t_preds_as_label_tensor.ne(-100)).sum() / stu_align_token_num

    t2s_ce_loss = compute_cross_entropy_loss(
        t2s_logits, t_preds_as_label_tensor, padding_id=padding_id, reduction="sum"
    )[0] / stu_align_token_num
    t2s_kd_loss = div_ref.dist_func(
        outputs.logits, t2s_logits.detach(), target, reduction="none"
    )
    if t2s_agreement <= distiller.args.t2s_agreement:
        t2s_kd_loss = (t2s_kd_loss * t2s_agreement_mask).sum() / stu_agreement_num
    else:
        t2s_kd_loss = (t2s_kd_loss * t_preds_as_label_tensor.ne(-100)).sum() / stu_align_token_num

    log["t2s_ce_loss"] = t2s_ce_loss

    s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens_align)
    s2t_kd_loss = div_ref.dist_func(
        s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
    )
    non_zero_rows = ~s2t_hiddens_align.eq(0).all(-1)
    s2t_kd_loss = (s2t_kd_loss * non_zero_rows).sum() / max(float(non_zero_rows.sum()), 1e-8)

    if getattr(args, "only_stu_kd", False):
        kd_loss = t2s_kd_loss + t2s_ce_loss
    elif getattr(args, "only_tea_kd", False):
        kd_loss = s2t_kd_loss
    else:
        kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss

    log["t2s_kd_loss"] = t2s_kd_loss
    log["s2t_kd_loss"] = s2t_kd_loss
    log["kd_loss"] = kd_loss

    return kd_loss, log
