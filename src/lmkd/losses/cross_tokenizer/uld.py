"""Universal Logit Distillation (ULD) criterion (T13).

Source: white-box-distillation/cross-tokenizer/IMPACT/code/criterions/ULD.py:61-113.
"""
from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def uld_loss(
    s_rows: torch.Tensor, t_rows: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Pairwise ULD loss: L1 distance between sorted probability vectors.

    Args:
        s_rows: Student logits [N, V_s].
        t_rows: Teacher logits [N, V_t].
        temperature: Softmax temperature.

    Returns:
        Tensor of shape [N] containing per-pair L1 distance.
    """
    ps = torch.softmax(s_rows.float() / temperature, dim=-1)
    pt = torch.softmax(t_rows.float() / temperature, dim=-1)
    ps_sorted = ps.sort(dim=-1, descending=True).values
    pt_sorted = pt.sort(dim=-1, descending=True).values

    v_s = ps_sorted.shape[-1]
    v_t = pt_sorted.shape[-1]
    v_max = max(v_s, v_t)
    if v_s < v_max:
        ps_sorted = F.pad(ps_sorted, (0, v_max - v_s), value=0.0)
    elif v_t < v_max:
        pt_sorted = F.pad(pt_sorted, (0, v_max - v_t), value=0.0)

    return (ps_sorted - pt_sorted).abs().sum(dim=-1)


def _find_prompt_len(plen_list: list[int] | None, lab_row: torch.Tensor, b: int) -> int:
    if plen_list is not None:
        plen = int(plen_list[b])
        if plen < lab_row.shape[0] and lab_row[plen] == -100:
            nz = (lab_row != -100).nonzero(as_tuple=False)
            if nz.numel() > 0:
                return int(nz[0].item())
        return plen
    nz = (lab_row != -100).nonzero(as_tuple=False)
    return int(nz[0].item()) if nz.numel() > 0 else 0


def _collect_uld_pairs(
    batch: Any,
    s_lab: torch.Tensor,
    t_lab: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract corresponding target token pairs from sequence start."""
    s_pairs: list[tuple[int, int]] = []
    t_pairs: list[tuple[int, int]] = []
    dropped = 0
    b_size = s_lab.shape[0]

    s_lab_cpu = s_lab.cpu()
    t_lab_cpu = t_lab.cpu()
    s_plen = batch["student.prompt_len"].cpu().tolist() if "student.prompt_len" in batch else None
    t_plen = batch["teacher.prompt_len"].cpu().tolist() if "teacher.prompt_len" in batch else None

    for b in range(b_size):
        plen_s = _find_prompt_len(s_plen, s_lab_cpu[b], b)
        plen_t = _find_prompt_len(t_plen, t_lab_cpu[b], b)

        r_s = int((s_lab_cpu[b] != -100).sum().item())
        r_t = int((t_lab_cpu[b] != -100).sum().item())
        k_b = min(r_s, r_t)

        for k in range(k_b):
            pos_s = plen_s + k
            pos_t = plen_t + k
            if (
                pos_s < s_lab.shape[1]
                and pos_t < t_lab.shape[1]
                and s_lab[b, pos_s] != -100
                and t_lab[b, pos_t] != -100
            ):
                s_pairs.append((b, pos_s))
                t_pairs.append((b, pos_t))
            else:
                dropped += 1

    s_t = (
        torch.tensor(s_pairs, dtype=torch.long, device=device)
        if s_pairs
        else torch.zeros((0, 2), dtype=torch.long, device=device)
    )
    t_t = (
        torch.tensor(t_pairs, dtype=torch.long, device=device)
        if t_pairs
        else torch.zeros((0, 2), dtype=torch.long, device=device)
    )
    return s_t, t_t, dropped


@CRITERIONS.register("uld", source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/ULD.py:61-113")
class ULD(Criterion):
    """Universal logit distillation (ULD).

    Formula:
        loss = sum_b sum_k |sort_desc(softmax(z_S[b, plen_S + k] / T)) - sort_desc(softmax(z_T[b, plen_T + k] / T))|_1
        where k in [0, min(R_S, R_T)) over valid (non-masked) response token pairs.

    Original: IMPACT criterions/ULD.py:61-113.
    """

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        temperature: float = 1.0
        pair_chunk: int = 512

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="full"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        s_lab = ctx.batch["student.labels"]
        t_lab = ctx.batch["teacher.labels"]
        device = s_lab.device
        s_pos, t_pos, dropped = _collect_uld_pairs(ctx.batch, s_lab, t_lab, device)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_pairs = s_pos.shape[0]
        if n_pairs > 0:
            assert ctx.student.logits is not None
            assert ctx.teacher is not None and ctx.teacher.logits is not None
            chunk_size = max(self.cfg.pair_chunk, 1)
            for i in range(0, n_pairs, chunk_size):
                s_chunk = s_pos[i : i + chunk_size]
                t_chunk = t_pos[i : i + chunk_size]
                s_rows = ctx.student.logits.at(s_chunk)
                with torch.no_grad():
                    t_rows = ctx.teacher.logits.at(t_chunk)
                d = uld_loss(s_rows, t_rows, temperature=self.cfg.temperature)
                total_loss = total_loss + d.sum()

        return LossOutput(
            terms={self.alias: LossTerm(total_loss, "token_sum")},
            metrics={"uld_dropped": torch.tensor(float(dropped), device=device)},
        )
