"""MinED: DTW-aligned cross-tokenizer KD with vocab mapping and one-hot fallback (T13).

Source: white-box-distillation/cross-tokenizer/IMPACT/code/criterions/MinED.py:22-210.
"""
from __future__ import annotations

from typing import ClassVar

import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, PrepareContext
from lmkd.losses.cross_tokenizer._common import get_vocab_map, vd_divergence
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def _compute_mined_chunk(
    ctx: LossContext,
    pos_chunk: torch.Tensor,
    s_lab: torch.Tensor,
    dtw_t: torch.Tensor,
    map_s: torch.Tensor,
    map_t: torch.Tensor,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Compute MinED loss for one chunk of target student positions."""
    b = pos_chunk[:, 0]
    t = pos_chunk[:, 1]
    m = dtw_t[b, t]
    aligned = m >= 0
    assert ctx.student.logits is not None
    s_rows = ctx.student.logits.at(pos_chunk)
    c_len, v_s = s_rows.shape

    b2b = torch.full((c_len, v_s), -100000.0, device=device, dtype=torch.float32)
    aligned_idx = aligned.nonzero(as_tuple=False).squeeze(-1)
    if aligned_idx.numel() > 0:
        pos_t_aligned = torch.stack([b[aligned_idx], m[aligned_idx]], dim=-1)
        assert ctx.teacher is not None and ctx.teacher.logits is not None
        with torch.no_grad():
            t_rows = ctx.teacher.logits.at(pos_t_aligned)
        b2b[aligned_idx[:, None], map_s] = t_rows[:, map_t]
        aligned = aligned & (b2b.max(dim=-1).values != -100000.0)

    s_lab_chunk = s_lab[b, t]
    onehot = torch.full((c_len, v_s), -100000.0, device=device, dtype=torch.float32)
    onehot.scatter_(1, s_lab_chunk.unsqueeze(-1), 100.0)

    tea = torch.where(aligned.unsqueeze(-1), b2b, onehot)
    kd_chunk = vd_divergence("forward_kl", s_rows, tea, kd_temp=temperature)
    return kd_chunk.sum(), int(aligned.sum().item())


@CRITERIONS.register(
    "mined_fkl",
    source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/MinED.py:22-210",
)
class MinEDFKL(Criterion):
    """MinED: FKL on DTW-aligned positions with vocab-mapped teacher logits.

    Formula:
        Align sequence context tokens via DTW with edit-distance cost.
        Map teacher logits to student vocabulary via stu2tea token mapping.
        Unaligned / unmapped positions use ground-truth one-hot pseudo-logits.
        kd = Forward_KL(student_logits, mapped_teacher_logits; T)

    Original: IMPACT criterions/MinED.py:22-210.
    """

    needs_teacher: ClassVar[bool] = True
    requires: ClassVar[tuple[str, ...]] = ("align.dtw_t_for_s",)

    class Config(BaseConfig):
        temperature: float = 1.0
        pair_chunk: int = 512

    def __init__(self, cfg: BaseConfig | None = None) -> None:
        super().__init__(cfg)
        self.map_s: torch.Tensor | None = None
        self.map_t: torch.Tensor | None = None

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="full"),
        )

    def prepare(self, ctx: PrepareContext) -> None:
        vm = getattr(getattr(getattr(ctx, "rt", None), "data_ctx", None), "vocab_map", None)
        if vm is not None and getattr(vm, "stu2tea", None) is not None:
            self._init_mapping(vm.stu2tea)

    def _init_mapping(self, stu2tea: torch.Tensor) -> None:
        valid = stu2tea >= 0
        self.map_s = valid.nonzero(as_tuple=False).squeeze(-1)
        self.map_t = stu2tea[self.map_s]

    def forward(self, ctx: LossContext) -> LossOutput:
        device = ctx.batch["student.labels"].device
        if self.map_s is None or self.map_t is None:
            vm = get_vocab_map(ctx)
            self._init_mapping(vm.stu2tea)
        assert self.map_s is not None and self.map_t is not None
        map_s = self.map_s.to(device)
        map_t = self.map_t.to(device)

        s_lab = ctx.batch["student.labels"]
        dtw_t = ctx.batch["align.dtw_t_for_s"]
        pos_s = (s_lab != -100).nonzero(as_tuple=False)

        total_n = pos_s.shape[0]
        if total_n == 0:
            return LossOutput(
                terms={self.alias: LossTerm(torch.tensor(0.0, device=device, requires_grad=True), "token_sum")},
                metrics={"align_ratio": torch.tensor(0.0, device=device)},
            )

        total_kd = torch.tensor(0.0, device=device, requires_grad=True)
        total_aligned = torch.tensor(0, device=device)
        chunk_size = max(self.cfg.pair_chunk, 1)

        for i in range(0, total_n, chunk_size):
            pos_chunk = pos_s[i : i + chunk_size]
            kd_val, n_aligned = _compute_mined_chunk(
                ctx, pos_chunk, s_lab, dtw_t, map_s, map_t, self.cfg.temperature, device
            )
            total_kd = total_kd + kd_val
            total_aligned = total_aligned + n_aligned

        align_ratio = (total_aligned.float() / float(total_n)).to(device)
        return LossOutput(
            terms={self.alias: LossTerm(total_kd, "token_sum")},
            metrics={"align_ratio": align_ratio},
        )
