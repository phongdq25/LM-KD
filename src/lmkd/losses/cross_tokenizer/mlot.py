"""Multi-Level Optimal Transport (MultiLevelOT) criterion (T13).

Source: white-box-distillation/cross-tokenizer/IMPACT/code/criterions/MultiLevelOT.py:141-204.
"""
from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, Reduction
from lmkd.losses.cross_tokenizer.uld import _collect_uld_pairs
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def mlot_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize logits by std without subtracting mean (IMPACT MultiLevelOT.py:7-11)."""
    return x / (x.std(dim=-1, keepdim=True) + 1e-4)


def mlot_sinkhorn(
    ps_b: torch.Tensor,
    pt_b: torch.Tensor,
    eps: float = 0.1,
    iters: int = 10,
    ot_scale: float = 0.001,
    sinkhorn_temp: float = 2.0,
) -> torch.Tensor:
    """Sinkhorn distance between sample sequence distributions (MultiLevelOT.py:25-50)."""
    p_s = torch.softmax(ps_b / sinkhorn_temp, dim=-1)
    p_t = torch.softmax(pt_b / sinkhorn_temp, dim=-1)
    W = torch.cdist(p_s, p_t, p=1)
    K = torch.exp(-W / eps)
    P = K
    for _ in range(iters):
        P = P / P.sum(dim=1, keepdim=True)
        P = P / P.sum(dim=0, keepdim=True)
    return ot_scale * (P * W).sum()


def mlot_kl_wo(ps: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
    """KL_wo from MultiLevelOT.py:19-23 applying log_softmax to already softmaxed probs."""
    p_s = F.log_softmax(ps, dim=-1)
    p_t = F.softmax(pt, dim=-1)
    return -torch.sum(p_t * p_s, dim=-1).mean()


def _compute_perm_sums(
    ctx: LossContext,
    s_pos: torch.Tensor,
    t_pos: torch.Tensor,
    temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass 1: Accumulate softmaxed normalized distributions across batch without grad."""
    n_pairs = s_pos.shape[0]
    accum_s: torch.Tensor | None = None
    accum_t: torch.Tensor | None = None

    with torch.no_grad():
        for i in range(0, n_pairs, chunk_size):
            s_c = s_pos[i : i + chunk_size]
            t_c = t_pos[i : i + chunk_size]
            assert ctx.student.logits is not None
            s_chunk = ctx.student.logits.at(s_c)
            assert ctx.teacher is not None and ctx.teacher.logits is not None
            t_chunk = ctx.teacher.logits.at(t_c)

            zs = mlot_normalize(s_chunk)
            zt = mlot_normalize(t_chunk)
            ps_c = torch.softmax(zs / temperature, dim=-1)
            pt_c = torch.softmax(zt / temperature, dim=-1)

            sum_s = ps_c.sum(dim=0)
            sum_t = pt_c.sum(dim=0)
            accum_s = sum_s if accum_s is None else accum_s + sum_s
            accum_t = sum_t if accum_t is None else accum_t + sum_t

    assert accum_s is not None and accum_t is not None
    return accum_s, accum_t


@CRITERIONS.register(
    "multilevel_ot",
    source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/MultiLevelOT.py:141-204",
)
class MultiLevelOT(Criterion):
    """Multi-level Optimal Transport (MLOT).

    Computes:
      - Sinkhorn distance on top-k sorted normalized probabilities (improved_sort across batch)
      - KL_wo: cross-entropy like term on probabilities
      - L1 distance on sorted probabilities

    Note on differences from original (see DESIGN Appendix A):
      Original ran Sinkhorn on full rolled+padded sequences, mixing prompt and padding tokens.
      This implementation only uses valid response tokens.

    Original: IMPACT criterions/MultiLevelOT.py:141-204.
    """

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        temperature: float = 2.0
        top_k: int = 50
        sinkhorn_eps: float = 0.1
        sinkhorn_iters: int = 10
        sinkhorn_temp: float = 2.0
        f: int = 1
        ot_scale: float = 0.001
        ot_weight: float = 0.1
        kl_weight: float = 0.1
        legacy_token_scale: bool = False
        pair_chunk: int = 512

    def __init__(self, cfg: BaseConfig | None = None) -> None:
        super().__init__(cfg)
        if self.cfg.f != 1:
            raise NotImplementedError(f"MultiLevelOT only supports f=1, got f={self.cfg.f}")

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
        s_pos, t_pos, _ = _collect_uld_pairs(ctx.batch, s_lab, t_lab, device)
        n_pairs = s_pos.shape[0]

        if n_pairs == 0:
            red: Reduction = "token_sum" if self.cfg.legacy_token_scale else "mean"
            return LossOutput(terms={self.alias: LossTerm(torch.tensor(0.0, device=device, requires_grad=True), red)})

        chunk_size = max(self.cfg.pair_chunk, 1)
        sum_s, sum_t = _compute_perm_sums(ctx, s_pos, t_pos, self.cfg.temperature, chunk_size)
        perm_s = torch.argsort(sum_s, descending=True)[: self.cfg.top_k]
        perm_t = torch.argsort(sum_t, descending=True)[: self.cfg.top_k]

        assert ctx.student.logits is not None
        s_rows = ctx.student.logits.at(s_pos)
        zs = mlot_normalize(s_rows)
        ps = torch.softmax(zs / self.cfg.temperature, dim=-1)[:, perm_s]

        assert ctx.teacher is not None and ctx.teacher.logits is not None
        with torch.no_grad():
            t_rows = ctx.teacher.logits.at(t_pos)
            zt = mlot_normalize(t_rows)
            pt = torch.softmax(zt / self.cfg.temperature, dim=-1)[:, perm_t]

        diff_size = ps.size(1) - pt.size(1)
        if diff_size > 0:
            pt = F.pad(pt, (0, diff_size), value=0.0)
        elif diff_size < 0:
            ps = F.pad(ps, (0, abs(diff_size)), value=0.0)

        ot = torch.tensor(0.0, device=device)
        b_indices = s_pos[:, 0]
        for b in range(s_lab.shape[0]):
            b_mask = b_indices == b
            if b_mask.any():
                ot = ot + mlot_sinkhorn(
                    ps[b_mask], pt[b_mask],
                    eps=self.cfg.sinkhorn_eps,
                    iters=self.cfg.sinkhorn_iters,
                    ot_scale=self.cfg.ot_scale,
                    sinkhorn_temp=self.cfg.sinkhorn_temp,
                )
        ot = ot * self.cfg.ot_weight
        kl = mlot_kl_wo(ps, pt) * self.cfg.kl_weight
        l1 = (ps - pt).abs().sum(dim=-1).mean()
        multi = ot + kl + l1

        if self.cfg.legacy_token_scale:
            term = LossTerm(multi * float(n_pairs), "token_sum")
        else:
            term = LossTerm(multi, "mean")

        return LossOutput(terms={self.alias: term})
