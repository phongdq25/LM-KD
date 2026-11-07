"""DistiLLM-2 Criterion for same-tokenizer contrastive LLM distillation.

Paper: Ko et al. "DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs", ICML 2025 (Oral).
"""

from __future__ import annotations

from typing import Any

import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.functional import skewed_forward_kl, skewed_reverse_kl
from lmkd.losses.functional.sequence import pairwise
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


def compute_dynamic_alpha(
    diff: torch.Tensor,
    base_alpha: float = 0.1,
    anchor: float | None = None,
    min_alpha: float = 1e-2,
) -> torch.Tensor | float:
    """Compute dynamic sentence-level alpha for DistiLLM-2 (distillm_trainer.py lines 1150-1156)."""
    if anchor is None:
        return base_alpha
    anchor_val = (1.0 - base_alpha) * anchor
    alpha = torch.clip(1.0 - anchor_val / (diff + 1e-5), min=min_alpha, max=base_alpha)
    return alpha.unsqueeze(-1).unsqueeze(-1)


@CRITERIONS.register(
    "distillm2",
    paper="DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs (ICML 2025 Oral)",
)
class DistiLLM2Criterion(Criterion):
    """DistiLLM-2 contrastive knowledge distillation criterion.

    Pits student against teacher contrastively on paired responses:
      - Chosen responses (y^+ ~ teacher): Skewed Forward KL towards teacher
      - Rejected responses (y^- ~ student): Skewed Reverse KL towards teacher
      Total Loss: L = (2 - beta) * L_chosen + beta * L_rejected
    """

    needs_teacher = True
    supports_packing = False

    class Config(BaseConfig):
        beta: float = 1.0
        base_alpha_1: float = 0.1
        base_alpha_2: float = 0.1
        update_alpha: bool = False
        gradual_beta: bool = False
        max_steps: int | None = None
        average_log_prob: bool = True
        reduction: str = "mean"
        chunk_size: int = 1024

    def __init__(self, cfg: BaseConfig | None = None):
        super().__init__(cfg)
        self.logp_logq: float | None = None
        self.logq_logp: float | None = None

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="full"),
        )

    def extra_state(self) -> dict[str, Any]:
        return {"logp_logq": self.logp_logq, "logq_logp": self.logq_logp}

    def load_extra_state(self, state: dict[str, Any]) -> None:
        self.logp_logq = state.get("logp_logq")
        self.logq_logp = state.get("logq_logp")

    def _compute_beta(self, ctx: LossContext) -> float:
        if not self.cfg.gradual_beta:
            return float(self.cfg.beta)
        step = getattr(ctx.state, "step", 0)
        total_steps = self.cfg.max_steps or getattr(ctx.state, "total_steps", 1000)
        return 1.0 + 0.5 * min(1.0, 2.0 * step / max(1, total_steps))

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.logits is None:
            raise ValueError("DistiLLM-2 requires online teacher logits (teacher.provider: hf_frozen | self)")

        labels = ctx.batch["student.labels"]
        B, T = labels.shape
        if B % 2 != 0:
            raise ValueError(f"DistiLLM-2 requires pairwise input with even batch size, got {B}")

        mask = ctx.supervised_mask()
        s_view = ctx.student_logits()
        t_view = ctx.teacher.logits
        device = labels.device

        c_mask, r_mask = pairwise(mask)
        c_tok_count = c_mask.sum(-1).clamp_min(1.0)
        r_tok_count = r_mask.sum(-1).clamp_min(1.0)

        # Sentence-level probabilities for dynamic alpha and logging
        s_tok_logps = s_view.target_logp(labels, chunk_size=self.cfg.chunk_size)
        t_tok_logps = t_view.target_logp(labels, chunk_size=self.cfg.chunk_size)

        s_c_tok, s_r_tok = pairwise(s_tok_logps)
        t_c_tok, t_r_tok = pairwise(t_tok_logps)

        s_c_seq_prob = ((s_c_tok * c_mask).sum(-1) / c_tok_count).exp()
        t_c_seq_prob = ((t_c_tok * c_mask).sum(-1) / c_tok_count).exp()
        s_r_seq_prob = ((s_r_tok * r_mask).sum(-1) / r_tok_count).exp()
        t_r_seq_prob = ((t_r_tok * r_mask).sum(-1) / r_tok_count).exp()

        logps_logqs_chosen = t_c_seq_prob - s_c_seq_prob
        logqs_logps_rejected = s_r_seq_prob - t_r_seq_prob

        # Dynamic alpha calculation
        if self.cfg.update_alpha and self.logp_logq is not None:
            alpha_1 = compute_dynamic_alpha(logps_logqs_chosen, self.cfg.base_alpha_1, self.logp_logq)
        else:
            alpha_1 = self.cfg.base_alpha_1

        if self.cfg.update_alpha and self.logq_logp is not None:
            alpha_2 = compute_dynamic_alpha(logqs_logps_rejected, self.cfg.base_alpha_2, self.logq_logp)
        else:
            alpha_2 = self.cfg.base_alpha_2

        # Update running anchors
        if self.cfg.update_alpha:
            step_logp_logq = float(logps_logqs_chosen.detach().mean().cpu())
            step_logq_logp = float(logqs_logps_rejected.detach().mean().cpu())
            self.logp_logq = (
                step_logp_logq if self.logp_logq is None else 0.9 * self.logp_logq + 0.1 * step_logp_logq
            )
            self.logq_logp = (
                step_logq_logp if self.logq_logp is None else 0.9 * self.logq_logp + 0.1 * step_logq_logp
            )

        beta_val = self._compute_beta(ctx)

        # Chunked KL divergence over sequence length T
        B_pairs = B // 2
        c_kl_sum = torch.zeros(B_pairs, device=device, dtype=torch.float32)
        r_kl_sum = torch.zeros(B_pairs, device=device, dtype=torch.float32)

        for s in range(0, T, self.cfg.chunk_size):
            e = min(T, s + self.cfg.chunk_size)
            c_m_ch = c_mask[:, s:e]
            r_m_ch = r_mask[:, s:e]

            if not (c_m_ch.any() or r_m_ch.any()):
                continue

            s_c_sl, s_r_sl = pairwise(s_view._target_slice(s, e))
            with torch.no_grad():
                t_c_sl, t_r_sl = pairwise(t_view._target_slice(s, e))

            V = min(s_c_sl.shape[-1], t_c_sl.shape[-1])
            s_c_sl, t_c_sl = s_c_sl[..., :V], t_c_sl[..., :V]
            s_r_sl, t_r_sl = s_r_sl[..., :V], t_r_sl[..., :V]

            if c_m_ch.any():
                c_kl_sum = c_kl_sum + (skewed_forward_kl(s_c_sl, t_c_sl, alpha=alpha_1) * c_m_ch).sum(-1)
            if r_m_ch.any():
                r_kl_sum = r_kl_sum + (skewed_reverse_kl(s_r_sl, t_r_sl, alpha=alpha_2) * r_m_ch).sum(-1)

        chosen_kl = c_kl_sum / c_tok_count if self.cfg.average_log_prob else c_kl_sum
        rejected_kl = r_kl_sum / r_tok_count if self.cfg.average_log_prob else r_kl_sum
        losses = (2.0 - beta_val) * chosen_kl + beta_val * rejected_kl

        if self.cfg.reduction == "mean":
            term = LossTerm(losses.mean(), "mean")
        elif self.cfg.reduction == "seq_sum":
            term = LossTerm(losses.sum() * 2, "seq_sum")
        else:
            term = LossTerm(losses.sum(), "token_sum")

        metrics = {
            "loss": losses.mean().detach(),
            "loss/chosen": chosen_kl.mean().detach(),
            "loss/rejected": rejected_kl.mean().detach(),
            "logqs_logps/rejected": logqs_logps_rejected.mean().detach(),
            "logps_logqs/chosen": logps_logqs_chosen.mean().detach(),
            "beta": torch.tensor(beta_val, device=device),
        }
        return LossOutput(terms={self.alias: term}, metrics=metrics)


DistiLLM2 = DistiLLM2Criterion
