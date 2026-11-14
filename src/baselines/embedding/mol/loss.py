"""MoL (Mixture of Layers in Cross-Tokenizer embedding model distillation) baseline.

Paper: MoL: Mixture of Layers in Cross-Tokenizer embedding model distillation (Knowledge-Based Systems 2026)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo, PrepareContext
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.modules import ModuleRegistry


class LinearCKA(nn.Module):
    """Linear Centered Kernel Alignment between student and teacher representations.

    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        student_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculate Linear CKA similarity between student and teacher hidden states.

        student_h: [B, Ls, Ds]
        teacher_h: [B, Ls, Dt]  # Aligned to student length Ls
        student_mask: [B, Ls], 1 = valid token

        return: scalar Linear CKA in [0, 1]
        """
        if student_mask is not None:
            valid = student_mask.bool().reshape(-1)
            student_h = student_h.reshape(-1, student_h.size(-1))[valid]
            teacher_h = teacher_h.reshape(-1, teacher_h.size(-1))[valid]
        else:
            student_h = student_h.reshape(-1, student_h.size(-1))
            teacher_h = teacher_h.reshape(-1, teacher_h.size(-1))

        if student_h.size(0) <= 1:
            return torch.tensor(0.0, device=student_h.device, dtype=torch.float32)

        student_h = student_h.float()
        teacher_h = teacher_h.float()

        # Center over observations
        student_h = student_h - student_h.mean(dim=0, keepdim=True)
        teacher_h = teacher_h - teacher_h.mean(dim=0, keepdim=True)

        cross = torch.norm(student_h.T @ teacher_h, p="fro")
        self_sim_s = torch.norm(student_h.T @ student_h, p="fro")
        self_sim_t = torch.norm(teacher_h.T @ teacher_h, p="fro")

        return cross.square() / (self_sim_s * self_sim_t + self.eps)


def align_teacher_to_student(
    query_projector: nn.Module,
    student_embedding: torch.Tensor,
    student_last_hidden: torch.Tensor,
    teacher_embedding: torch.Tensor,
    teacher_last_hidden: torch.Tensor,
    teacher_layer_hidden: torch.Tensor,
    teacher_attention_mask: torch.Tensor | None = None,
    teacher_hidden_size: int | None = None,
) -> torch.Tensor:
    """Align teacher layer hidden states to student sequence length via cross-attention.


    student_embedding:      [B, Ls, Ds]
    student_last_hidden:    [B, Ls, Ds]
    teacher_embedding:      [B, Lt, Dt]
    teacher_last_hidden:    [B, Lt, Dt]
    teacher_layer_hidden:   [B, Lt, Dt]
    teacher_attention_mask: [B, Lt]

    return:
        aligned_teacher: [B, Ls, Dt]
    """
    # Q = Proj([S^(0), S^(-1)])
    student_repr = torch.cat([student_embedding, student_last_hidden], dim=-1)
    student_query = query_projector(student_repr).float()

    # K = Norm([T^(0), T^(-1)])
    teacher_repr = torch.cat([teacher_embedding, teacher_last_hidden], dim=-1)
    teacher_key = teacher_repr.float()

    # V^(l) = Norm(T^(l))
    teacher_value = teacher_layer_hidden.float()

    # Normalize across hidden dimension
    teacher_key = teacher_key / teacher_key.std(dim=-1, keepdim=True).clamp_min(1e-6)
    teacher_value = teacher_value / teacher_value.std(dim=-1, keepdim=True).clamp_min(1e-6)

    # A = softmax(QK^T / sqrt(D))
    d_scale = math.sqrt(teacher_hidden_size if teacher_hidden_size is not None else teacher_layer_hidden.size(-1))
    attn_scores = (student_query @ teacher_key.transpose(-1, -2)) / d_scale

    # Mask teacher padding
    if teacher_attention_mask is not None:
        teacher_pad_mask = ~teacher_attention_mask.bool()
        if teacher_pad_mask.ndim == 2:
            attn_scores = attn_scores.masked_fill(teacher_pad_mask.unsqueeze(1), -1e4)

    attn_weights = torch.softmax(attn_scores, dim=-1)

    # T_tilde^(l) = A V^(l)
    return attn_weights @ teacher_value


def mol_kd_loss(
    query_projector: nn.Module,
    student_hidden_states: Sequence[torch.Tensor],
    teacher_hidden_states: Sequence[torch.Tensor],
    student_attention_mask: torch.Tensor | None = None,
    teacher_attention_mask: torch.Tensor | None = None,
    student_layers: Sequence[int] | None = None,
    teacher_window: int = 2,
    teacher_hidden_size: int | None = None,
    eps: float = 1e-8,
    detach_layer_weights: bool = False,
    cka_fn: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute MoL (Mixture of Layers) distillation loss.

    Paper: MoL: Mixture of Layers in Cross-Tokenizer embedding model distillation
    """
    device_s = student_hidden_states[-1].device
    dtype_s = student_hidden_states[-1].dtype

    cka = cka_fn if cka_fn is not None else LinearCKA(eps=eps)

    teacher_num_layers = len(teacher_hidden_states) - 1
    student_num_layers = len(student_hidden_states) - 1

    student_embedding = student_hidden_states[0]
    student_last = student_hidden_states[-1]

    teacher_embedding = teacher_hidden_states[0].to(device_s, non_blocking=True)
    teacher_last = teacher_hidden_states[-1].to(device_s, non_blocking=True)

    if teacher_attention_mask is not None:
        teacher_attention_mask = teacher_attention_mask.to(device_s, non_blocking=True)

    if student_layers is None:
        resolved_s_layers = list(range(max(1, student_num_layers - 1), student_num_layers + 1))
    else:
        resolved_s_layers = [l if l > 0 else l + student_num_layers + 1 for l in student_layers]

    total_kd_loss = student_last.new_zeros((), dtype=torch.float32)
    num_pairs = 0
    layer_metrics: dict[str, torch.Tensor] = {}

    for s_layer in resolved_s_layers:
        if s_layer <= 0 or s_layer >= len(student_hidden_states):
            continue

        center_t_layer = round(s_layer * teacher_num_layers / student_num_layers)
        candidate_t_layers = range(
            max(1, center_t_layer - teacher_window),
            min(teacher_num_layers, center_t_layer + teacher_window) + 1,
        )

        aligned_teacher_candidates = []
        similarities = []

        for t_layer in candidate_t_layers:
            teacher_layer = teacher_hidden_states[t_layer].to(device_s, non_blocking=True)

            aligned_teacher = align_teacher_to_student(
                query_projector=query_projector,
                student_embedding=student_embedding,
                student_last_hidden=student_last,
                teacher_embedding=teacher_embedding,
                teacher_last_hidden=teacher_last,
                teacher_layer_hidden=teacher_layer,
                teacher_attention_mask=teacher_attention_mask,
                teacher_hidden_size=teacher_hidden_size,
            )

            cka_sim = cka(
                student_hidden_states[s_layer],
                aligned_teacher,
                student_mask=student_attention_mask,
            )
            aligned_teacher_candidates.append(aligned_teacher)
            similarities.append(cka_sim)

        if not similarities:
            continue

        cka_scores = torch.stack(similarities)
        cka_scores = cka_scores.clamp(min=0.0, max=1.0)
        # Eq. (18): c_l = sqrt(CKA_l)
        scores = torch.sqrt(cka_scores)
        # Eq. (19): w_l = softmax(c_l)
        sim_weights = torch.softmax(scores, dim=0)
        if detach_layer_weights:
            sim_weights = sim_weights.detach()

        weighted_teacher = sum(weight * hidden for weight, hidden in zip(sim_weights, aligned_teacher_candidates))

        final_cka = cka(
            student_hidden_states[s_layer],
            weighted_teacher,
            student_mask=student_attention_mask,
        )

        total_kd_loss = total_kd_loss + (1.0 - final_cka)
        layer_metrics[f"cka_layer_{s_layer}"] = final_cka.detach()
        num_pairs += 1

    kd_loss = total_kd_loss / max(num_pairs, 1)

    logs = {
        "cka_kd_loss": kd_loss.detach(),
        "cka_num_pairs": torch.tensor(num_pairs, device=device_s),
        **layer_metrics,
    }
    return kd_loss.to(dtype_s), logs


@CRITERIONS.register(
    "mol",
    paper="MoL: Mixture of Layers in Cross-Tokenizer embedding model distillation",
)
class MoL(Criterion):
    """MoL: Mixture of Layers in Cross-Tokenizer embedding model distillation.

    Paper: MoL: Mixture of Layers in Cross-Tokenizer embedding model distillation (Knowledge-Based Systems 2026)
    """

    supports_packing = False
    needs_teacher = True

    class Config(BaseConfig):
        student_layers: tuple[int, ...] = (-2, -1)
        teacher_window: int = 2
        teacher_dim: int | None = None
        eps: float = 1e-8
        detach_layer_weights: bool = False
        proj_init: str | None = None
        projector_ckpt: str = ""

    cfg: Config

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        d_s = info.student_hidden
        d_t = self.cfg.teacher_dim or info.teacher_hidden
        if d_t is None:
            raise ValueError(f"{self.alias} requires teacher_dim to be set or teacher model configured")
        self._student_layers = info.student_layers
        self._teacher_layers = info.teacher_layers
        self._d_t = d_t

        proj = nn.Linear(2 * d_s, 2 * d_t, bias=False)
        self.register_aux_module(modules, "query_proj", proj, init=self.cfg.proj_init, group="projector")

    def prepare(self, ctx: PrepareContext | None) -> None:
        if self.cfg.projector_ckpt:
            self.load_aux_checkpoint(ctx, "query_proj", allow_raw=False)

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(hidden="all", n_views=1),
            teacher=SideRequest(hidden="all", n_views=1),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.side is None:
            raise ValueError(f"{self.alias} requires teacher side outputs (teacher.provider: hf_frozen)")

        proj = self.aux(ctx).query_proj

        s_side = ctx.student
        t_side = ctx.teacher.side

        student_hidden_states = [s_side.hidden(i) for i in range(s_side.num_layers + 1)]
        teacher_hidden_states = [t_side.hidden(i) for i in range(t_side.num_layers + 1)]

        student_attention_mask = ctx.batch.get("student.attention_mask", s_side.attention_mask)
        teacher_attention_mask = ctx.batch.get("teacher.attention_mask", t_side.attention_mask)

        teacher_dim = self.cfg.teacher_dim or getattr(self, "_d_t", None) or t_side.last_hidden.size(-1)

        loss, logs = mol_kd_loss(
            query_projector=proj,
            student_hidden_states=student_hidden_states,
            teacher_hidden_states=teacher_hidden_states,
            student_attention_mask=student_attention_mask,
            teacher_attention_mask=teacher_attention_mask,
            student_layers=self.cfg.student_layers,
            teacher_window=self.cfg.teacher_window,
            teacher_hidden_size=teacher_dim,
            eps=self.cfg.eps,
            detach_layer_weights=self.cfg.detach_layer_weights,
        )

        metrics = {
            f"{self.alias}/cka_loss": logs["cka_kd_loss"],
            f"{self.alias}/num_pairs": logs["cka_num_pairs"],
        }
        for k, v in logs.items():
            if k.startswith("cka_layer_"):
                metrics[f"{self.alias}/{k}"] = v

        return LossOutput(terms={self.alias: LossTerm(loss, "mean")}, metrics=metrics)


@CRITERIONS.register(
    "mol_kd",
    paper="MoL: Mixture of Layers in Cross-Tokenizer embedding model distillation",
)
class MoLKD(MoL):
    """Alias for MoL criterion."""
