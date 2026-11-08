"""EMO: Embedding Model Distillation via Intra-Model Relation and Optimal Transport Alignments.

Paper: EMO: Embedding Model Distillation via Intra-Model Relation and Optimal Transport Alignments (EMNLP 2024)

Key components:
1. CKALoss: Linear Centered Kernel Alignment on token-to-token attention patterns.
2. Token alignment & reciprocal mapping based on subword edit distance (MinED).
3. Token importance mass assignment derived from attention centrality.
4. Optimal Transport (Sinkhorn) on student sequences and projected teacher sequences.
5. Multi-head intra-model attention distillation on top-k salient reciprocal tokens.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
import torch.nn as nn

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo, PrepareContext
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.modules import ModuleRegistry


class CKALoss(nn.Module):
    """Linear Centered Kernel Alignment loss.

    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, SH: torch.Tensor, TH: torch.Tensor) -> torch.Tensor:
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(SH.device, torch.float64)

        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)

        num = torch.norm(SH.t().matmul(TH), "fro")
        den1 = torch.norm(SH.t().matmul(SH), "fro") + self.eps
        den2 = torch.norm(TH.t().matmul(TH), "fro") + self.eps

        return (1 - num / torch.sqrt(den1 * den2)).to(torch.float32)


def align_tokens(
    teacher_tokens: list[str],
    student_tokens: list[str],
    teacher_special: str = "<s>",
    student_special: str = "[CLS]",
) -> dict[str, str]:
    """Align tokens between teacher and student using minimal edit distance.

    """
    teacher_to_student: dict[str, str] = {}
    if not teacher_tokens or not student_tokens:
        return teacher_to_student

    try:
        import editdistance
    except ImportError as e:
        raise ImportError("EMO requires editdistance: pip install editdistance") from e

    if teacher_special in teacher_tokens and student_special in student_tokens:
        teacher_to_student[teacher_special] = student_special

    for t in teacher_tokens:
        if t == teacher_special:
            continue
        tmp_t = t.replace(" ", "").replace("##", "")
        best_s: str | None = None
        best_dist = float("inf")
        for s in student_tokens:
            if s == student_special:
                continue
            d = editdistance.eval(tmp_t, s)
            if d < best_dist:
                best_s = s
                best_dist = d
        if best_s is not None:
            teacher_to_student[t] = best_s

    return teacher_to_student


def build_reciprocal_mapping_from_token_lists(
    teacher_tokens: list[str],
    student_tokens: list[str],
    teacher_special: str = "<s>",
    student_special: str = "[CLS]",
) -> dict[str, str]:
    """Build mutual/reciprocal token alignment (t -> s and s -> t match).

    """
    teacher_to_student = align_tokens(teacher_tokens, student_tokens, teacher_special, student_special)
    student_to_teacher = align_tokens(student_tokens, teacher_tokens, student_special, teacher_special)

    reciprocal_mapping: dict[str, str] = {}
    for t, s in teacher_to_student.items():
        if s in student_to_teacher and student_to_teacher[s] == t:
            reciprocal_mapping[t] = s
    return reciprocal_mapping


def compute_token_importance(attention_weights: torch.Tensor, tokens: list[str]) -> torch.Tensor:
    """Compute token importance based on received attention centrality.

    """
    if len(attention_weights.shape) == 3:
        avg_attention = attention_weights.mean(dim=0)
    else:
        avg_attention = attention_weights

    seq_len = min(avg_attention.shape[0], len(tokens))
    avg_attention = avg_attention[:seq_len, :seq_len]
    token_importance = avg_attention.sum(dim=0)
    return torch.softmax(token_importance, dim=0)


def project_importance(
    teacher_importance: torch.Tensor,
    teacher_tokens: list[str],
    student_tokens: list[str],
    mapping: dict[str, str],
) -> torch.Tensor:
    """Project teacher token importance vector onto student token positions.

    """
    device = teacher_importance.device
    student_importance = torch.zeros(len(student_tokens), device=device)
    valid_teacher_tokens = teacher_tokens[: teacher_importance.shape[0]]

    mapped_student_indices = set()
    for t_idx, t in enumerate(valid_teacher_tokens):
        if t in mapping:
            s = mapping[t]
            s_indices = [i for i, token in enumerate(student_tokens) if token == s]
            for s_idx in s_indices:
                if s_idx < len(student_importance):
                    student_importance[s_idx] = teacher_importance[t_idx]
                    mapped_student_indices.add(s_idx)

    min_importance = teacher_importance.min().item() if len(teacher_importance) > 0 else 0.0
    for s_idx in range(len(student_tokens)):
        if s_idx not in mapped_student_indices and s_idx < len(student_importance):
            student_importance[s_idx] = min_importance

    return torch.softmax(student_importance, dim=0)


def pairwise_attention_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise attention distance matrix between tokens: 1.0 - softmax(x y^T / sqrt(d)).

    """
    d = x.shape[1]
    sim_mt = torch.mm(x, y.transpose(0, 1)) / math.sqrt(d)
    attention_weights = torch.softmax(sim_mt, dim=1)
    return 1.0 - attention_weights


def sinkhorn(
    cost_matrix: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 0.1,
    max_iter: int = 100,
    stop_thr: float = 1e-9,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinkhorn Optimal Transport solver.

    """
    m, n = cost_matrix.shape
    device = cost_matrix.device
    dtype = cost_matrix.dtype

    if m == 0 or n == 0:
        return (
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.zeros((m, n), device=device, dtype=dtype),
        )

    a = a.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)

    if a.dim() == 1:
        a = a.view(-1, 1)
    if b.dim() == 1:
        b = b.view(-1, 1)
    if a.shape[0] != m:
        a = torch.ones(m, 1, device=device, dtype=dtype) / m
    if b.shape[0] != n:
        b = torch.ones(n, 1, device=device, dtype=dtype) / n

    if torch.sum(a) < eps or torch.sum(b) < eps:
        a = torch.ones(m, 1, device=device, dtype=dtype) / m
        b = torch.ones(n, 1, device=device, dtype=dtype) / n
    else:
        a = a / torch.sum(a)
        b = b / torch.sum(b)

    K = torch.exp(-cost_matrix / alpha)
    u = torch.ones(m, 1, device=device, dtype=dtype)
    v = torch.ones(n, 1, device=device, dtype=dtype)

    for _ in range(max_iter):
        u_prev = u.clone()
        KTu = torch.matmul(K.t(), u)
        v = b / (KTu + eps)
        Kv = torch.matmul(K, v)
        u = a / (Kv + eps)

        err = torch.norm(u - u_prev, p=float("inf"))
        if err < stop_thr:
            break

    P = u * K * v.t()
    ot_loss = torch.sum(P * cost_matrix)
    return ot_loss, P


def compute_ot_loss(
    teacher_last: torch.Tensor,
    student_last: torch.Tensor,
    teacher_att_last: torch.Tensor,
    attention_mask_teacher: torch.Tensor,
    attention_mask_student: torch.Tensor,
    input_ids_tea: torch.Tensor,
    input_ids_stu: torch.Tensor,
    tok_teacher: Any,
    tok_student: Any,
    projector: nn.Module,
    teacher_special: str = "<s>",
    student_special: str = "[CLS]",
    alpha: float = 0.1,
    max_iter: int = 100,
) -> torch.Tensor:
    """Compute EMO optimal transport loss over a batch.

    """
    device = teacher_last.device
    batch_size = teacher_last.size(0)
    total_loss = torch.tensor(0.0, device=device)

    for b in range(batch_size):
        valid_teacher_len = int(attention_mask_teacher[b].sum().item())
        valid_student_len = int(attention_mask_student[b].sum().item())
        if valid_teacher_len == 0 or valid_student_len == 0:
            continue

        valid_teacher_input_ids = input_ids_tea[b, :valid_teacher_len]
        valid_student_input_ids = input_ids_stu[b, :valid_student_len]

        teacher_tokens = tok_teacher.convert_ids_to_tokens(
            valid_teacher_input_ids.detach().cpu().tolist()
        )
        student_tokens = tok_student.convert_ids_to_tokens(
            valid_student_input_ids.detach().cpu().tolist()
        )

        teacher_seq = teacher_last[b, :valid_teacher_len, :]
        student_seq = student_last[b, :valid_student_len, :]

        projected_teacher_seq = projector(teacher_seq)

        teacher_attention_full = teacher_att_last[b]
        valid_teacher_attention = teacher_attention_full[:, :valid_teacher_len, :valid_teacher_len]

        teacher_importance = compute_token_importance(valid_teacher_attention, teacher_tokens)

        token_mapping = align_tokens(
            teacher_tokens,
            student_tokens,
            teacher_special=teacher_special,
            student_special=student_special,
        )

        student_importance = project_importance(
            teacher_importance,
            teacher_tokens,
            student_tokens,
            token_mapping,
        )

        tea_mass = teacher_importance.view(-1, 1).to(device=device, dtype=torch.float32)
        stu_mass = student_importance.view(-1, 1).to(device=device, dtype=torch.float32)

        student_seq_f = student_seq.to(device=device, dtype=torch.float32)
        proj_teacher_seq_f = projected_teacher_seq.to(device=device, dtype=torch.float32)
        cost_matrix = pairwise_attention_distance(student_seq_f, proj_teacher_seq_f)
        cost_matrix = cost_matrix.to(device=device, dtype=torch.float32)

        ot_loss_b, _ = sinkhorn(cost_matrix, stu_mass, tea_mass, alpha=alpha, max_iter=max_iter)
        total_loss = total_loss + ot_loss_b

    return total_loss / max(1, batch_size)


def compute_att_loss_2(
    teacher_atts: list[torch.Tensor],
    student_atts: list[torch.Tensor],
    input_ids_tea: torch.Tensor,
    input_ids_stu: torch.Tensor,
    attention_mask_tea: torch.Tensor,
    attention_mask_stu: torch.Tensor,
    tok_teacher: Any,
    tok_student: Any,
    k: int = 1,
    device: torch.device | None = None,
    teacher_special: str = "<s>",
    student_special: str = "[CLS]",
) -> torch.Tensor:
    """Compute EMO attention CKA distillation loss on salient reciprocal tokens.

    """
    if device is None:
        device = input_ids_stu.device
    batch_size = input_ids_stu.size(0)
    att_loss_total = torch.tensor(0.0, device=device)

    teacher_layer_num = len(teacher_atts)
    student_layer_num = len(student_atts)
    if teacher_layer_num == 0 or student_layer_num == 0:
        return att_loss_total

    layers_per_block = max(1, teacher_layer_num // student_layer_num)
    new_teacher_atts = [
        teacher_atts[min(idx * layers_per_block + layers_per_block - 1, teacher_layer_num - 1)]
        for idx in range(student_layer_num)
    ]

    teacher_last_k_layers = new_teacher_atts[-k:]
    student_last_k_layers = student_atts[-k:]

    cka = CKALoss(eps=1e-8).to(device)

    for b in range(batch_size):
        L_t_valid = int(attention_mask_tea[b].sum().item())
        L_s_valid = int(attention_mask_stu[b].sum().item())
        if L_t_valid == 0 or L_s_valid == 0:
            continue

        ids_tea = input_ids_tea[b, :L_t_valid]
        ids_stu = input_ids_stu[b, :L_s_valid]

        teacher_tokens = tok_teacher.convert_ids_to_tokens(ids_tea.detach().cpu().tolist())
        student_tokens = tok_student.convert_ids_to_tokens(ids_stu.detach().cpu().tolist())

        last_teacher_att_full = teacher_atts[-1][b]
        last_teacher_att = last_teacher_att_full[:, :L_t_valid, :L_t_valid]

        teacher_importance = compute_token_importance(last_teacher_att, teacher_tokens)

        reciprocal = build_reciprocal_mapping_from_token_lists(
            teacher_tokens,
            student_tokens,
            teacher_special=teacher_special,
            student_special=student_special,
        )
        if len(reciprocal) == 0:
            continue

        n_map = len(reciprocal)
        k_top = max(1, n_map // 3)

        token_scores: dict[str, float] = {}
        for idx_t, tok_t in enumerate(teacher_tokens):
            if tok_t in reciprocal:
                score = teacher_importance[idx_t].item()
                if tok_t not in token_scores or score > token_scores[tok_t]:
                    token_scores[tok_t] = score

        if len(token_scores) == 0:
            continue

        top_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:k_top]
        allowed_teacher_tokens = {t for t, _ in top_tokens}

        aligned_teacher_indices: list[int] = []
        aligned_student_indices: list[int] = []

        for t_tok in allowed_teacher_tokens:
            s_tok = reciprocal[t_tok]
            try:
                t_idx = teacher_tokens.index(t_tok)
            except ValueError:
                continue
            try:
                s_idx = student_tokens.index(s_tok)
            except ValueError:
                continue
            if t_idx < L_t_valid and s_idx < L_s_valid:
                aligned_teacher_indices.append(t_idx)
                aligned_student_indices.append(s_idx)

        N = len(aligned_teacher_indices)
        if N == 0:
            continue

        for teacher_att_layer, student_att_layer in zip(teacher_last_k_layers, student_last_k_layers):
            tea_sub = teacher_att_layer[b, :, :L_t_valid, :L_t_valid]
            stu_sub = student_att_layer[b, :, :L_s_valid, :L_s_valid]

            tea_tok_att = tea_sub[:, aligned_teacher_indices, :].mean(dim=0)
            stu_tok_att = stu_sub[:, aligned_student_indices, :].mean(dim=0)

            tea_tok_att = torch.where(
                tea_tok_att <= -1e2,
                torch.zeros_like(tea_tok_att, device=device),
                tea_tok_att,
            )
            stu_tok_att = torch.where(
                stu_tok_att <= -1e2,
                torch.zeros_like(stu_tok_att, device=device),
                stu_tok_att,
            )

            d_s = stu_tok_att.size(1)
            d_t = tea_tok_att.size(1)
            d = min(d_s, d_t)
            if d == 0:
                continue

            SH = stu_tok_att[:, :d]
            TH = tea_tok_att[:, :d]

            att_loss_total = att_loss_total + cka(SH, TH)

    return att_loss_total


@CRITERIONS.register(
    "emo",
    paper="EMO: Embedding Model Distillation via Intra-Model Relation and Optimal Transport Alignments",
)
class EMO(Criterion):
    """EMO distillation criterion combining Optimal Transport and Attention CKA distillation.

    Formula (from embedding-distill/emo-unsup-simcse.ipynb:cell 18):
      loss = w_ot * ot_loss + w_att * att_loss
      where:
        ot_loss = compute_ot_loss(T_last, S_last, T_att_last, ..., projector=proj_t2s)
        att_loss = compute_att_loss_2(T_atts, S_atts, ..., k=k_layers)
    """

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        teacher_dim: int | None = None
        w_att: float = 0.1
        w_ot: float = 1.0
        k_layers: int = 1
        sinkhorn_alpha: float = 0.1
        sinkhorn_max_iter: int = 100
        teacher_special: str | None = None
        student_special: str | None = None
        projector_ckpt: str | None = None

    cfg: Config

    def __init__(self, cfg: EMO.Config, alias: str = "emo") -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.alias = alias
        self.teacher_tokenizer: Any = None
        self.student_tokenizer: Any = None

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        student_dim = info.student_hidden
        teacher_dim = self.cfg.teacher_dim or info.teacher_hidden
        if teacher_dim is None:
            raise ValueError("EMO requires teacher_dim")
        # In cell 18: proj_t2s = nn.Linear(d_t, d_s, bias=False)
        proj = nn.Linear(teacher_dim, student_dim, bias=False)
        self.register_aux_module(modules, "projector", proj, group="projector")

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(hidden="all", attentions="all", n_views=1),
            teacher=SideRequest(hidden="all", attentions="all", n_views=1),
        )

    def prepare(self, pctx: PrepareContext) -> None:
        self.teacher_tokenizer = getattr(pctx.rt, "teacher_tokenizer", None)
        self.student_tokenizer = getattr(pctx.rt, "student_tokenizer", None)

        rt = pctx.rt
        models_cfg = getattr(getattr(rt, "config", None), "models", None)
        if models_cfg is not None:
            for side in ("student", "teacher"):
                spec = getattr(models_cfg, side, None)
                impl = getattr(spec, "attn_implementation", None)
                if impl in ("sdpa", "flash_attention_2"):
                    raise ValueError(
                        f"EMO requires full attention matrices ('attentions: all'), but models.{side}.attn_implementation "
                        f"is set to '{impl}'. SDPA / FlashAttention do not materialize attention weights. "
                        f"Please set models.{side}.attn_implementation: eager in your recipe."
                    )

        if self.cfg.projector_ckpt:
            self.load_aux_checkpoint(pctx, "projector", missing_ok=True)

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.side is None:
            raise ValueError("EMO requires teacher side outputs (hidden states and attentions)")

        # Student representations (view 0)
        s_side = ctx.student
        student_last = s_side.last_hidden
        student_num_layers = s_side.num_layers
        student_atts = [s_side.attention(i) for i in range(1, student_num_layers + 1)]
        attention_mask_student = ctx.batch.get("student.attention_mask", s_side.attention_mask)
        input_ids_stu = ctx.batch["student.input_ids"]

        # Teacher representations
        t_side = ctx.teacher.side
        teacher_last = t_side.last_hidden
        teacher_num_layers = t_side.num_layers
        teacher_atts = [t_side.attention(i) for i in range(1, teacher_num_layers + 1)]
        teacher_att_last = t_side.attention(-1)
        attention_mask_teacher = ctx.batch.get("teacher.attention_mask", t_side.attention_mask)
        input_ids_tea = ctx.batch.get("teacher.input_ids", input_ids_stu)

        tok_s = self.student_tokenizer or getattr(ctx.rt, "student_tokenizer", None)
        tok_t = self.teacher_tokenizer or getattr(ctx.rt, "teacher_tokenizer", None) or tok_s

        teacher_sp = self.cfg.teacher_special or getattr(tok_t, "cls_token", None) or getattr(tok_t, "bos_token", None) or "<s>"
        student_sp = self.cfg.student_special or getattr(tok_s, "cls_token", None) or getattr(tok_s, "bos_token", None) or "[CLS]"

        proj_mod = self.aux(ctx).projector

        ot_loss = compute_ot_loss(
            teacher_last=teacher_last,
            student_last=student_last,
            teacher_att_last=teacher_att_last,
            attention_mask_teacher=attention_mask_teacher,
            attention_mask_student=attention_mask_student,
            input_ids_tea=input_ids_tea,
            input_ids_stu=input_ids_stu,
            tok_teacher=tok_t,
            tok_student=tok_s,
            projector=proj_mod,
            teacher_special=teacher_sp,
            student_special=student_sp,
            alpha=self.cfg.sinkhorn_alpha,
            max_iter=self.cfg.sinkhorn_max_iter,
        )

        att_loss = compute_att_loss_2(
            teacher_atts=teacher_atts,
            student_atts=student_atts,
            input_ids_tea=input_ids_tea,
            input_ids_stu=input_ids_stu,
            attention_mask_tea=attention_mask_teacher,
            attention_mask_stu=attention_mask_student,
            tok_teacher=tok_t,
            tok_student=tok_s,
            k=self.cfg.k_layers,
            device=student_last.device,
            teacher_special=teacher_sp,
            student_special=student_sp,
        )

        total = self.cfg.w_ot * ot_loss + self.cfg.w_att * att_loss

        return LossOutput(
            terms={self.alias: LossTerm(total, "mean")},
            metrics={
                f"{self.alias}/ot_loss": ot_loss.detach(),
                f"{self.alias}/att_loss": att_loss.detach(),
                f"{self.alias}/total": total.detach(),
            },
        )


@CRITERIONS.register(
    "emo_kd",
    paper="EMO: Embedding Model Distillation via Intra-Model Relation and Optimal Transport Alignments",
)
class EMOKD(EMO):
    """Alias for EMO criterion."""

    pass
