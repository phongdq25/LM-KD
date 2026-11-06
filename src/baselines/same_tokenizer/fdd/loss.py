"""Feature Dynamics Distillation (FDD) Criterion for same-tokenizer LLM distillation.

Source: Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation (ACL)
"""

from __future__ import annotations

from baselines.same_tokenizer.fdd.functional import compute_fdd_loss
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec


@CRITERIONS.register(
    "fdd",
    paper="Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation",
    source="ACL: Feature Dynamics Distillation",
)
class FDDCriterion(Criterion):
    """Feature Dynamics Distillation (FDD) Criterion.

    Aligns internal representations across discrete Transformer layers:
    1. Trajectory loss (traj_loss): Layer-wise soft-label distillation loss via lm_head logits.
    2. Derivative loss (der_loss): Cosine similarity matching between adjacent layer feature deltas
       in log-probability space: delta = log_p_{l} - log_p_{l-1}.
    """

    needs_teacher = True
    supports_packing = False

    class Config(BaseConfig):
        student_layers: list[int] = [-4, -3, -2, -1]
        teacher_layers: list[int] = [-4, -3, -2, -1]
        temperature: float = 1.0
        w_traj: float = 1.0
        w_der: float = 1.0

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(hidden=tuple(self.cfg.student_layers)),
            teacher=SideRequest(hidden=tuple(self.cfg.teacher_layers)),
            teacher_signal=TeacherSignalSpec(kind="hidden"),
        )

    def _resolve_heads(self, ctx: LossContext):
        s_head = None
        t_head = None

        if ctx.student.adapter is not None:
            s_head = ctx.student.adapter.lm_head(ctx.student.model)
        if ctx.teacher is not None and ctx.teacher.side is not None and ctx.teacher.side.adapter is not None:
            t_head = ctx.teacher.side.adapter.lm_head(ctx.teacher.side.model)

        if s_head is None and ctx.rt is not None:
            s_head = getattr(ctx.rt, "student_adapter", None)
            if s_head is not None:
                s_head = s_head.lm_head(ctx.rt.student_model)
        if t_head is None and ctx.rt is not None:
            t_head = getattr(ctx.rt, "teacher_adapter", None)
            if t_head is not None:
                t_head = t_head.lm_head(ctx.rt.teacher_model)

        if s_head is None:
            s_head = getattr(ctx.student.model, "lm_head", None)
            get_out_emb = getattr(ctx.student.model, "get_output_embeddings", None)
            if s_head is None and callable(get_out_emb):
                s_head = get_out_emb()
            if s_head is None and ctx.student.logits is not None:
                s_head = getattr(ctx.student.logits, "lm_head", None)

        if t_head is None and ctx.teacher is not None and ctx.teacher.side is not None:
            t_head = getattr(ctx.teacher.side.model, "lm_head", None)
            get_t_out_emb = getattr(ctx.teacher.side.model, "get_output_embeddings", None)
            if t_head is None and callable(get_t_out_emb):
                t_head = get_t_out_emb()
            if t_head is None and ctx.teacher.logits is not None:
                t_head = getattr(ctx.teacher.logits, "lm_head", None)

        if s_head is None or t_head is None:
            raise ValueError("FDD requires accessible student and teacher lm_head modules")

        return s_head, t_head

    def forward(self, ctx: LossContext) -> LossOutput:
        if len(self.cfg.student_layers) != len(self.cfg.teacher_layers):
            raise ValueError(
                f"FDD student_layers ({len(self.cfg.student_layers)}) and "
                f"teacher_layers ({len(self.cfg.teacher_layers)}) must have the same length"
            )

        if ctx.teacher is None or ctx.teacher.side is None:
            raise ValueError("FDD requires teacher hidden states (online teacher model)")

        s_head, t_head = self._resolve_heads(ctx)

        s_hiddens = [ctx.student.hidden(l) for l in self.cfg.student_layers]
        t_hiddens = [ctx.teacher.side.hidden(l) for l in self.cfg.teacher_layers]
        mask = ctx.supervised_mask()

        s_model = getattr(ctx.student, "model", None)
        t_model = getattr(ctx.teacher.side, "model", None)

        total_loss, traj_loss, der_loss = compute_fdd_loss(
            s_hiddens=s_hiddens,
            t_hiddens=t_hiddens,
            mask=mask,
            s_head=s_head,
            t_head=t_head,
            temperature=self.cfg.temperature,
            w_traj=self.cfg.w_traj,
            w_der=self.cfg.w_der,
            s_model=s_model,
            t_model=t_model,
        )

        return LossOutput(
            terms={self.alias: LossTerm(total_loss, "mean")},
            metrics={
                f"{self.alias}/traj": traj_loss.detach(),
                f"{self.alias}/der": der_loss.detach(),
            },
        )


FDD = FDDCriterion
