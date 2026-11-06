"""Multi-granularity Text Alignment (MTA) distillation criterion.

MTA performs multi-granularity alignment across student and teacher hidden states:
1. Token-level: Attention-weighted cosine similarity on aligned tokens.
2. Span-level: Relational similarity matching between linguistic phrase spans (noun chunks, verb phrases).
3. Word-level: Relational similarity matching between word spans.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from baselines.mta.functional import (
    aggregate_spans_for_model,
    compute_relational_span_loss,
    compute_token_cosine_loss,
    compute_token_weights,
    extract_spans_and_words,
)
from lmkd.core.batch import FIELDS, FieldSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo, PrepareContext
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.modules import LinearProjector, ModuleRegistry
from lmkd.teachers.signals import TeacherSignalSpec

# Register batch fields for offline / precomputed span extraction (if not already registered)
if "align.mta_spans" not in FIELDS:
    FIELDS.register(
        FieldSpec("align.mta_spans", "ragged", torch.long, 0, None, "[S, 2] char [start, end) offsets per span")
    )
if "align.mta_words" not in FIELDS:
    FIELDS.register(
        FieldSpec("align.mta_words", "ragged", torch.long, 0, None, "[W, 2] char [start, end) offsets per word")
    )


@CRITERIONS.register("mta")
class MTA(Criterion):
    """Multi-granularity Text Alignment (MTA) distillation loss.

    Formula:
        L_MTA = (L_word + L_span) / len(layers)
        where for each layer:
            L_layer = L_span_relational + lambda_token * L_token_cosine
            L_span_relational = MSE(S_intra_sim, T_intra_sim) weighted by span pair importance
            L_token_cosine = mean_tokens(1 - cos(P(H_S), H_T)) weighted by attention weights
    """

    needs_teacher = True
    supports_packing = False

    class Config(BaseConfig):
        student_layers: list[int] = [-1]
        teacher_layers: list[int] = [-1]
        split_layer_mapping: list[int] | None = None
        w_span_loss: float = 1.0
        token_loss_weight: float = 0.1
        mode: Literal["auto", "same_tokenizer", "cross_tokenizer"] = "auto"
        spacy_model: str = "en_core_web_sm"

    def __init__(self, cfg: BaseConfig | None = None):
        super().__init__(cfg)
        self.nlp: Any = None

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(hidden=tuple(self.cfg.student_layers)),
            teacher=SideRequest(hidden=tuple(self.cfg.teacher_layers)),
            teacher_signal=TeacherSignalSpec(kind="hidden"),
        )

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        if len(self.cfg.student_layers) != len(self.cfg.teacher_layers) or not self.cfg.student_layers:
            raise ValueError("MTA student_layers and teacher_layers must have the same nonzero length")
        t_hidden = info.teacher_hidden or info.student_hidden
        for idx in range(len(self.cfg.student_layers)):
            cfg = LinearProjector.Config(
                in_dim=info.student_hidden,
                out_dim=t_hidden,
                bias=True,
            )
            self.register_aux_module(modules, f"proj_{idx}", LinearProjector(cfg), group="projector")

    def prepare(self, ctx: PrepareContext) -> None:
        try:
            import spacy

            self.nlp = spacy.load(self.cfg.spacy_model)
        except Exception:
            self.nlp = None

    def _resolve_spans(self, ctx: LossContext) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
        batch = ctx.batch
        raw_spans = batch.get("align.mta_spans")
        raw_words = batch.get("align.mta_words")

        if raw_spans is not None and raw_words is not None:
            def to_tuples(raw_list: list[Any]) -> list[list[tuple[int, int]]]:
                out: list[list[tuple[int, int]]] = []
                for sample in raw_list:
                    sample_tuples: list[tuple[int, int]] = []
                    for item in sample:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            sample_tuples.append((int(item[0]), int(item[1])))
                        elif isinstance(item, torch.Tensor) and item.numel() >= 2:
                            sample_tuples.append((int(item[0].item()), int(item[1].item())))
                    out.append(sample_tuples)
                return out

            return to_tuples(raw_spans), to_tuples(raw_words)

        if "meta.response_text" in batch:
            texts = batch["meta.response_text"]
        elif "meta.prompt_text" in batch:
            texts = batch["meta.prompt_text"]
        else:
            texts = [""] * batch["student.input_ids"].shape[0]

        return extract_spans_and_words(texts, self.nlp)

    def _resolve_offsets(
        self, ctx: LossContext, B: int, SeqLen_s: int, SeqLen_t: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = ctx.batch
        if "student.offsets" in batch:
            s_off = batch["student.offsets"]
            if s_off.dim() == 3:
                s_offsets = s_off
            else:
                s_offsets = torch.zeros(B, SeqLen_s, 2, dtype=torch.long, device=device)
        else:
            s_offsets = torch.zeros(B, SeqLen_s, 2, dtype=torch.long, device=device)

        if "teacher.offsets" in batch:
            t_off = batch["teacher.offsets"]
            if t_off.dim() == 3:
                t_offsets = t_off
            else:
                t_offsets = s_offsets
        else:
            t_offsets = s_offsets

        return s_offsets, t_offsets

    def _compute_layer_pair_loss(
        self,
        projector: Any,
        s_hidden: torch.Tensor,
        t_hidden: torch.Tensor,
        s_att_mask: torch.Tensor,
        t_att_mask: torch.Tensor,
        s_offsets: torch.Tensor,
        t_offsets: torch.Tensor,
        offsets: list[list[tuple[int, int]]],
        is_same_tokenizer: bool,
    ) -> torch.Tensor:
        device = s_hidden.device
        s_weights = compute_token_weights(s_hidden, s_att_mask)
        t_weights = compute_token_weights(t_hidden, t_att_mask)

        s_repr, _, valid_mask = aggregate_spans_for_model(
            s_hidden, s_weights, s_att_mask, s_offsets, offsets
        )
        t_repr, t_w_sum, _ = aggregate_spans_for_model(
            t_hidden, t_weights, t_att_mask, t_offsets, offsets
        )

        if s_repr is None or t_repr is None or valid_mask is None or t_w_sum is None:
            return torch.tensor(0.0, device=device)

        span_loss = compute_relational_span_loss(
            s_repr, t_repr, valid_mask, t_w_sum, projector=projector
        )

        if is_same_tokenizer and self.cfg.token_loss_weight > 0:
            token_mask = s_att_mask.bool() & (t_offsets[..., 1] > t_offsets[..., 0])
            token_loss = compute_token_cosine_loss(
                s_hidden, t_hidden, projector, t_weights, token_mask
            )
            return span_loss + self.cfg.token_loss_weight * token_loss

        return span_loss

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.side is None:
            raise ValueError("MTA requires teacher hidden states")

        num_layers = len(self.cfg.student_layers)
        split = self.cfg.split_layer_mapping
        if split is None:
            split = [0, num_layers // 2, num_layers] if num_layers > 1 else [0, 0, 1]

        s_att_mask = ctx.batch["student.attention_mask"]
        t_att_mask = ctx.batch.get("teacher.attention_mask", s_att_mask)
        device = s_att_mask.device
        B, SeqLen_s = s_att_mask.shape
        SeqLen_t = t_att_mask.shape[1]

        spans_offsets, words_offsets = self._resolve_spans(ctx)
        s_offsets, t_offsets = self._resolve_offsets(ctx, B, SeqLen_s, SeqLen_t, device)

        is_same_tok = (
            self.cfg.mode == "same_tokenizer"
            or (self.cfg.mode == "auto" and SeqLen_s == SeqLen_t and "teacher.offsets" not in ctx.batch)
        )

        word_loss = torch.tensor(0.0, device=device)
        span_loss = torch.tensor(0.0, device=device)
        aux = self.aux(ctx)

        for i, (s_layer, t_layer) in enumerate(zip(self.cfg.student_layers, self.cfg.teacher_layers)):
            projector = aux[f"proj_{i}"]
            s_h = ctx.student.hidden(s_layer)
            t_h = ctx.teacher.side.hidden(t_layer)

            if split[0] <= i < split[1]:
                layer_loss = self._compute_layer_pair_loss(
                    projector, s_h, t_h, s_att_mask, t_att_mask, s_offsets, t_offsets,
                    words_offsets, is_same_tok
                )
                word_loss = word_loss + layer_loss
            else:
                layer_loss = self._compute_layer_pair_loss(
                    projector, s_h, t_h, s_att_mask, t_att_mask, s_offsets, t_offsets,
                    spans_offsets, is_same_tok
                )
                span_loss = span_loss + layer_loss

        overall_loss = (word_loss + span_loss) / max(num_layers, 1)
        final_loss = overall_loss * self.cfg.w_span_loss

        return LossOutput(
            {"mta": LossTerm(final_loss, "mean")},
            metrics={
                "mta_word": word_loss.detach(),
                "mta_span": span_loss.detach(),
            },
        )

