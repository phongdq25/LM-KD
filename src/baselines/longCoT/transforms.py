"""Common Long-CoT transforms: templates, tokenization, and step segmentation.

These transforms are shared across Long-CoT baselines (SSFT, P-ALIGN, SGL-Spectral).
"""

from __future__ import annotations

import unicodedata
from typing import Any

import torch

from baselines.longCoT.segmenters import SEGMENTERS
from lmkd.core.batch import FIELDS, FieldSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.labels import make_labels
from lmkd.data.spans import char_cuts_to_token_spans, encode_with_offsets, verify_round_trip
from lmkd.data.templates import (
    PROMPT_TEMPLATE,
    build_prompt,
    reconcile_thinking_markers,
)
from lmkd.data.transforms.base import Transform

# -----------------------------------------------------------------------------
# Common Field registrations for Long-CoT
# -----------------------------------------------------------------------------
FIELDS.register(FieldSpec("seq.response_span", "vector", torch.long, doc="[start, end) token of text response"))
FIELDS.register(FieldSpec("seq.steps", "ragged", torch.long, doc="[K, 2] absolute token spans per step"))
FIELDS.register(FieldSpec("meta.selected_steps", "meta"))
FIELDS.register(FieldSpec("meta.response_text", "meta"))
FIELDS.register(FieldSpec("meta.prompt_text", "meta"))
FIELDS.register(FieldSpec("meta.prefix_boundary", "meta"))
FIELDS.register(FieldSpec("meta.token_starts", "meta"))
FIELDS.register(FieldSpec("meta.segments", "meta"))
FIELDS.register(FieldSpec("signals.step_strength", "meta"))
FIELDS.register(FieldSpec("signals.step_entropy", "meta"))
FIELDS.register(FieldSpec("signals.step_gain", "meta"))
FIELDS.register(FieldSpec("signals.segment_ig", "meta"))


@TRANSFORMS.register("render_cot", source="SpectralGuidedLearning/src/sgl/data/prepare.py:85-213")
class RenderCoT(Transform):
    """Render prompt and response texts with thinking markers and templates.

    Source: SGL data/prepare.py:32-101.
    """

    requires = ()
    provides = ("meta.prompt_text", "meta.response_text")

    class Config(BaseConfig):
        problem_field: str = "problem"
        response_fields: list[str] = ["response"]
        response_format: str = "{0}"
        thinking_mode: str = "on"
        chat_template: bool = True
        instruction_separator: str = ""
        template: str = PROMPT_TEMPLATE

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        problem = sample.get(self.cfg.problem_field)
        if problem is None:
            problem = sample.get("problem") or sample.get("question") or sample.get("input", "")
        if self.cfg.response_fields == ["response"] and "response" not in sample:
            if sample.get("deepseek_thinking_trajectory") and sample.get("deepseek_attempt"):
                response = (
                    f"<think>\n{str(sample['deepseek_thinking_trajectory']).strip()}\n</think>\n\n"
                    f"{str(sample['deepseek_attempt']).strip()}"
                )
            elif sample.get("generated_response"):
                response = f"<think>\n{str(sample['generated_response']).strip()}"
            else:
                response = sample.get("output") or sample.get("solution") or ""
        else:
            resp_vals = [sample[f] for f in self.cfg.response_fields]
            response = self.cfg.response_format.format(*resp_vals)

        problem_norm = unicodedata.normalize("NFC", str(problem))
        response_norm = unicodedata.normalize("NFC", str(response))

        prompt = build_prompt(
            ctx.student_tokenizer,
            problem_norm,
            chat_template=self.cfg.chat_template,
            thinking_mode=self.cfg.thinking_mode,
            instruction_separator=self.cfg.instruction_separator,
            template=self.cfg.template,
        )
        response_final = reconcile_thinking_markers(prompt, response_norm)

        sample["meta.prompt_text"] = prompt
        sample["meta.response_text"] = response_final
        sample["meta.question"] = problem_norm
        if "answer" in sample:
            sample["meta.answer"] = str(sample["answer"])
        elif "solution" in sample and ("deepseek_thinking_trajectory" in sample or "gemini_thinking_trajectory" in sample):
            sample["meta.answer"] = str(sample["solution"])
            sample["answer"] = str(sample["solution"])
        return sample


@TRANSFORMS.register("tokenize_cot", source="SpectralGuidedLearning/src/sgl/data/prepare.py:104-145")
class TokenizeCoT(Transform):
    """Tokenize prompt and response with offsets, target labels, and stop tokens.

    Source: SGL data/prepare.py:104-145.
    """

    requires = ("meta.prompt_text", "meta.response_text")
    provides = (
        "student.input_ids",
        "student.attention_mask",
        "student.labels",
        "student.prompt_len",
        "seq.response_span",
        "meta.token_starts",
    )

    class Config(BaseConfig):
        stop_suffix: str = ""
        verify: bool = True

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        tok = ctx.student_tokenizer
        prompt = sample["meta.prompt_text"]
        response = sample["meta.response_text"]

        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        response_ids, starts = encode_with_offsets(tok, response)

        if len(prompt_ids) + len(response_ids) > ctx.max_length:
            return None

        stop_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
        if self.cfg.stop_suffix:
            stop_ids += tok(self.cfg.stop_suffix, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids + stop_ids
        labels = make_labels(input_ids, prompt_len=len(prompt_ids))

        if self.cfg.verify and not verify_round_trip(tok, response_ids, [(0, len(response_ids))], response):
            raise ValueError("TokenizeCoT: round-trip verification failed on response text")

        out = {
            k: v
            for k, v in sample.items()
            if k.startswith("meta.") or k in FIELDS or k in ("question", "problem", "answer", "input")
        }
        out["student.input_ids"] = input_ids
        out["student.attention_mask"] = [1] * len(input_ids)
        out["student.labels"] = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        out["student.prompt_len"] = len(prompt_ids)
        out["seq.response_span"] = [len(prompt_ids), len(prompt_ids) + len(response_ids)]
        out["meta.token_starts"] = starts
        return out


@TRANSFORMS.register("segment", source="SpectralGuidedLearning/src/sgl/segment/sentence.py:63-88")
class Segment(Transform):
    """Segment response into reasoning steps using a lossless segmenter.

    Source: SGL segment/sentence.py:54-88.
    """

    requires = ("seq.response_span", "meta.token_starts", "meta.response_text")
    provides = ("seq.steps",)

    class Config(BaseConfig):
        segmenter: str | dict = "sentence"
        store_texts: bool = False

    def __init__(self, cfg: Segment.Config):
        super().__init__(cfg)
        spec = self.cfg.segmenter if isinstance(self.cfg.segmenter, dict) else {"type": self.cfg.segmenter}
        self.seg = SEGMENTERS.build(spec)
        if not getattr(self.seg, "lossless", False):
            raise ValueError(f"segment requires a lossless segmenter, but {self.cfg.segmenter!r} has lossless=False")

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        response = sample["meta.response_text"]
        starts = sample["meta.token_starts"]
        resp_start, _ = sample["seq.response_span"]

        pieces = self.seg(response)
        cuts = []
        pos = 0
        for piece in pieces:
            cuts.append(pos)
            pos += len(piece)
        cuts.append(pos)

        step_spans = char_cuts_to_token_spans(cuts, starts, offset=resp_start)
        sample["seq.steps"] = step_spans
        if self.cfg.store_texts:
            sample["meta.segments"] = pieces
        return sample


__all__ = [
    "RenderCoT",
    "TokenizeCoT",
    "Segment",
]

