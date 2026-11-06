"""``mta_span`` aligner: extract linguistic phrase and word spans for MTA."""

from __future__ import annotations

from typing import Any

from baselines.mta.functional import extract_spans_and_words
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import ALIGNERS
from lmkd.data.aligners.base import Aligner


@ALIGNERS.register("mta_span", source="white-box-distillation/same-tokenizer/finetune.py:MTALoss")
class MtaSpan(Aligner):
    """Extract linguistic verb-phrase / noun-chunk spans and word spans for MTA.

    Provides ``align.mta_spans`` and ``align.mta_words`` as lists of [start_char, end_char].
    """

    requires = ("student.input_ids",)
    provides = ("align.mta_spans", "align.mta_words")

    class Config(BaseConfig):
        spacy_model: str = "en_core_web_sm"

    def __init__(self, cfg: BaseConfig | None = None):
        super().__init__(cfg)
        self.nlp: Any = None

    def setup(self, rt: Any) -> None:
        try:
            import spacy

            self.nlp = spacy.load(self.cfg.spacy_model)
        except Exception:
            self.nlp = None

    def empty(self, sample: dict) -> dict[str, Any]:
        return {"align.mta_spans": [], "align.mta_words": []}

    def __call__(self, sample: dict, ctx: Any) -> dict[str, Any]:
        text = sample.get("meta.response_text")
        if text is None:
            prompt = sample.get("meta.prompt_text", "")
            text = prompt

        if not text:
            return self.empty(sample)

        spans_list, words_list = extract_spans_and_words([text], self.nlp)
        spans = [[s[0], s[1]] for s in spans_list[0]] if spans_list else []
        words = [[w[0], w[1]] for w in words_list[0]] if words_list else []

        return {"align.mta_spans": spans, "align.mta_words": words}

