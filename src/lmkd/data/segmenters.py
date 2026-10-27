"""Step segmentation rules (SGL segment/sentence.py:14-33, segment/rules.py:12-62)."""
from __future__ import annotations

import re
from typing import ClassVar

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import SEGMENTERS

_STEP_BOUNDARY = re.compile(r"([.?!\}\]])([\s\n]+)([A-Z])")

CUE_PATTERN = (
    r"(\n\nWait|\n\nAlternatively|\n\nBut wait|\n\nBut alternatively|\n\nBut just to|\n\nHowever"
    r"|\n\nNot sure|\n\nGoing back|\n\nBacktrack|\n\nTrace back|\n\nAnother)"
)
PARAGRAPH_PATTERN = r"(\n\n)"


def _drop_empty(segments: list[str], text: str) -> list[str]:
    merged: list[str] = []
    for segment in segments:
        if segment == "":
            continue
        if merged and merged[-1] == "":
            merged[-1] = segment
        else:
            merged.append(segment)
    if not merged:
        return [text] if text else []
    assert "".join(merged) == text, "split_segments lost characters"
    return merged


def _split_segments(text: str, pattern: str) -> list[str]:
    if not text:
        return []
    parts = re.split(pattern, text)
    segments = [parts[0]]
    for index in range(1, len(parts), 2):
        segments.append(parts[index] + parts[index + 1])
    return _drop_empty(segments, text)


class Segmenter(Plugin):
    """Base class for step segmenters."""

    lossless: ClassVar[bool] = True

    class Config(BaseConfig):
        pass

    def __call__(self, text: str) -> list[str]:
        raise NotImplementedError


@SEGMENTERS.register("sentence", source="SpectralGuidedLearning/src/sgl/segment/sentence.py:14-33")
class SentenceSegmenter(Segmenter):
    """Split at sentence-ending punctuation/braces followed by whitespace and capital letter (lossless)."""

    lossless: ClassVar[bool] = True

    def __call__(self, text: str) -> list[str]:
        if not text:
            return []
        pieces, cursor = [], 0
        for match in _STEP_BOUNDARY.finditer(text):
            boundary = match.start(3)
            pieces.append(text[cursor:boundary])
            cursor = boundary
        if cursor < len(text):
            pieces.append(text[cursor:])
        return pieces


@SEGMENTERS.register("paragraph", source="SpectralGuidedLearning/src/sgl/segment/rules.py:23-53")
class ParagraphSegmenter(Segmenter):
    """Split at paragraph break (\\n\\n) with delimiter attached to following segment (lossless)."""

    lossless: ClassVar[bool] = True

    def __call__(self, text: str) -> list[str]:
        return _split_segments(text, PARAGRAPH_PATTERN)


@SEGMENTERS.register("cue", source="SpectralGuidedLearning/src/sgl/segment/rules.py:23-53")
class CueSegmenter(Segmenter):
    """Split at backtracking cues (\\n\\nWait, \\n\\nAlternatively...) (lossless)."""

    lossless: ClassVar[bool] = True

    def __call__(self, text: str) -> list[str]:
        return _split_segments(text, CUE_PATTERN)


@SEGMENTERS.register("palign_sentences", source="SpectralGuidedLearning/src/sgl/segment/rules.py:56-62")
class PAlignSentencesSegmenter(Segmenter):
    """P-ALIGN '. ' sentence splitter (non-lossless, strips each piece and appends '.')."""

    lossless: ClassVar[bool] = False

    def __call__(self, text: str) -> list[str]:
        sentences = text.split(". ")
        return [
            sentence.strip() + "." if not sentence.endswith(".") else sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

