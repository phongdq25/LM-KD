"""Long-CoT Distillation Baselines.

Includes:
- SSFT (Segment-Selective SFT)
- P-ALIGN (Prefix-search Alignment)
- SGL-Spectral (Spectral Guided Learning)
"""

from __future__ import annotations

from baselines.longCoT import palign, segmenters, sgl_spectral, ssft, transforms
from baselines.longCoT.palign import PAlignStage
from baselines.longCoT.segmenters import (
    SEGMENTERS,
    CueSegmenter,
    PAlignSentencesSegmenter,
    ParagraphSegmenter,
    Segmenter,
    SentenceSegmenter,
)
from baselines.longCoT.sgl_spectral import SGLSpectralStage
from baselines.longCoT.ssft import SSFTStage
from baselines.longCoT.transforms import RenderCoT, Segment, TokenizeCoT

__all__ = [
    "ssft",
    "palign",
    "sgl_spectral",
    "segmenters",
    "transforms",
    "SEGMENTERS",
    "Segmenter",
    "SentenceSegmenter",
    "ParagraphSegmenter",
    "CueSegmenter",
    "PAlignSentencesSegmenter",
    "SSFTStage",
    "PAlignStage",
    "SGLSpectralStage",
    "RenderCoT",
    "TokenizeCoT",
    "Segment",
]
