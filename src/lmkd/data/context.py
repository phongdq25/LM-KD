"""Context handed to data transforms / sources."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataContext:
    """What transforms/aligners need besides the sample: tokenizers, max length, artifact store, vocab map."""
    student_tokenizer: Any
    teacher_tokenizer: Any | None = None
    max_length: int = 1024
    template: Any | None = None      # lmkd.data.templates (task T18)
    span_service: Any | None = None  # lmkd.data.spans (task T11)
    vocab_map: Any | None = None     # cross-tokenizer vocab map (task T11)
    artifacts: Any | None = None     # ArtifactStore, for transforms reading stage outputs
    extra: dict = field(default_factory=dict)
