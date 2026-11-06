from __future__ import annotations

import pytest

from baselines.longCoT import SEGMENTERS, Segment


@pytest.mark.parametrize("seg_type", ["sentence", "paragraph", "cue"])
def test_lossless(seg_type):
    seg = SEGMENTERS.build({"type": seg_type})
    assert seg.lossless is True

    test_strings = [
        "\n\n\n\nFirst paragraph.\n\nSecond paragraph. However, this is next.\n\nWait, another idea.",
        "Simple text with no special delimiters.",
        "Step 1: calculate x. Then calculate y. Finally, report \\boxed{42}.",
        "\n\nWait, but alternatively we can check.\n\nAnother case.",
        "A string with symbols like } and ] inside {x + y} and [1, 2] followed by A capital.",
    ]
    for s in test_strings:
        parts = seg(s)
        assert "".join(parts) == s, f"Segmenter {seg_type} lost characters on {s!r}"


def test_palign_not_lossless_rejected():
    palign_seg = SEGMENTERS.build({"type": "palign_sentences"})
    assert palign_seg.lossless is False

    with pytest.raises(ValueError, match="lossless"):
        Segment(Segment.Config(segmenter="palign_sentences"))
