"""S-SPPO (Semantic-Calibrated SPPO) baseline."""

from __future__ import annotations

from baselines.self_play.s_sppo.loss import (
    S_SPPO,
    SSPPO,
    s_sppo_loss,
    sspppo_loss,
)

__all__ = ["SSPPO", "S_SPPO", "sspppo_loss", "s_sppo_loss"]
