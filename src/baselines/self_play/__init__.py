"""Self-Play Fine-Tuning baselines for LM-KD.

Contains canonical method implementations ("phương pháp của từng cái") from the
self-play fine-tuning literature:
- SPIN: Self-Play Fine-Tuning (Chen et al., ICML 2024)
- T-SPIN: Triplet Self-Play Fine-Tuning with base model proto anchor
- SPACE: Self-Play via Noise Contrastive Estimation (Wu et al., 2024)
- SPPO: Self-Play Preference Optimization (Wu et al., ICML 2024)
- S-SPPO: Semantic-Calibrated SPPO
- RSPO: Regularized Self-Play Preference Optimization (reverse-KL regularized SPO)
"""

from __future__ import annotations

from baselines.self_play import (
    common,
    rspo,
    s_sppo,
    space,
    spin,
    sppo,
    tspin,
)
from baselines.self_play.common import (
    _check_pairwise,
    _reduction_term,
    pairwise,
    sequence_logps,
    triplet,
)
from baselines.self_play.rspo import RSPO, rspo_loss
from baselines.self_play.s_sppo import S_SPPO, SSPPO, s_sppo_loss, sspppo_loss
from baselines.self_play.space import SPACE, space_loss
from baselines.self_play.spin import SPIN, spin_loss
from baselines.self_play.sppo import SPPO, sppo_loss
from baselines.self_play.tspin import TSPIN, tspin_loss

__all__ = [
    # Submodules
    "common",
    "spin",
    "tspin",
    "space",
    "sppo",
    "s_sppo",
    "rspo",
    # Criterion Classes
    "SPIN",
    "TSPIN",
    "SPACE",
    "SPPO",
    "SSPPO",
    "S_SPPO",
    "RSPO",
    # Functional Primitives
    "spin_loss",
    "tspin_loss",
    "space_loss",
    "sppo_loss",
    "s_sppo_loss",
    "sspppo_loss",
    "rspo_loss",
    # Utilities
    "pairwise",
    "triplet",
    "sequence_logps",
    "_check_pairwise",
    "_reduction_term",
]
