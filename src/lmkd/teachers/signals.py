"""Teacher signal spec / output (DESIGN §5.4)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import torch

from lmkd.models.logits_view import LogitsView

if TYPE_CHECKING:
    from lmkd.models.forward import SideOutput

SignalKind = Literal["none", "full", "topk", "sampled_logp", "seq_logp", "hidden", "embedding"]
_ORDER = {"none": 0, "sampled_logp": 1, "seq_logp": 1, "embedding": 1, "topk": 2, "hidden": 3, "full": 4}


@dataclass(frozen=True)
class TeacherSignalSpec:
    """Which teacher signal a criterion needs (full logits, top-k, sampled logp, sequence logp, hidden,
    embedding) and how.
    """
    kind: SignalKind = "full"
    k: int = 0
    support: Literal["teacher", "student"] = "teacher"   # SDPO evaluates teacher at student top-k
    tail: Literal["none", "bucket", "renorm"] = "bucket"
    temperature: float = 1.0

    def merge(self, o: TeacherSignalSpec) -> TeacherSignalSpec:
        """The richer kind wins; 'full' subsumes topk/sampled. Conflicting temperatures are an error
        (criteria apply their own temperature on the LogitsView instead)."""
        if self.temperature != o.temperature:
            raise ValueError("criteria disagree on teacher signal temperature; use temperature on the view")
        a, b = (self, o) if _ORDER[self.kind] >= _ORDER[o.kind] else (o, self)
        return replace(a, k=max(a.k, b.k))


@dataclass
class TeacherOutput:
    """Teacher signals for one batch; only the fields requested by the spec are filled."""
    logits: LogitsView | None = None          # kind=full (online)
    topk_ids: torch.Tensor | None = None      # [B,T,K] target-aligned
    topk_logp: torch.Tensor | None = None     # [B,T,K]
    tail_logp: torch.Tensor | None = None     # [B,T]  log(1 - sum p_topk)
    token_logp: torch.Tensor | None = None    # [B,T]  log p(target)
    seq_logp: torch.Tensor | None = None      # [B]
    side: SideOutput | None = None            # when the teacher ran online (hidden/attn)
    embedding: torch.Tensor | None = None     # [B,D]


def response_shift_index(batch, teacher_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Map every STUDENT position to the TEACHER position holding the same response token, when
    teacher input = (different/privileged prompt) + (identical response tokens) (K8).

    position p (student) -> p - student.prompt_len + teacher.prompt_len.
    Works for both HF positional tensors (hidden) and target-aligned tensors (same shift).
    Returns (index [B,Ts] clamped into [0, teacher_len), valid [B,Ts] bool).
    """
    ps = batch["student.prompt_len"].view(-1, 1)
    pt = batch["teacher.prompt_len"].view(-1, 1).to(ps.device)
    Ts = batch["student.input_ids"].shape[1]
    pos = torch.arange(Ts, device=ps.device).view(1, -1) - ps + pt
    valid = (pos >= 0) & (pos < teacher_len)
    return pos.clamp(0, teacher_len - 1), valid


def gather_positions(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """x [B,T,...] , index [B,T'] -> [B,T',...]"""
    idx = index.to(x.device)
    if x.dim() == 2:
        return x.gather(1, idx)
    return x.gather(1, idx.view(*idx.shape, *([1] * (x.dim() - 2))).expand(*idx.shape, *x.shape[2:]))
