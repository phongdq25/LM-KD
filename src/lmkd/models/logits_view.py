"""LogitsView: lazy, TARGET-ALIGNED access to logits without materializing [B,T,V] (DESIGN §5.3).

Row ``t`` of every method below corresponds to target token ``input_ids[t]`` (it is computed from
the hidden state at position ``t-1``). Row 0 is meaningless and always masked by labels.

Current implementation status
-----------------------------
* ``full``, ``chunks``, ``target_logp``, ``gather``, ``topk``, ``entropy``, ``at`` work.
* Student target_logp and token divergence recompute linear-head chunks in backward.
* gather retains the existing chunked graph; unsupported/ZeRO-3 heads use module forward.
* Custom autograd requires full linear-head weight and bias. DeepSpeed parameter gathering,
  gradient reduction and external-parameter registration belong to the T05 engine.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmkd.compat import ds_api
from lmkd.models.chunked_head import ChunkedHeadFunction

LOGITS_VIEW_EFFICIENT = True
_WARNED_HEADS: set[type] = set()


@dataclass
class LogitChunk:
    """A slice of target-aligned logits [B, end-start, V] produced by ``LogitsView.chunks``."""
    start: int            # first target position (inclusive) of this chunk along T
    end: int              # exclusive
    logits: torch.Tensor  # [B, end-start, V] fp32, target-aligned


class LogitsView:
    """Lazy ``[B,T,V]`` logits: keeps the final hidden states + lm_head and computes chunks/gathers/top-k on
    demand (target-aligned, DESIGN §4.2).
    """
    def __init__(
        self,
        hidden: torch.Tensor | None = None,
        lm_head: nn.Module | None = None,
        *,
        logits: torch.Tensor | None = None,
        temperature: float = 1.0,
        vocab_limit: int | None = None,
    ):
        """Build from (hidden [B,T,D] after final norm, lm_head) — preferred — or from already
        computed raw HF logits [B,T,V] (``logits=``, NOT shifted: HF convention)."""
        if (hidden is None) == (logits is None):
            raise ValueError("give exactly one of hidden(+lm_head) or logits")
        if hidden is not None and lm_head is None:
            raise ValueError("hidden requires lm_head")
        self._hidden = hidden
        self._head = lm_head
        self._logits = logits
        self.temperature = temperature
        self.vocab_limit = vocab_limit  # truncate vocab (e.g. padded embedding rows)

    # ------------------------------------------------------------------ basic info
    @property
    def _source(self) -> torch.Tensor:
        src = self._hidden if self._hidden is not None else self._logits
        assert src is not None   # __init__ guarantees exactly one of hidden / logits
        return src

    @property
    def shape_bt(self) -> tuple[int, int]:
        src = self._source
        return src.shape[0], src.shape[1]

    @property
    def device(self) -> torch.device:
        return self._source.device

    @property
    def hidden(self) -> torch.Tensor | None:
        """Final hidden states (HF positional convention, NOT shifted)."""
        return self._hidden

    def with_temperature(self, t: float) -> LogitsView:
        return LogitsView(self._hidden, self._head, logits=self._logits, temperature=t,
                          vocab_limit=self.vocab_limit) if self._logits is None else \
            LogitsView(logits=self._logits, temperature=t, vocab_limit=self.vocab_limit)

    # ------------------------------------------------------------------ core: raw slice
    def _raw(self, pos_start: int, pos_end: int) -> torch.Tensor:
        """Raw (positional) logits for positions [pos_start, pos_end), fp32, temperature applied."""
        if self._logits is not None:
            out = self._logits[:, pos_start:pos_end]
        else:
            assert self._head is not None and self._hidden is not None
            out = self._head(self._hidden[:, pos_start:pos_end])
        out = out.float()
        if self.vocab_limit is not None:
            out = out[..., : self.vocab_limit]
        if self.temperature != 1.0:
            out = out / self.temperature
        return out

    def _target_slice(self, t_start: int, t_end: int) -> torch.Tensor:
        """Target-aligned logits for targets [t_start, t_end). Target t uses position t-1.
        Row for target 0 is filled with zeros."""
        t_start = max(t_start, 0)
        if t_start == 0:
            body = self._raw(0, max(t_end - 1, 0))
            zero = torch.zeros_like(body[:, :1]) if body.shape[1] else self._raw(0, 1) * 0
            return torch.cat([zero, body], dim=1)
        return self._raw(t_start - 1, t_end - 1)

    def _efficient_ok(self) -> bool:
        """Use custom autograd only for an ordinary linear head with full parameters.

        ZeRO-3 parameters stay marked as partitioned (``compat.ds_api``) even while temporarily gathered. Recomputing them
        directly in custom backward would bypass the engine's parameter lifecycle.
        """
        if (not LOGITS_VIEW_EFFICIENT or self._hidden is None or not torch.is_grad_enabled()
                or not self._hidden.requires_grad or getattr(self._head, "_lmkd_fsdp_managed", False)):
            return False
        ok = type(self._head) is nn.Linear and all(
            p is None or (p.numel() > 0 and not ds_api.is_partitioned_param(p))
            for p in (self._head.weight, self._head.bias)
        )
        if not ok and type(self._head) not in _WARNED_HEADS:
            logging.getLogger(__name__).warning(
                "LogitsView efficient fallback to module forward for unsupported or "
                "sharded head %s; DeepSpeed parameter availability is managed by the engine",
                type(self._head).__name__,
            )
            _WARNED_HEADS.add(type(self._head))
        return ok

    def _apply_chunked(self, chunk_fn, chunk_size, skip_mask=None):
        h, head = self._hidden, self._head
        assert h is not None and head is not None and isinstance(head, nn.Linear)
        h_t = torch.cat([h.new_zeros(h[:, :1].shape), h[:, :-1]], 1)
        def aligned(x, s, e):
            # Public API defines target zero as zero logits, including biased heads.
            if s == 0:
                x = torch.cat([x[:, :1] * 0, x[:, 1:]], 1)
            return chunk_fn(x, s, e)
        return ChunkedHeadFunction.apply(h_t, head.weight, head.bias, aligned,
                                         chunk_size, self.temperature, self.vocab_limit, skip_mask)

    # ------------------------------------------------------------------ public API
    def full(self) -> torch.Tensor:
        """[B,T,V] fp32 target-aligned. EXPENSIVE: only when a loss truly needs it (comment why)."""
        _, T = self.shape_bt
        return self._target_slice(0, T)

    def positional(self) -> torch.Tensor:
        """[B,T,V] fp32 in HF convention (NOT shifted: row t predicts token t+1), as returned by
        ``model(...).logits``. For ported code (loss tiers 0/1); same cost as :meth:`full`."""
        _, T = self.shape_bt
        return self._raw(0, T)

    def chunks(self, size: int = 1024) -> Iterator[LogitChunk]:
        _, T = self.shape_bt
        for s in range(0, T, size):
            e = min(T, s + size)
            yield LogitChunk(s, e, self._target_slice(s, e))

    def target_logp(self, targets: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
        """[B,T] log p(targets[t]) (targets = labels; -100 positions return 0)."""
        if self._efficient_ok():
            def fn(x, s, e):
                tgt = targets[:, s:e]
                return x.log_softmax(-1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) * (tgt != -100)
            return self._apply_chunked(fn, chunk_size, targets != -100)
        out = []
        for ch in self.chunks(chunk_size):
            tgt = targets[:, ch.start:ch.end]
            safe = tgt.clamp(min=0)
            lp = torch.log_softmax(ch.logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
            out.append(lp * (tgt != -100))
        return torch.cat(out, dim=1)

    def gather(self, ids: torch.Tensor, chunk_size: int = 1024, log_softmax: bool = True) -> torch.Tensor:
        """ids [B,T,K] -> [B,T,K] (log-probs if log_softmax else logits) at the given vocab ids."""
        out = []
        for ch in self.chunks(chunk_size):
            x = torch.log_softmax(ch.logits, -1) if log_softmax else ch.logits
            out.append(x.gather(-1, ids[:, ch.start:ch.end]))
        return torch.cat(out, dim=1)

    def topk(self, k: int, chunk_size: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (ids [B,T,k], logp [B,T,k]) of the top-k tokens."""
        ids, lps = [], []
        for ch in self.chunks(chunk_size):
            lp = torch.log_softmax(ch.logits, -1)
            v, i = lp.topk(k, dim=-1)
            ids.append(i)
            lps.append(v)
        return torch.cat(ids, 1), torch.cat(lps, 1)

    def entropy(self, chunk_size: int = 1024) -> torch.Tensor:
        """[B,T] entropy of the predictive distribution for each target."""
        out = []
        for ch in self.chunks(chunk_size):
            lp = torch.log_softmax(ch.logits, -1)
            out.append(-(lp.exp() * lp).sum(-1))
        return torch.cat(out, 1)

    def at(self, target_positions: torch.Tensor) -> torch.Tensor:
        """Logits [N,V] for selected (batch, target_pos) pairs given as LongTensor [N,2]."""
        b, t = target_positions[:, 0], target_positions[:, 1]
        if self._logits is not None:
            x = self._logits[b, t - 1]
        else:
            assert self._head is not None and self._hidden is not None
            x = self._head(self._hidden[b, t - 1])
        x = x.float()
        if self.vocab_limit is not None:
            x = x[..., : self.vocab_limit]
        return x / self.temperature if self.temperature != 1.0 else x


# ---------------------------------------------------------------------- chunked divergence
TokenFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""fn(student_logits_chunk [B,t,V] fp32, teacher_logits_chunk [B,t,V] fp32) -> per-token [B,t]."""


def chunked_token_divergence(
    student: LogitsView,
    teacher: LogitsView,
    fn: TokenFn,
    mask: torch.Tensor | None = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Per-target divergence [B,T] computed chunk by chunk along T.

    ``mask`` [B,T] (optional) zeroes unsupervised positions; chunks where the mask is all zero
    are skipped entirely (saves compute on prompts). Teacher chunks are computed under no_grad.
    Linear student heads recompute chunks in backward without saving logits.
    """
    if student._efficient_ok():
        def chunk_fn(x, s, e):
            with torch.no_grad():
                t = teacher._target_slice(s, e)
            v = min(x.shape[-1], t.shape[-1])
            val = fn(x[..., :v], t[..., :v])
            return val if mask is None else val * mask[:, s:e]
        return student._apply_chunked(chunk_fn, chunk_size, mask)
    B, T = student.shape_bt
    parts: list[torch.Tensor] = []
    for s in range(0, T, chunk_size):
        e = min(T, s + chunk_size)
        if mask is not None and not bool(mask[:, s:e].any()):
            parts.append(torch.zeros(B, e - s, device=student.device, dtype=torch.float32))
            continue
        s_chunk = student._target_slice(s, e)
        with torch.no_grad():
            t_chunk = teacher._target_slice(s, e)
        V = min(s_chunk.shape[-1], t_chunk.shape[-1])  # vocab padding rows may differ
        val = fn(s_chunk[..., :V], t_chunk[..., :V])
        if mask is not None:
            val = val * mask[:, s:e]
        parts.append(val.float())
    return torch.cat(parts, dim=1)


def logits_view_kl_check(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    """Debug helper: full-vocab KL(a||b) per token from raw logits."""
    return (F.softmax(a, -1) * (F.log_softmax(a, -1) - F.log_softmax(b, -1))).sum(-1)
