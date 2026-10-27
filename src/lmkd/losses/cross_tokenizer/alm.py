"""ALM chunk-level binarized probability matching (IMPACT/code/criterions/ALM.py)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.losses.cross_tokenizer._common import get_vocab_map
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.logits_view import LogitsView
from lmkd.teachers.signals import TeacherSignalSpec


def _last_index(matrix: torch.Tensor) -> torch.Tensor:
    present = matrix.bool()
    last = present.clone()
    last[:, :-1] = present[:, :-1] & ~present[:, 1:]
    return last.long().argmax(-2)


def _alignment(chunks: torch.Tensor, attention: torch.Tensor, prompt: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("ALM needs at least one aligned chunk")
    if bool((chunks >= count).any()):
        raise ValueError("ALM chunk id exceeds the selected chunk dimension")
    positions = torch.arange(chunks.shape[1], device=chunks.device)[None, :]
    loss_mask = (positions >= prompt[:, None]) & attention.bool() & (chunks >= 0)
    matrix = F.one_hot(chunks.clamp_min(0), count).float() * loss_mask[..., None]
    assert not bool(matrix[:, 0].any()), "ALM requires position 0 to belong to the prompt"
    return matrix


def _space_logp(view: LogitsView, mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """For positional p, read the target-aligned distribution at p+1."""
    selected = []
    for chunk in view.chunks(chunk_size):
        probs = chunk.logits.softmax(-1)
        chosen = mask.to(probs.device, dtype=probs.dtype)
        if chosen.numel() < probs.shape[-1]:
            chosen = F.pad(chosen, (0, probs.shape[-1] - chosen.numel()))
        chosen = chosen[:probs.shape[-1]]
        selected.append(torch.log(probs @ chosen + 1e-10).clamp(max=0))
    target = torch.cat(selected, dim=1)
    return torch.cat([target[:, 1:], torch.zeros_like(target[:, :1])], dim=1)


def _merge(t_main: torch.Tensor, s_main: torch.Tensor, t_space: torch.Tensor,
           s_space: torch.Tensor, counts: torch.Tensor, a: torch.Tensor,
           threshold: float) -> tuple[torch.Tensor, ...]:
    b, c = t_space.shape
    selected = torch.exp(t_space) > threshold
    suffix = torch.cumsum(selected.flip(-1), -1).flip(-1)
    dest = suffix.max(-1, keepdim=True).values - suffix
    merge = torch.zeros((b * c, c), dtype=a.dtype, device=a.device)
    rows = torch.arange(b * c, device=a.device)
    merge[rows, dest.reshape(-1).long()] = (counts > 0).reshape(-1).to(a.dtype)
    merge = merge.reshape(b, c, c)
    last = _last_index(merge)
    return ((t_main[:, None] @ merge.float()).squeeze(1),
            (s_main[:, None] @ merge.float()).squeeze(1),
            t_space.gather(-1, last), s_space.gather(-1, last),
            (counts[:, None].float() @ merge.float()).squeeze(1))


def _binary_ce(true_logp: torch.Tensor, pred_logp: torch.Tensor, temp: float, eps: float) -> torch.Tensor:
    y = true_logp.float() / temp - eps
    pred = pred_logp.float() / temp - eps
    half = -torch.log(torch.tensor(2.0, device=pred.device, dtype=pred.dtype))
    log1m = torch.where(pred < half, torch.log1p(-torch.exp(pred)), torch.log(-torch.expm1(pred)))
    return -(torch.exp(y) * pred + (-torch.expm1(y) * log1m))


def _finish(t_main: torch.Tensor, s_main: torch.Tensor, t_space: torch.Tensor,
            s_space: torch.Tensor, counts: torch.Tensor, mode: str,
            temp: float, eps: float) -> torch.Tensor:
    valid = (counts > 0).float()
    s_main = s_main * valid
    t_space = t_space * valid
    s_space = s_space * valid
    if "append_space" in mode:
        last = torch.flip(torch.cumsum(torch.flip(counts > 0, [-1]).long(), -1) == 1, [-1])
        s_main = s_main + s_space * last.float()
        t_main = t_main + t_space * last.float()
    neg = torch.tensor(-0.7 * torch.finfo(torch.float32).max, device=s_main.device)
    s_full = torch.where(counts > 0, s_main, neg)
    t_full = torch.where(counts > 0, t_main, neg)
    numerator = (counts > 0).float()
    return (_binary_ce(t_full, s_full, temp, eps) * numerator).mean() / (numerator.mean() + eps)


@CRITERIONS.register("alm", source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/ALM.py:133-289")
class ALM(Criterion):
    """BCE of binarized student/teacher chunk log probabilities, optionally merging and appending spaces.

    Source: IMPACT/code/criterions/ALM.py:133-289. legacy_scale reproduces the original
    erroneous token-denominator division; the default treats ALM as a micro-batch mean.
    """

    requires = ("align.chunk_s", "align.chunk_t")
    needs_teacher = True

    class Config(BaseConfig):
        binarization_temp: float = 100.0
        bias_threshold: float = 0.1
        mode: str = "merge_by_space_prob+append_space"
        epsilon: float = 1e-5
        chunk_dim: str = "shared_length"
        chunk_size: int = 1024
        legacy_scale: bool = False

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"), teacher=SideRequest(logits="view"),
                              teacher_signal=TeacherSignalSpec(kind="full"))

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.logits is None:
            raise ValueError("ALM requires full teacher logits")
        vm = get_vocab_map(ctx)
        if vm.space_mask_s is None or vm.space_mask_t is None:
            raise ValueError("ALM needs load_vocab_map(space_mask_mode='space+tab+newline+special')")
        batch = ctx.batch
        ids_s, ids_t = batch["student.input_ids"], batch["teacher.input_ids"]
        chunks_s, chunks_t = batch["align.chunk_s"], batch["align.chunk_t"]
        if self.cfg.chunk_dim == "shared_length":
            count = min(ids_s.shape[1], ids_t.shape[1])
        elif self.cfg.chunk_dim == "max_chunk":
            count = max(int(chunks_s.max()), int(chunks_t.max())) + 1
        else:
            raise ValueError(f"unknown ALM chunk_dim {self.cfg.chunk_dim!r}")
        a = _alignment(chunks_s, batch["student.attention_mask"], batch["student.prompt_len"], count)
        b = _alignment(chunks_t, batch["teacher.attention_mask"], batch["teacher.prompt_len"], count)
        s_view, t_view = ctx.student_logits(), ctx.teacher.logits
        s_ids = ids_s.masked_fill(~batch["student.attention_mask"].bool(), -100)
        t_ids = ids_t.masked_fill(~batch["teacher.attention_mask"].bool(), -100)
        s_main = (s_view.target_logp(s_ids, self.cfg.chunk_size)[:, None] @ a).squeeze(1).clamp(max=0)
        t_main = (t_view.target_logp(t_ids, self.cfg.chunk_size)[:, None] @ b).squeeze(1).clamp(max=0)
        s_space = _space_logp(s_view, vm.space_mask_s, self.cfg.chunk_size).gather(-1, _last_index(a))
        t_space = _space_logp(t_view, vm.space_mask_t, self.cfg.chunk_size).gather(-1, _last_index(b))
        counts = b[:, 1:].sum(-2)
        if "merge_by_space_prob" in self.cfg.mode:
            t_main, s_main, t_space, s_space, counts = _merge(
                t_main, s_main, t_space, s_space, counts, a, self.cfg.bias_threshold)
        loss = _finish(t_main, s_main, t_space, s_space, counts, self.cfg.mode,
                       self.cfg.binarization_temp, self.cfg.epsilon)
        return LossOutput({"alm": LossTerm(loss, "token_sum" if self.cfg.legacy_scale else "mean")})
