"""ResidualKD projector pretraining and residual student loss from IMPACT."""
from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.core.state import TrainState
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo, PrepareContext
from lmkd.losses.cross_tokenizer._common import DetachedLinear
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.models.logits_view import LogitsView
from lmkd.models.modules import ModuleRegistry
from lmkd.teachers.signals import TeacherSignalSpec


class ProjectorTA(nn.Module):
    """Teacher-to-anchor encoder and anchor-to-teacher decoder, both bias-free."""

    def __init__(self, d_teacher: int, d_anchor: int) -> None:
        super().__init__()
        self.P_TA = nn.Linear(d_teacher, d_anchor, bias=False)
        self.P_AT = nn.Linear(d_anchor, d_teacher, bias=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(hidden)
        return encoded, self.P_AT(encoded)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project teacher hidden states into the anchor space."""
        return self.P_TA(hidden.to(self.P_TA.weight.dtype))


class ProjectorSA(nn.Module):
    """Student-to-anchor projection learned in ResidualKD stage 2."""

    def __init__(self, d_student: int, d_anchor: int) -> None:
        super().__init__()
        self.P_SA = nn.Linear(d_student, d_anchor, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.P_SA(hidden.to(self.P_SA.weight.dtype))


class ProjectorAS(nn.Module):
    """Anchor-to-student projection, initialized to zero in stage 2."""

    def __init__(self, d_anchor: int, d_student: int) -> None:
        super().__init__()
        self.P_AS = nn.Linear(d_anchor, d_student, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.P_AS(hidden.to(self.P_AS.weight.dtype))


def positional_labels(labels: torch.Tensor) -> torch.Tensor:
    """Convert target labels to HF positional labels without supervising the last position."""
    shifted = labels.new_full(labels.shape, -100)
    shifted[:, :-1] = labels[:, 1:]
    return shifted


def positional_logits(view: LogitsView, chunk_size: int = 128) -> torch.Tensor:
    """Read positional teacher logits through ``at(p+1)`` including the last position."""
    batch, length = view.shape_bt
    result = []
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        bi = torch.arange(batch, device=view.device).repeat_interleave(end - start)
        pi = torch.arange(start + 1, end + 1, device=view.device).repeat(batch)
        result.append(view.at(torch.stack((bi, pi), -1)).reshape(batch, end - start, -1))
    return torch.cat(result, dim=1)


def residual_core(hs: torch.Tensor, ht: torch.Tensor, ta: ProjectorTA, sa: nn.Module,
                  as_: nn.Module, labels_pos: torch.Tensor, teacher_logits: torch.Tensor,
                  cross: bool, threshold: float,
                  teacher_pad: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port IMPACT residual attention, response mask, beta and hidden correction."""
    with torch.no_grad():
        hta = ta.encode(ht)
    hsa = sa(hs)
    query = hsa / hsa.std(-1, keepdim=True).clamp(min=1e-5)
    key = hta / hta.std(-1, keepdim=True).clamp(min=1e-5)
    scores = query @ key.transpose(-1, -2) / math.sqrt(hsa.shape[-1])
    if teacher_pad is not None:
        scores = scores.masked_fill(~teacher_pad[:, None].bool(), torch.finfo(scores.dtype).min)
    attention = F.softmax(scores, dim=-1)
    projected = as_(attention @ hta)
    response = labels_pos.ne(-100)
    if cross:
        prob = teacher_logits.float().softmax(-1)
        entropy = -(prob * prob.clamp(min=1e-9).log()).sum(-1) / math.log(prob.shape[-1])
        mask = ((attention.float() @ entropy.unsqueeze(-1)).squeeze(-1) > threshold) & response
    else:
        if teacher_logits.shape[1] != hs.shape[1]:
            raise ValueError("same-tokenizer ResidualKD requires equal sequence lengths")
        mask = teacher_logits.argmax(-1).ne(labels_pos) & response
    ratio = hs.float().norm(dim=-1) / projected.float().norm(dim=-1).clamp(min=1e-5)
    per_seq = (ratio * response.float()).sum(-1) / response.sum(-1).clamp(min=1)
    beta = (math.sqrt(hs.shape[-1] / hsa.shape[-1]) * per_seq.mean()).detach().clamp(0.05, 10)
    return hs - beta * projected * mask.unsqueeze(-1), beta, mask


@CRITERIONS.register("residual_pretrain", source="white-box-distillation/cross-tokenizer/IMPACT/code/distillation_residualkd_pretrain.py:40-92")
class ResidualPretrain(Criterion):
    """Stage 1 trains only ProjectorTA through the frozen student's detached LM head."""

    class Config(BaseConfig):
        """Anchor size for stage-1 projector training."""
        d_bottleneck: int = 64
        pretrain_all_tokens: bool = True

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(hidden=(-1,)))

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        modules.add(f"{self.alias}__ta", ProjectorTA(info.student_hidden, self.cfg.d_bottleneck),
                    group="projector", owner=self.alias)

    def forward(self, ctx: LossContext) -> LossOutput:
        head = ctx.rt.student_adapter.lm_head(ctx.rt.student_model)
        if head is None:
            raise ValueError("residual_pretrain needs a student LM head")
        ta: ProjectorTA = ctx.modules.get(f"{self.alias}__ta")
        _, reconstructed = ta(ctx.student.hidden(-1).detach())
        view = LogitsView(hidden=reconstructed, lm_head=DetachedLinear(head))
        if self.cfg.pretrain_all_tokens and "student.attention_mask" in ctx.batch:
            targets = ctx.batch["student.input_ids"]
            mask = ctx.batch["student.attention_mask"].float().clone()
            mask[:, 0] = 0.0
            ce = -(view.target_logp(targets) * mask).sum()
        else:
            labels = ctx.batch["student.labels"]
            ce = -(view.target_logp(labels) * ctx.supervised_mask()).sum()
        return LossOutput({"pretrain": LossTerm(ce, "token_sum")})


@CRITERIONS.register("residualkd", source="white-box-distillation/cross-tokenizer/IMPACT/code/criterions/ResidualKD.py:15-140")
class ResidualKD(Criterion):
    """Stage 2 combines ordinary CE with CE on teacher-guided residual hidden states.

    ``warmup_unit=step`` repairs source micro-step warmup; ``micro`` reproduces it.
    ``mask_teacher_pad`` optionally repairs unmasked teacher padding in cross-model attention.
    """

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        """Residual correction, projector checkpoint and source-compatibility settings."""
        lambda_res: float = 0.5
        lambda_warmup_steps: int = 0
        d_bottleneck: int = 64
        projector_ckpt: str = ""
        projector_ckpt_alias: str = "residual_pretrain"
        cross_tokenizer: bool = False
        entropy_threshold: float = 0.5
        warmup_unit: str = "step"
        mask_teacher_pad: bool = False

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view", hidden=(-1,)),
                              teacher=SideRequest(logits="view", hidden=(-1,)),
                              teacher_signal=TeacherSignalSpec(kind="full"))

    def build_modules(self, info: ModelPairInfo, modules: ModuleRegistry) -> None:
        if info.teacher_hidden is None:
            raise ValueError("ResidualKD needs a teacher hidden size")
        modules.add(f"{self.alias}__ta", ProjectorTA(info.teacher_hidden, self.cfg.d_bottleneck),
                    frozen=True, owner=self.alias)
        modules.add(f"{self.alias}__sa", ProjectorSA(info.student_hidden, self.cfg.d_bottleneck),
                    group="projector", owner=self.alias)
        modules.add(f"{self.alias}__as", ProjectorAS(self.cfg.d_bottleneck, info.student_hidden),
                    group="projector", init="zeros", owner=self.alias)

    def prepare(self, ctx: PrepareContext | None) -> None:
        if not self.cfg.projector_ckpt:
            raise ValueError("projector_ckpt is empty; run ResidualKD stage 1 first")
        path = Path(self.cfg.projector_ckpt)
        if path.is_dir():
            path = path / "checkpoint-final" if (path / "checkpoint-final").is_dir() else path
            path = path / "modules.pt"
        data = torch.load(path, map_location="cpu", weights_only=True)
        source = f"mods.{self.cfg.projector_ckpt_alias}__ta."
        weights = {key[len(source):]: value for key, value in data.items() if key.startswith(source)}
        if not weights:
            raise ValueError(f"stage 1 checkpoint {path} contains no {source} weights")
        if ctx is None:
            raise ValueError("stage 2 needs a runtime to load its projector")
        ctx.rt.modules.get(f"{self.alias}__ta").load_state_dict(weights, strict=True)

    def lambda_at(self, state: TrainState) -> float:
        """Return warmed-up λ measured in optimizer steps or source-compatible micro-steps."""
        if self.cfg.warmup_unit not in ("step", "micro"):
            raise ValueError("warmup_unit must be 'step' or 'micro'")
        count = state.step if self.cfg.warmup_unit == "step" else state.micro_step
        if self.cfg.lambda_warmup_steps <= 0:
            return self.cfg.lambda_res
        return self.cfg.lambda_res * min(1.0, count / self.cfg.lambda_warmup_steps)

    def forward(self, ctx: LossContext) -> LossOutput:
        if ctx.teacher is None or ctx.teacher.side is None or ctx.teacher.logits is None:
            raise ValueError("residualkd needs full teacher logits and hidden states")
        head = ctx.rt.student_adapter.lm_head(ctx.rt.student_model)
        if head is None:
            raise ValueError("residualkd needs a student LM head")
        labels = ctx.batch["student.labels"]
        pos_labels = positional_labels(labels)
        ta: ProjectorTA = ctx.modules.get(f"{self.alias}__ta")
        sa = ctx.modules.get(f"{self.alias}__sa")
        as_ = ctx.modules.get(f"{self.alias}__as")
        teacher_pad = ctx.batch["teacher.attention_mask"] if self.cfg.mask_teacher_pad else None
        teacher_logits = positional_logits(ctx.teacher.logits)
        hidden, beta, mask = residual_core(ctx.student.hidden(-1), ctx.teacher.side.hidden(-1),
                                           ta, sa, as_, pos_labels, teacher_logits,
                                           self.cfg.cross_tokenizer, self.cfg.entropy_threshold, teacher_pad)
        supervised = ctx.supervised_mask()
        sft = -(ctx.student_logits().target_logp(labels) * supervised).sum()
        weight = getattr(head, "weight", None)
        head_dtype = weight.dtype if isinstance(weight, torch.Tensor) else hidden.dtype
        res = -(LogitsView(hidden=hidden.to(head_dtype), lm_head=head).target_logp(labels) * supervised).sum()
        lam = self.lambda_at(ctx.state)
        metrics = {"residualkd_beta": beta.detach(),
                   "residualkd_mask_ratio": (mask.float().sum() / pos_labels.ne(-100).sum().clamp(min=1)).detach(),
                   "residualkd_lambda": torch.tensor(lam, device=beta.device)}
        return LossOutput({"sft": LossTerm((1 - lam) * sft, "token_sum"),
                           "res": LossTerm(lam * res, "token_sum")}, metrics)
