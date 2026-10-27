"""Long-CoT criteria: per_source_objective (NLL/DFT) and transition_loss."""
from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field, field_validator

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm, ModelPairInfo
from lmkd.models.forward import FeatureRequest, SideRequest


class TransitionPredictor(nn.Module):
    """f(s) = W2 GELU(W1 LN(s)); d -> hidden -> d. Trained fully (own param group), dropped at inference."""

    def __init__(self, d_model: int, hidden: int = 1024) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Match weight dtype (e.g. bf16/fp32) and return fp32 for stable cosine computation
        s = s.to(self.up.weight.dtype)
        return self.down(F.gelu(self.up(self.norm(s)))).float()


def compute_transition_hidden_index(layer: float | int | str, num_layers: int) -> int:
    """Map a zero-based SGL block to HF hidden index (0=embeddings, blocks start at 1).

    Fractions select round(fraction * num_layers); negative indices keep HF semantics.
    """
    if layer == "two_thirds" or (isinstance(layer, float) and abs(layer - 2.0 / 3.0) < 0.005):
        block = round(2.0 * num_layers / 3.0)
        return block + 1
    if isinstance(layer, float):
        block = round(layer * num_layers)
        return block + 1
    if isinstance(layer, int) and not isinstance(layer, bool):
        return layer if layer < 0 else layer + 1
    raise ValueError(f"invalid layer spec {layer!r}")


def pool_step_targets(
    hidden: torch.Tensor, step_id: torch.Tensor, num_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool hidden [T, d] per step and remove the per-sequence mean."""
    hidden = hidden.detach().float()
    valid = step_id >= 0
    sums = hidden.new_zeros(num_steps, hidden.size(-1))
    sums.index_add_(0, step_id[valid], hidden[valid])
    counts = torch.bincount(step_id[valid], minlength=num_steps).clamp_min(1).unsqueeze(-1)
    z_raw = sums / counts.to(hidden.dtype)
    z_tilde = z_raw - z_raw.mean(0, keepdim=True)
    return z_raw, z_tilde


def shuffled_targets(pair_src: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Control: for source step i pick a random j != i+1 from the same sequence."""
    draws = torch.randint(0, num_steps - 1, pair_src.shape, device=pair_src.device)
    return draws + (draws >= pair_src + 1).long()


def transition_loss_one(
    hidden: torch.Tensor,
    step_id: torch.Tensor,
    step_end: torch.Tensor,
    pair_src: torch.Tensor,
    predictor: nn.Module,
    shuffle_targets: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute transition loss for one sequence."""
    num_steps = int(step_end.numel())
    if pair_src.numel() == 0 or num_steps < 2:
        zero = predictor(hidden[:1].float()).sum() * 0.0
        return zero, {"pairs": 0, "cos": 0.0}

    _, z_tilde = pool_step_targets(hidden, step_id, num_steps)
    targets = F.normalize(z_tilde, dim=-1)

    tgt_index = shuffled_targets(pair_src, num_steps) if shuffle_targets else pair_src + 1
    positions = step_end[pair_src]
    source = hidden[positions].float()
    pred = F.normalize(predictor(source), dim=-1)
    cos = (pred * targets[tgt_index]).sum(-1)
    loss = (1.0 - cos).mean()

    stats = {
        "pairs": int(pair_src.numel()),
        "cos": float(cos.detach().mean()),
    }
    return loss, stats


@CRITERIONS.register(
    "per_source_objective",
    source="Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/training/losses.py:23-88; train.py:59-88",
)
class PerSourceObjective(Criterion):
    """sum_t coef_t * nll_t * mask_t ; coef = w_t * (sg(p_t) if obj(source_t)=='dft' else 1)."""

    SOURCE_IDS: ClassVar[dict[str, int]] = {"prompt": 0, "prefix": 1, "cont": 2, "junction": 3}

    class Config(BaseConfig):
        default: str = "nll"
        objectives: dict[str, str] = Field(default_factory=dict)
        use_token_weights: bool = True
        chunk_size: int = 1024

        @field_validator("default")
        @classmethod
        def _check_default(cls, v: str) -> str:
            if v not in ("nll", "dft"):
                raise ValueError(f"default must be 'nll' or 'dft', got '{v}'")
            return v

        @field_validator("objectives")
        @classmethod
        def _check_objectives(cls, v: dict[str, str]) -> dict[str, str]:
            allowed_sources = {"prompt", "prefix", "cont", "junction"}
            for k, val in v.items():
                if k not in allowed_sources:
                    raise ValueError(f"unknown source '{k}'; allowed: {sorted(allowed_sources)}")
                if val not in ("nll", "dft"):
                    raise ValueError(f"objective value must be 'nll' or 'dft', got '{val}' for source '{k}'")
            return v

    @classmethod
    def required_keys(cls, cfg: Any) -> tuple[str, ...]:
        if cfg is not None and getattr(cfg, "objectives", None):
            return ("seq.token_source",)
        return ()

    def features(self) -> FeatureRequest:
        return FeatureRequest(student=SideRequest(logits="view"))

    def forward(self, ctx: LossContext) -> LossOutput:
        labels = ctx.batch["student.labels"]
        mask = ctx.supervised_mask()
        logp = ctx.student_logits().target_logp(labels, chunk_size=self.cfg.chunk_size)
        nll = -logp

        obj_default = 1 if self.cfg.default == "dft" else 0
        obj = torch.full_like(labels, obj_default)
        if self.cfg.objectives:
            token_source = ctx.batch["seq.token_source"]
            for name, o in self.cfg.objectives.items():
                obj[token_source == self.SOURCE_IDS[name]] = 1 if o == "dft" else 0

        coef = torch.where(obj == 1, torch.exp(-nll).detach(), torch.ones_like(nll))
        if self.cfg.use_token_weights and "seq.token_weights" in ctx.batch:
            coef = coef * ctx.batch["seq.token_weights"].to(coef.dtype)

        loss_term = (nll * coef * mask).sum()
        mask_bool = mask > 0
        if mask_bool.any():
            dft_frac = (obj == 1).float()[mask_bool].mean().detach()
        else:
            dft_frac = torch.tensor(0.0, device=labels.device)

        return LossOutput(
            terms={"obj": LossTerm(loss_term, "token_sum")},
            metrics={"dft_frac": dft_frac},
        )


@CRITERIONS.register(
    "transition_loss",
    source="Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/training/transition_loss.py:21-140; transition_trainer.py:94-202",
)
class TransitionLoss(Criterion):
    """L_trans = mean_P(1 - cos(normalize(f(h[step_end[i]])), normalize(z_tilde[i+1])))."""

    requires = ("seq.step_id", "seq.step_end", "seq.pair_src")
    supports_packing = False

    class Config(BaseConfig):
        layer: float | int | str = "two_thirds"
        predictor_hidden: int = 1024
        shuffled_targets: bool = False
        warmup_ratio: float = 0.0
        grad_probe_every: int = 0

        @field_validator("grad_probe_every")
        @classmethod
        def _check_probe(cls, value: int) -> int:
            if value != 0:
                raise ValueError("grad_probe_every is not implemented; configure a grad.hooks probe instead")
            return value

        @field_validator("layer")
        @classmethod
        def _check_layer(cls, v: Any) -> Any:
            if isinstance(v, str):
                if v != "two_thirds":
                    raise ValueError(f"layer str must be 'two_thirds', got '{v}'")
                return v
            if isinstance(v, float):
                if not (0.0 < v < 1.0):
                    raise ValueError(f"layer float must be in (0, 1), got {v}")
                return v
            if isinstance(v, int) and not isinstance(v, bool):
                if v < 1:
                    raise ValueError(f"layer int must be >= 1, got {v}")
                return v
            raise ValueError(f"layer must be float in (0, 1), int >= 1, or 'two_thirds', got {v!r}")

    def __init__(self, cfg: BaseConfig | None = None) -> None:
        super().__init__(cfg)
        self._hidden_index: int | None = None
        self._L: int | None = None

    def build_modules(self, info: ModelPairInfo, modules: Any) -> None:
        self._L = info.student_layers
        self._hidden_index = compute_transition_hidden_index(self.cfg.layer, self._L)
        pred = TransitionPredictor(d_model=info.student_hidden, hidden=self.cfg.predictor_hidden)
        modules.add(f"{self.alias}__predictor", pred, group="trans_predictor", owner=self.alias)

    def features(self) -> FeatureRequest:
        if self._hidden_index is not None:
            idx = self._hidden_index
        elif isinstance(self.cfg.layer, int) and not isinstance(self.cfg.layer, bool):
            idx = self.cfg.layer
        else:
            num_layers = self._L if self._L is not None else 36
            idx = compute_transition_hidden_index(self.cfg.layer, num_layers)
        return FeatureRequest(student=SideRequest(hidden=(idx,)))

    def forward(self, ctx: LossContext) -> LossOutput:
        if self._hidden_index is None:
            raise RuntimeError(f"{self.alias}: forward() called before _hidden_index is set")
        H = ctx.student.hidden(self._hidden_index)
        pred = ctx.modules.get(f"{self.alias}__predictor")

        losses: list[torch.Tensor] = []
        stats_list: list[dict[str, Any]] = []
        B = H.size(0)

        step_id_all = ctx.batch["seq.step_id"]
        step_end_all = ctx.batch["seq.step_end"]
        pair_src_all = ctx.batch["seq.pair_src"]

        for b in range(B):
            se = step_end_all[b]
            ps = pair_src_all[b]
            sid = step_id_all[b]
            if not isinstance(se, torch.Tensor):
                se = torch.as_tensor(se, device=H.device, dtype=torch.long)
            else:
                se = se.to(H.device)
            if not isinstance(ps, torch.Tensor):
                ps = torch.as_tensor(ps, device=H.device, dtype=torch.long)
            else:
                ps = ps.to(H.device)
            sid = sid.to(H.device)

            loss_b, stats_b = transition_loss_one(
                H[b],
                sid,
                se[se >= 0],
                ps[ps >= 0],
                pred,
                shuffle_targets=self.cfg.shuffled_targets,
            )
            losses.append(loss_b)
            stats_list.append(stats_b)

        loss_trans = torch.stack(losses).mean()

        lam = 1.0
        if self.cfg.warmup_ratio > 0 and ctx.state is not None:
            total_steps = ctx.state.total_steps
            step = ctx.state.step
            warmup_steps = max(1, int(self.cfg.warmup_ratio * total_steps))
            lam = min(1.0, float(step) / float(warmup_steps))

        scored = [s for s in stats_list if s["pairs"] > 0]
        mean_cos = sum(s["cos"] for s in scored) / len(scored) if scored else 0.0
        avg_pairs = sum(s["pairs"] for s in stats_list) / max(1, len(stats_list))

        return LossOutput(
            terms={"trans": LossTerm(lam * loss_trans, "mean", log_value=loss_trans.detach())},
            metrics={
                "trans_cos": torch.tensor(mean_cos, device=H.device),
                "trans_pairs": torch.tensor(float(avg_pairs), device=H.device),
                "lambda": torch.tensor(float(lam), device=H.device),
            },
        )
