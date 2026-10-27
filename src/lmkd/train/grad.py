"""Gradient intervention (DESIGN §8.2).

Three levels:
  1. token/sample level  -> weights & masks inside criteria (no code here)
  2. activation level    -> TensorGradHook: modifies dL/dh of a captured activation
  3. parameter level     -> GradTransform chain: runs after backward, before optimizer.step

Implemented: GradTransform ``clip_norm``, ``log_grad_norm``, ``pcgrad``; TensorGradHook infra:
``Standard.loss`` resolves ``cfg.target`` on the student SideOutput after the forward and calls
``tensor.register_hook``; the hook's ``features()`` is merged into the FeatureRequest so the target
tensor is captured. Concrete hooks: ``probe`` and ``project_out_topk_svd``.
"""
from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import GRAD_HOOKS, GRAD_TRANSFORMS


@dataclass
class ParamWithGroup:
    """A trainable parameter with its name, parameter group and optimizer, for gradient transforms."""
    name: str
    param: torch.nn.Parameter
    group: str
    optimizer: str


@dataclass
class StepContext:
    """Arguments of a gradient transform: runtime and the step's metrics dict."""
    rt: Any
    metrics: dict
    term_grads: dict | None = None   # filled by multi_objective_grad strategy (task T14/T24)


class GradTransform(Plugin):
    """Edits parameter gradients before ``optimizer.step`` (clip, project, combine objectives)."""
    needs_full_grad: ClassVar[bool] = False   # True -> cannot run on ZeRO-2/3 sharded grads without gather

    def __call__(self, params: list[ParamWithGroup], ctx: StepContext) -> None:
        raise NotImplementedError


@GRAD_TRANSFORMS.register("clip_norm")
class ClipNorm(GradTransform):
    """Global (or per-group) gradient norm clipping. Logs grad_norm before clipping."""

    class Config(BaseConfig):
        max_norm: float = 1.0
        groups: list[str] | None = None   # None = all groups

    def __call__(self, params, ctx):
        ps = [p.param for p in params if p.param.grad is not None and (self.cfg.groups is None or p.group in self.cfg.groups)]
        if not ps:
            return
        norm = ctx.rt.engine.clip_grad_norm(ps, self.cfg.max_norm)
        ctx.metrics["grad/norm"] = norm.detach() if isinstance(norm, torch.Tensor) else torch.tensor(norm)


@GRAD_TRANSFORMS.register("log_grad_norm")
class LogGradNorm(GradTransform):
    """Logs gradient norms (total and per group) without changing gradients."""
    class Config(BaseConfig):
        per_group: bool = True

    def __call__(self, params, ctx):
        by_group: dict[str, list[torch.Tensor]] = {}
        for p in params:
            if p.param.grad is not None:
                by_group.setdefault(p.group, []).append(p.param.grad.detach().float().norm())
        for g, norms in by_group.items():
            ctx.metrics[f"grad/norm_{g}"] = torch.stack(norms).norm()


def pcgrad_combine(grads: list[list[torch.Tensor]], seed: int = 0) -> list[torch.Tensor]:
    """Project conflicting term gradients, then sum them per parameter.

    For each term i, project a copy of g_i against ORIGINAL g_j in a seeded random order when
    <g_i, g_j> < 0, using a dot product over all parameters. PCGrad, Yu et al. (2020).
    """
    if not grads:
        return []
    n_params = len(grads[0])
    if any(len(term) != n_params for term in grads):
        raise ValueError("every PCGrad term must contain one gradient per parameter")
    original = [[g.detach().clone() for g in term] for term in grads]
    projected = [[g.clone() for g in term] for term in original]
    rng = random.Random(seed)
    for i, term in enumerate(projected):
        others = [j for j in range(len(original)) if j != i]
        rng.shuffle(others)
        for j in others:
            reference = original[j]
            dot = sum(float((a.float() * b.float()).sum()) for a, b in zip(term, reference))
            if dot >= 0:
                continue
            squared_norm = sum(float(b.float().square().sum()) for b in reference)
            if squared_norm > 0:
                coefficient = dot / squared_norm
                term[:] = [a - coefficient * b for a, b in zip(term, reference)]
    return [sum((term[j] for term in projected), torch.zeros_like(original[0][j]))
            for j in range(n_params)]


@GRAD_TRANSFORMS.register("pcgrad", paper="PCGrad (Yu et al. 2020)")
class PCGrad(GradTransform):
    """Replace parameter grads with PCGrad's sum of conflict-projected term gradients.

    Formula: g_i <- g_i - min(0, <g_i,g_j>) g_j / ||g_j||², then sum_i g_i.
    Source: Yu et al. (2020), PCGrad. Requires T14's multi_objective_grad strategy.
    """
    needs_full_grad: ClassVar[bool] = True

    class Config(BaseConfig):
        terms: list[str]

    def __call__(self, params: list[ParamWithGroup], ctx: StepContext) -> None:
        if ctx.term_grads is None:
            raise RuntimeError("pcgrad needs step: multi_objective_grad (T14)")
        missing = [name for name in self.cfg.terms if name not in ctx.term_grads]
        if missing:
            raise KeyError(f"pcgrad missing term gradients: {missing}")
        combined = pcgrad_combine([ctx.term_grads[name] for name in self.cfg.terms])
        if len(combined) != len(params):
            raise ValueError("pcgrad term gradients must match the trainable parameter list")
        for wrapped, grad in zip(params, combined):
            wrapped.param.grad = grad.to(device=wrapped.param.device, dtype=wrapped.param.dtype)


@dataclass
class HookContext:
    """Arguments of an activation-gradient hook: runtime, micro-batch, state."""
    rt: Any
    batch: Any          # micro-batch
    state: Any          # TrainState (use state.log(key, value) for metrics)
    micro_index: int = 0


class TensorGradHook(Plugin):
    """Modifies/reads dL/d(tensor) of a captured student activation. Return None = unchanged."""

    class Config(BaseConfig):
        target: str     # "student.hidden[-8]" | "student.last_hidden" | "student.input_embeds"

    def features(self):
        from lmkd.models.forward import FeatureRequest, parse_target

        side, _, req = parse_target(self.cfg.target)
        if side != "student":
            raise ValueError("grad hooks only apply to the student (teacher runs without grad)")
        return FeatureRequest(student=req)

    def __call__(self, grad: torch.Tensor, ctx: HookContext) -> torch.Tensor | None:
        raise NotImplementedError


def attach_grad_hooks(hooks: list[TensorGradHook], student_view: Any, ctx: HookContext) -> None:
    """Register every hook on its target tensor of ``student_view`` (a SideOutput)."""
    from lmkd.models.forward import parse_target

    for h in hooks:
        _, body, _ = parse_target(h.cfg.target)
        t = student_view.resolve(body)
        if t.requires_grad:
            t.register_hook(lambda g, h=h: h(g, ctx))


def _micro_count(ctx: HookContext) -> int:
    train = ctx.rt.config.train
    return max(1, train.batch_size // (train.micro_batch_size * ctx.rt.engine.world_size))


def _mean_metric(ctx: HookContext, key: str, value: float) -> None:
    metrics = ctx.state.extra.setdefault("metrics", {})
    metrics[key] = metrics.get(key, 0.0) + value / _micro_count(ctx)


def _warn_frozen_input_embeds(hook: TensorGradHook, rt: Any) -> None:
    if hook.cfg.target != "student.input_embeds":
        return
    embedding = rt.student_adapter.embed_tokens(rt.student_model)
    if not any(p.requires_grad for p in embedding.parameters()) and not getattr(hook, "_warned_frozen_input", False):
        warnings.warn("student.input_embeds is frozen; its gradient hook may be skipped", RuntimeWarning,
                      stacklevel=2)
        hook._warned_frozen_input = True  # type: ignore[attr-defined]  # one-time warning flag


@GRAD_HOOKS.register("probe",
                     source="Long-CoT-Disitllation/SpectralGuidedLearning/src/sgl/training/transition_trainer.py:54-91")
class GradProbe(TensorGradHook):
    """Read ||dL/dh||₂, mean|dL/dh|, and max|dL/dh| without changing the gradient.

    Norm and absolute mean are averaged over micro-batches; absolute max is the step maximum.
    Source: SGL transition_trainer.py:54-91. Per-term probing belongs inside the cooperating
    transition_loss criterion (T20), following SGL's source/hidden hook pattern.
    """

    class Config(TensorGradHook.Config):
        name: str | None = None

    def setup(self, rt: Any) -> None:
        _warn_frozen_input_embeds(self, rt)

    def __call__(self, grad: torch.Tensor, ctx: HookContext) -> None:
        name = self.cfg.name or self.cfg.target
        g = grad.detach().float()
        prefix = f"gradprobe/{name}"
        _mean_metric(ctx, f"{prefix}/norm", float(g.norm()))
        _mean_metric(ctx, f"{prefix}/absmean", float(g.abs().mean()))
        metrics = ctx.state.extra.setdefault("metrics", {})
        key = f"{prefix}/absmax"
        metrics[key] = max(metrics.get(key, 0.0), float(g.abs().max()))
        return None


@GRAD_HOOKS.register("project_out_topk_svd")
class ProjectOutTopK(TensorGradHook):
    """Remove dL/dh's top-k right-singular directions from each valid token gradient.

    Formula: G' = G - (G Pᵀ) P, where P contains the leading right singular vectors of G.
    The removed fraction is averaged over micros.
    """

    class Config(TensorGradHook.Config):
        k: int = 4
        method: str = "lowrank"
        mask_padding: bool = True

    def setup(self, rt: Any) -> None:
        _warn_frozen_input_embeds(self, rt)

    def __call__(self, grad: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        if self.cfg.k < 0:
            raise ValueError("project_out_topk_svd.k must be non-negative")
        if self.cfg.method not in ("lowrank", "full"):
            raise ValueError("project_out_topk_svd.method must be 'lowrank' or 'full'")
        d = grad.shape[-1]
        flat = grad.reshape(-1, d).float()
        if self.cfg.mask_padding:
            valid = ctx.batch["student.attention_mask"].reshape(-1).bool()
        else:
            valid = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
        source = flat[valid]
        rank = min(self.cfg.k, source.shape[0], d)
        result = flat.clone()
        if rank:
            if self.cfg.method == "full":
                _, _, vh = torch.linalg.svd(source, full_matrices=False)
                directions = vh[:rank]
            else:
                _, _, v = torch.svd_lowrank(source, q=min(self.cfg.k + 4, source.shape[0], d))
                directions = v[:, :rank].T
            result[valid] = source - (source @ directions.T) @ directions
        total_sq = flat.square().sum()
        removed = (1.0 - result.square().sum() / total_sq).clamp(0.0, 1.0) if total_sq > 0 else flat.new_zeros(())
        _mean_metric(ctx, f"gradhook/{self.cfg.target}/removed_frac", float(removed))
        return result.to(dtype=grad.dtype).reshape_as(grad)
