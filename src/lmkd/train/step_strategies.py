"""StepStrategy (Axis E, DESIGN §7.4): everything between "batch for this optimizer step" and
"parameters updated". ``standard`` is fully working and is the base class to extend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from pydantic import Field

from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import STEP_STRATEGIES
from lmkd.data.labels import supervised_mask
from lmkd.losses.aggregator import Norms
from lmkd.train.engine import LocalEngine
from lmkd.train.grad import ClipNorm, StepContext


@dataclass
class StepOutput:
    """Result of one optimizer step: metrics, supervised tokens, samples, whether the step was skipped."""
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    tokens: int = 0
    samples: int = 0
    skipped: bool = False


def count_supervised(b: Batch) -> tuple[float, float]:
    """(supervised target tokens, sequences) of a micro-batch."""
    if "student.labels" in b:
        z_tok = float(supervised_mask(b["student.labels"], b.get("seq.loss_mask")).sum())
    else:
        z_tok = 0.0
    z_seq = float(b["pack.num_seqs"].sum()) if "pack.num_seqs" in b else float(b.batch_size)
    return z_tok, z_seq


class StepStrategy(Plugin):
    """One optimizer step (axis E): micro-batching, backward, gradient transforms, optimizer. Subclasses reuse
    the building blocks defined here.
    """
    def setup(self, rt: Any) -> None:
        self.rt = rt

    # ------------------------------------------------------------- reusable building blocks
    def micro_batches(self, batch: Batch) -> list[Batch]:
        mbs = self.rt.config.train.micro_batch_size
        n = max(1, batch.batch_size // mbs)
        return list(batch.split(n))

    def global_norms(self, micros: list[Batch]) -> Norms:
        eng = self.rt.engine
        z = torch.tensor([0.0, 0.0], device=eng.device)
        for m in micros:
            t, s = count_supervised(m)
            z[0] += t
            z[1] += s
        z = eng.all_reduce_sum(z)
        return Norms(z_tok=float(z[0]), z_seq=float(z[1]), n_micro=len(micros) * eng.world_size)

    def _normalize_micro_metrics(self, metrics: dict[str, torch.Tensor], n_micros: int) -> None:
        if n_micros > 1:
            for k in list(metrics.keys()):
                if k.startswith("metric/"):
                    metrics[k] = metrics[k] / n_micros

    def accumulate(self, micros: list[Batch], norms: Norms) -> dict[str, torch.Tensor]:
        """Forward+backward every micro-batch; grads accumulate in .grad. Returns summed metrics."""
        rt = self.rt
        eng = rt.engine
        metrics: dict[str, torch.Tensor] = {}
        for i, m in enumerate(micros):
            if m is not None and hasattr(m, "to") and getattr(eng, "device", None) is not None:
                m = m.to(eng.device)
            last = i == len(micros) - 1
            with eng.no_sync(rt.student_module, enabled=not last), eng.autocast():
                outs = rt.algorithm.loss(m, rt)
                loss, mets = rt.aggregator.combine(outs, norms)
            # x world_size: data-parallel reduction averages grads; Norms are global sums.
            eng.backward(loss * eng.world_size)
            for k, v in mets.items():
                metrics[k] = metrics.get(k, 0.0) + v.float()
            rt.state.micro_step += 1
        self._normalize_micro_metrics(metrics, len(micros))
        return metrics

    def apply_grad_transforms(self, metrics: dict) -> None:
        """Unscale (fp16) once, then run the GradTransform chain on real-scale grads."""
        self.rt.engine.unscale(self.rt.optimizers)
        ctx = StepContext(rt=self.rt, metrics=metrics)
        for gt in self.rt.grad_transforms:
            gt(self.rt.trainable_params, ctx)

    def optimizer_step(self) -> bool:
        """Step optimizers; step schedulers ONLY if the step really happened (fp16 overflow skips
        it, K2). Always zero grads. Returns whether the step happened."""
        stepped = self.rt.engine.step_optimizers(self.rt.optimizers)
        if stepped:
            for sch in self.rt.schedulers:
                sch.step()
        for opt in self.rt.optimizers.values():
            opt.zero_grad(set_to_none=True)
        return stepped

    def nonfinite(self, metrics: dict) -> bool:
        is_bad = False
        for k, v in metrics.items():
            if (k == "loss/total" or k.startswith("loss/")) and (
                (isinstance(v, torch.Tensor) and not bool(torch.isfinite(v).all()))
                or (isinstance(v, (float, int)) and not math.isfinite(v))
            ):
                is_bad = True
                break
        eng = getattr(self.rt, "engine", None)
        if eng is not None and getattr(eng, "world_size", 1) > 1:
            flag = torch.tensor(1.0 if is_bad else 0.0, device=eng.device)
            flag = eng.all_reduce_sum(flag)
            return float(flag) > 0.0
        return is_bad

    # ------------------------------------------------------------- main entry
    def run(self, batch: Batch, rt: Any) -> StepOutput:
        raise NotImplementedError


@STEP_STRATEGIES.register("standard")
class StandardStep(StepStrategy):
    """Standard step: accumulate over micro-batches with global loss normalization, clip, optimizer + scheduler
    step.
    """
    class Config(BaseConfig):
        skip_nonfinite: bool = True

    def run(self, batch: Batch, rt: Any) -> StepOutput:
        micros = self.micro_batches(batch)
        norms = self.global_norms(micros)
        metrics = self.accumulate(micros, norms)
        if self.cfg.skip_nonfinite and self.nonfinite(metrics):
            for opt in rt.optimizers.values():
                opt.zero_grad(set_to_none=True)
            return StepOutput(metrics, int(norms.z_tok), int(norms.z_seq), skipped=True)
        self.apply_grad_transforms(metrics)
        stepped = self.optimizer_step()
        if "grad/norm" not in metrics:
            gn = rt.engine.get_global_grad_norm()
            if gn is not None:
                metrics["grad/norm"] = torch.tensor(float(gn))
        for lr_i, opt in enumerate(rt.optimizers.values()):
            for g in opt.param_groups:
                metrics[f"lr/{g.get('group_name', lr_i)}"] = torch.tensor(g["lr"])
        return StepOutput(metrics, int(norms.z_tok), int(norms.z_seq), skipped=not stepped)


@STEP_STRATEGIES.register("sam", paper="SAM (Foret et al. 2021)")
class SAMStep(StepStrategy):
    """Two-pass sharpness-aware step: perturb by rho*g/||g||, then update at w+e.

    Source: Foret et al. (2021), SAM; TALAS-M train_asam.py:88-134. Example recipe:
    ``step: {type: asam, rho: 0.05, clip_before_perturb: true}``.
    Works with the local engine and bf16; DeepSpeed and fp16 GradScaler are unsupported.
    Unlike TALAS, configured ClipNorm transforms can clip any parameter group, not just student.
    Each optimizer step processes every micro-batch twice, so state.micro_step advances twice.
    """

    class Config(BaseConfig):
        rho: float = 0.05
        adaptive: bool = False
        eps: float = 1e-12
        clip_before_perturb: bool = True
        keep_dropout_rng: bool = False
        perturb_groups: list[str] | None = None
        skip_nonfinite: bool = True

    def setup(self, rt: Any) -> None:
        if getattr(rt.engine.cfg, "backend", None) == "deepspeed":
            raise NotImplementedError("SAM/ASAM does not support DeepSpeed; use engine local")
        if rt.engine.scaler is not None or getattr(rt.engine.cfg, "mixed_precision", "no") == "fp16":
            raise NotImplementedError("SAM/ASAM does not support fp16 GradScaler; use bf16")
        if not isinstance(rt.engine, LocalEngine):
            raise NotImplementedError("SAM/ASAM currently only supports the local engine")
        super().setup(rt)

    @staticmethod
    def _get_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return cpu, cuda

    @staticmethod
    def _set_rng_state(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
        cpu, cuda = state
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)

    def run(self, batch: Batch, rt: Any) -> StepOutput:
        micros = self.micro_batches(batch)
        norms = self.global_norms(micros)
        rng = self._get_rng_state() if self.cfg.keep_dropout_rng else None
        phase1 = self.accumulate(micros, norms)
        if self.cfg.skip_nonfinite and self.nonfinite(phase1):
            for opt in rt.optimizers.values():
                opt.zero_grad(set_to_none=True)
            return StepOutput(phase1, int(norms.z_tok), int(norms.z_seq), skipped=True)

        # GradScaler is rejected in setup; unscale is therefore a no-op for local bf16/fp32.
        rt.engine.unscale(rt.optimizers)
        if self.cfg.clip_before_perturb:
            ctx = StepContext(rt=rt, metrics=phase1)
            for transform in rt.grad_transforms:
                if isinstance(transform, ClipNorm):
                    transform(rt.trainable_params, ctx)

        selected = [p for p in rt.trainable_params
                    if p.param.grad is not None
                    and (self.cfg.perturb_groups is None or p.group in self.cfg.perturb_groups)]
        lengths = [((p.param.abs() if self.cfg.adaptive else 1.0) * p.param.grad).norm(2)
                   for p in selected]
        grad_norm = torch.norm(torch.stack(lengths), 2) if lengths else torch.zeros((), device=rt.engine.device)
        scale = self.cfg.rho / (grad_norm + self.cfg.eps)
        changes: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            with torch.no_grad():
                for wrapped in selected:
                    param = wrapped.param
                    delta = (param.pow(2) if self.cfg.adaptive else 1.0) * param.grad * scale
                    param.add_(delta)
                    changes.append((param, delta))
            for opt in rt.optimizers.values():
                opt.zero_grad(set_to_none=True)
            if rng is not None:
                self._set_rng_state(rng)
            phase2 = self.accumulate(micros, norms)
        except BaseException:
            for opt in rt.optimizers.values():
                opt.zero_grad(set_to_none=True)
            raise
        finally:
            with torch.no_grad():
                for param, delta in changes:
                    param.sub_(delta)

        phase2["sam/loss_phase1"] = phase1["loss/total"]
        phase2["sam/grad_norm"] = grad_norm.detach()
        if self.cfg.skip_nonfinite and self.nonfinite(phase2):
            for opt in rt.optimizers.values():
                opt.zero_grad(set_to_none=True)
            return StepOutput(phase2, int(norms.z_tok), int(norms.z_seq), skipped=True)
        self.apply_grad_transforms(phase2)
        stepped = self.optimizer_step()
        for lr_i, opt in enumerate(rt.optimizers.values()):
            for group in opt.param_groups:
                phase2[f"lr/{group.get('group_name', lr_i)}"] = torch.tensor(group["lr"])
        return StepOutput(phase2, int(norms.z_tok), int(norms.z_seq), skipped=not stepped)


@STEP_STRATEGIES.register("asam",
                          source="embedding-distill/TALAS-M-main/train_asam.py:88-134")
class ASAMStep(SAMStep):
    """Adaptive SAM: use |w|g for the norm and w²g for the perturbation.

    Source: TALAS-M train_asam.py:88-134; Kwon et al. (2021), ASAM.
    """

    class Config(SAMStep.Config):
        adaptive: bool = True


@STEP_STRATEGIES.register("multi_objective_grad", source="IMPACT/code/alm_multitask.py:95-116")
class MultiObjectiveGradStep(StandardStep):
    """Weight each task by its local last-block gradient norm, then run the ordinary backward.

    Source: IMPACT/code/alm_multitask.py:68-116. Weights are recomputed for every micro-batch
    and detached, so the final gradient is sum_i w_i * grad(L_i). Extra autograd.grad calls
    do not write .grad or replace engine.backward. DeepSpeed/ZeRO is unsupported because its
    sharded gradients do not satisfy this local autograd.grad contract. Each micro-batch adds
    one partial backward per task; ordinary accumulation and optimizer stepping are reused.

    With combine=pcgrad, accumulate weighted/global-normalized task gradients across all micros,
    then project once over all trainable parameters before clipping. The local engine owns scaler
    handling and the ordinary backward/optimizer step; grad_params only selects GradMag probes.
    """
    class Config(StandardStep.Config):
        combine: str = "gradmag"     # gradmag | pcgrad | sum
        grad_params: str = "last_block"   # which params per-term grads are computed on

    def setup(self, rt):
        from lmkd.losses.aggregator import GradMag
        from lmkd.train.multitask import last_block_params

        if not isinstance(rt.engine, LocalEngine):
            raise NotImplementedError("multi_objective_grad supports only the local engine, not DeepSpeed")
        if self.cfg.combine not in ("gradmag", "pcgrad", "sum"):
            raise ValueError(f"unknown multi-objective combine {self.cfg.combine!r}")
        if self.cfg.combine == "gradmag" and not isinstance(rt.aggregator, GradMag):
            raise ValueError("multi_objective_grad combine=gradmag needs aggregator: gradmag")
        if self.cfg.combine == "sum" and isinstance(rt.aggregator, GradMag):
            raise ValueError("combine=sum needs a non-GradMag aggregator")
        if self.cfg.combine == "pcgrad" and isinstance(rt.aggregator, GradMag):
            raise ValueError("combine=pcgrad needs a non-GradMag aggregator")
        if self.cfg.grad_params not in ("last_block", "all_trainable"):
            raise ValueError(f"unknown grad_params {self.cfg.grad_params!r}")
        super().setup(rt)
        self.params = (last_block_params(rt) if self.cfg.grad_params == "last_block" and self.cfg.combine != "pcgrad"
                       else [p.param for p in rt.trainable_params])

    def accumulate(self, micros: list[Batch], norms: Norms) -> dict[str, torch.Tensor]:
        from lmkd.train.multitask import gradmag_weights, task_grads

        if self.cfg.combine == "sum":
            return super().accumulate(micros, norms)
        if self.cfg.combine == "pcgrad":
            return self._pcgrad_accumulate(micros, norms)
        rt, eng = self.rt, self.rt.engine
        metrics: dict[str, torch.Tensor] = {}
        for i, micro in enumerate(micros):
            if micro is not None and hasattr(micro, "to") and getattr(eng, "device", None) is not None:
                micro = micro.to(eng.device)
            with eng.no_sync(rt.student_module, enabled=i != len(micros) - 1), eng.autocast():
                outputs = rt.algorithm.loss(micro, rt)
                tasks = rt.aggregator.task_losses(outputs, norms)
                weights, magnitudes = gradmag_weights(
                    task_grads([loss for _, loss in tasks], self.params),
                    rt.aggregator.cfg.mode, rt.aggregator.cfg.epsilon)
                rt.aggregator.set_weights(weights, magnitudes)
                loss, current = rt.aggregator.combine(outputs, norms)
            eng.backward(loss * eng.world_size)
            for key, value in current.items():
                metrics[key] = metrics.get(key, 0.0) + value.float()
            rt.state.micro_step += 1
        self._normalize_micro_metrics(metrics, len(micros))
        return metrics

    def _pcgrad_accumulate(self, micros: list[Batch], norms: Norms) -> dict[str, torch.Tensor]:
        from lmkd.train.multitask import task_grads

        rt, eng = self.rt, self.rt.engine
        self.term_grads: dict[str, list[torch.Tensor]] = {}
        metrics: dict[str, torch.Tensor] = {}
        for i, micro in enumerate(micros):
            if micro is not None and hasattr(micro, "to") and getattr(eng, "device", None) is not None:
                micro = micro.to(eng.device)
            with eng.no_sync(rt.student_module, enabled=i != len(micros) - 1), eng.autocast():
                outputs = rt.algorithm.loss(micro, rt)
                loss, current = rt.aggregator.combine(outputs, norms)
                # Each aggregator contribution includes configured weights and global normalization.
                contributions = [rt.aggregator.combine([out], norms)[0] for out in outputs]
                gradients = task_grads(contributions, self.params)
                for out, grads in zip(outputs, gradients):
                    if out.alias not in self.term_grads:
                        self.term_grads[out.alias] = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
                    for total, grad in zip(self.term_grads[out.alias], grads):
                        total.add_(grad.detach().float())
            eng.backward(loss * eng.world_size)
            for key, value in current.items():
                metrics[key] = metrics.get(key, 0.) + value.float()
            rt.state.micro_step += 1
        self._normalize_micro_metrics(metrics, len(micros))
        metrics["pcgrad/tasks"] = torch.tensor(float(len(self.term_grads)))
        return metrics

    def apply_grad_transforms(self, metrics: dict) -> None:
        if self.cfg.combine != "pcgrad":
            super().apply_grad_transforms(metrics)
            return
        from lmkd.train.grad import PCGrad, pcgrad_combine

        rt = self.rt
        rt.engine.unscale(rt.optimizers)
        ctx = StepContext(rt=rt, metrics=metrics, term_grads=self.term_grads)
        unused = [p for p in self.params if p.grad is None]
        if not any(isinstance(transform, PCGrad) for transform in rt.grad_transforms):
            combined = pcgrad_combine(list(self.term_grads.values()))
            for p, grad in zip(self.params, combined):
                if p.grad is not None:
                    p.grad.copy_(grad.to(p.grad))
        for transform in rt.grad_transforms:
            transform(rt.trainable_params, ctx)
        for p in unused:
            p.grad = None  # Do not apply weight decay to parameters unused by all tasks.
        self.term_grads.clear()


@STEP_STRATEGIES.register("ppo_epochs", source="Self-Distill/RLSD-main/verl/workers/actor/dp_opsd_actor.py:426-579")
class PPOEpochsStep(StepStrategy):
    """PPO epochs with token-mean normalization across all ranks/micros of each mini-batch.

    Scheduler steps count trainer steps (rollout batches), once after successful mini-batch updates.
    Unlike verl's micro-batch mean, gradients use the shared global token denominator.
    """
    class Config(BaseConfig):
        epochs: int = Field(default=1, ge=1)
        mini_batches: int = Field(default=1, ge=1)
        shuffle: bool = True

    def micro_batches(self, batch: Batch) -> list[Batch]:
        # Group-dependent signals are already computed in augment; criteria may use any row order.
        size = self.rt.config.train.micro_batch_size
        return [batch.select(list(range(i, min(i + size, batch.batch_size)))).to(self.rt.engine.device)
                for i in range(0, batch.batch_size, size)]

    def run(self, batch: Batch, rt: Any) -> StepOutput:
        count = batch.batch_size
        if count % self.cfg.mini_batches:
            raise ValueError("rollout batch size must be divisible by mini_batches")
        size = count // self.cfg.mini_batches
        totals: dict[str, torch.Tensor] = {}
        stepped = False
        norms = self.global_norms(self.micro_batches(batch))
        for epoch in range(self.cfg.epochs):
            generator = torch.Generator().manual_seed(rt.config.train.seed + rt.state.step * self.cfg.epochs + epoch)
            indices = torch.randperm(count, generator=generator).tolist() if self.cfg.shuffle else list(range(count))
            for start in range(0, count, size):
                mini = batch.select(indices[start:start + size])
                metrics, success = self._mini_step(mini, rt)
                stepped |= success
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.) + value
        if stepped:
            for scheduler in rt.schedulers:
                scheduler.step()
        divisor = self.cfg.epochs * self.cfg.mini_batches
        metrics = {key: value / divisor for key, value in totals.items()}
        return StepOutput(metrics, int(norms.z_tok), int(norms.z_seq), skipped=not stepped)

    def _mini_step(self, batch: Batch, rt: Any) -> tuple[dict[str, torch.Tensor], bool]:
        micros = self.micro_batches(batch)
        metrics = self.accumulate(micros, self.global_norms(micros))
        if self.nonfinite(metrics):
            success = False
        else:
            self.apply_grad_transforms(metrics)
            success = rt.engine.step_optimizers(rt.optimizers)
        for optimizer in rt.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        return metrics, success
