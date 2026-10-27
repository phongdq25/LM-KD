"""Algorithm (DESIGN §7.3): ``augment`` a step batch, then compute weighted criterion outputs per
micro-batch. The StepStrategy owns backward/optimizer; the aggregator owns normalization.
"""
from __future__ import annotations

from typing import Any

import torch
from pydantic import Field

from lmkd.core.batch import FIELDS, Batch, FieldSpec
from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import ADV_TRANSFORMS, ADVANTAGES, ALGORITHMS
from lmkd.data.labels import supervised_mask
from lmkd.losses.aggregator import WeightedOutput
from lmkd.losses.base import LossContext
from lmkd.losses.functional.rl import masked_mean
from lmkd.losses.simple import as_loss_output
from lmkd.models.forward import FeatureRequest, ForwardRunner, SideRequest, side_inputs
from lmkd.teachers.signals import TeacherSignalSpec
from lmkd.train.grad import HookContext, attach_grad_hooks

for _key in ("advantage", "old_logp", "ref_logp"):
    FIELDS.register(FieldSpec(f"rollout.{_key}", "token", torch.float32, 0., "student"))
FIELDS.register(FieldSpec("rollout.is_weight", "token", torch.float32, 1., "student"))


class Algorithm(Plugin):
    """Prepares a step's batch (teacher inputs, RL signals) and computes the weighted criterion outputs of each
    micro-batch.
    """
    def setup(self, rt: Any) -> None:
        self.rt = rt
        self._req_cache: dict[frozenset, FeatureRequest] = {}
        _ = self.request  # validate teacher capabilities early

    @property
    def request(self) -> FeatureRequest:
        """Union of ``features()`` of criteria whose CURRENT weight is non-zero (+ grad hooks).
        Recomputed when the active set changes, so a weight scheduled from 0 to >0 later gets its
        features (K6). Cached per active set."""
        rt = self.rt
        active = frozenset(t.alias for t in rt.criteria if rt.loss_weights[t.alias] != 0.0)
        req = self._req_cache.get(active)
        if req is None:
            req = FeatureRequest()
            for t in rt.criteria:
                if t.alias in active:
                    req = req.merge(t.criterion.features())
            for h in rt.grad_hooks:
                req = req.merge(h.features())
            if req.teacher is not None or req.teacher_signal is not None:
                rt.teacher.check(req.teacher_signal, req.teacher)
            self._req_cache[active] = req
        return req

    def augment(self, batch: Batch, rt: Any) -> Batch:
        return rt.teacher_input.build(batch, rt)

    def loss(self, micro: Batch, rt: Any) -> list[WeightedOutput]:
        raise NotImplementedError


@ALGORITHMS.register("standard")
class Standard(Algorithm):
    """Off-policy / on-policy distillation: student forward (grad) + teacher signals (no grad) +
    every criterion with non-zero weight."""

    class Config(BaseConfig):
        pass

    def loss(self, micro: Batch, rt: Any) -> list[WeightedOutput]:
        outs = []
        if any(rt.loss_weights[t.alias] != 0.0 for t in rt.criteria):
            outs = self.criteria_loss(micro, rt)
        if rt.loss_fn is not None:   # tier 0 (compute_loss / rt.train_step(loss_fn=...))
            outs.append(WeightedOutput("loss_fn", 1.0, as_loss_output(rt.loss_fn(micro, rt))))
        if not outs:
            raise ValueError("no active loss: every loss.terms weight is 0 and no loss function is set")
        return outs

    def criteria_loss(self, micro: Batch, rt: Any) -> list[WeightedOutput]:
        """Outputs of every criterion with non-zero weight (one shared forward)."""
        req = self.request
        views = ForwardRunner.run(rt.student_model, rt.student_adapter, side_inputs(micro, "student"),
                                  req.student, grad=True, module=rt.student_module)
        if not views:
            raise ValueError("no criterion requested any student feature")
        if rt.grad_hooks:
            attach_grad_hooks(rt.grad_hooks, views[0], HookContext(rt=rt, batch=micro, state=rt.state))
        teacher_out = None
        if req.teacher is not None or req.teacher_signal is not None:
            teacher_out = rt.teacher.signals(micro, req.teacher_signal, req.teacher, views[0])
        ctx = LossContext(batch=micro, student=views[0], student_views=views, teacher=teacher_out,
                          modules=rt.modules, state=rt.state, rt=rt)
        outs = []
        for t in rt.criteria:
            w = rt.loss_weights[t.alias]
            if w == 0.0:
                continue
            outs.append(WeightedOutput(t.alias, w, t.criterion(ctx)))
        return outs


@ALGORITHMS.register("compute_loss")
class ComputeLoss(Standard):
    """Tier 0: a user function ``fn(batch, rt, **kwargs)`` computes the loss and may call
    ``rt.forward`` / ``rt.teacher_forward`` itself (port original code nearly verbatim).
    ``fn`` = name registered with ``@lmkd.compute_loss`` (file listed in ``imports:``);
    ``kwargs`` = its hyperparameters, e.g. ``{type: compute_loss, fn: my_kd, kwargs: {temperature: 2.0}}``.
    Criteria in ``loss.terms`` (if any) are added on top, with their weights."""

    class Config(BaseConfig):
        fn: str
        kwargs: dict = {}            # hyperparameters passed to fn as keyword arguments

    def setup(self, rt: Any) -> None:
        from lmkd.core.registry import LOSS_FNS

        fn = LOSS_FNS.build(self.cfg.fn)
        kw = dict(self.cfg.kwargs)
        rt.loss_fn = (lambda batch, rt_: fn(batch, rt_, **kw)) if kw else fn
        super().setup(rt)


@ALGORITHMS.register("rl", source="Self-Distill/RLSD-main/verl/trainer/opsd_trainer.py:1911-1980")
class RL(Standard):
    """Freeze old policy log-probabilities before updates, then compute group and teacher-guided advantages."""
    requires = ("student.labels", "rollout.reward", "group.uid")
    provides = ("rollout.old_logp", "rollout.advantage", "rollout.is_weight")

    class Config(BaseConfig):
        advantage: dict | str = "grpo"
        adv_transforms: list[dict] = []
        recompute_old_logp: bool = True
        rollout_is: str = "none"
        rollout_is_threshold: float = Field(default=2., gt=0)
        ppo_epochs: int = Field(default=1, ge=1)  # legacy stub fields; configure optimization via step
        mini_batches: int = Field(default=1, ge=1)

    @classmethod
    def provided_keys(cls, cfg):
        # IS is optional when the rollout engine supplies no log-probabilities.
        keys = ("rollout.advantage",)
        return keys + (("rollout.old_logp",) if cfg.recompute_old_logp else ())

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        self.estimator = ADVANTAGES.build(self.cfg.advantage)
        self.transforms = [ADV_TRANSFORMS.build(spec) for spec in self.cfg.adv_transforms]
        for legacy, current in (("ppo_epochs", "epochs"), ("mini_batches", "mini_batches")):
            if legacy in self.cfg.model_fields_set and getattr(self.cfg, legacy) != getattr(rt.step_strategy.cfg, current, 1):
                raise ValueError(f"algorithm.{legacy} must match step.{current}; configure PPO updates in step")
        if self.cfg.rollout_is not in ("none", "token"):
            raise ValueError("rollout_is must be none or token")
        if not self.cfg.recompute_old_logp:
            cfg = rt.step_strategy.cfg
            if self.transforms or self.cfg.rollout_is != "none" or getattr(cfg, "epochs", 1) != 1 or getattr(cfg, "mini_batches", 1) != 1:
                raise ValueError("recompute_old_logp=False requires a single on-policy update without reweighting")
        for plugin in [self.estimator, *self.transforms]:
            if hasattr(plugin, "setup"):
                plugin.setup(rt)
            if getattr(plugin, "needs_teacher_logp", False):
                rt.teacher.check(TeacherSignalSpec(kind="sampled_logp"), SideRequest(logits="view"))

    @torch.no_grad()
    def _logp_chunks(self, batch: Batch, rt: Any, teacher=False) -> torch.Tensor:
        size = rt.config.train.micro_batch_size
        chunks = []
        was = rt.student_model.training
        rt.student_model.eval()
        try:
            for start in range(0, batch.batch_size, size):
                chunk = batch.select(list(range(start, min(start + size, batch.batch_size)))).to(rt.engine.device)
                with rt.engine.autocast():
                    if teacher:
                        output = rt.teacher.signals(chunk, TeacherSignalSpec(kind="sampled_logp"), SideRequest(logits="view"), None)
                        logp = output.token_logp
                    else:
                        view = ForwardRunner.run(rt.student_model, rt.student_adapter, side_inputs(chunk, "student"),
                                                 SideRequest(logits="view"), grad=False, module=rt.student_module)[0]
                        if view.logits is None:
                            raise ValueError("RL requires student logits")
                        logp = view.logits.target_logp(chunk["student.labels"])
                if logp is None:
                    raise ValueError("RL requires target log-probabilities")
                chunks.append(logp.to(batch["student.labels"].device))
        finally:
            rt.student_model.train(was)
        return torch.cat(chunks)

    def augment(self, batch: Batch, rt: Any) -> Batch:
        batch = rt.teacher_input.build(batch.copy(), rt)
        old = self._logp_chunks(batch, rt) if self.cfg.recompute_old_logp else None
        if old is not None:
            batch["rollout.old_logp"] = old.detach()
        mask = supervised_mask(batch["student.labels"])
        advantage = self.estimator(batch).detach()[:, None] * mask
        teacher_logp = None
        for transform in self.transforms:
            if getattr(transform, "needs_teacher_logp", False) and teacher_logp is None:
                teacher_logp = self._logp_chunks(batch, rt, teacher=True)
            advantage = transform(advantage, batch, teacher_logp, old, rt.state)
        batch["rollout.advantage"] = advantage.detach()
        if self.cfg.rollout_is == "token" and "rollout.rollout_logp" in batch:
            batch["rollout.is_weight"] = (old - batch["rollout.rollout_logp"]).clamp(-20, 20).exp().clamp(max=self.cfg.rollout_is_threshold) * mask
        rt.state.log("rl/adv_abs_mean", masked_mean(advantage.abs(), mask))
        if old is not None:
            rt.state.log("rl/old_logp_mean", masked_mean(old, mask))
        return batch
