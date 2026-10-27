"""Callbacks (DESIGN §7.5). Events are methods; implement only those you need."""
from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import CALLBACKS

if TYPE_CHECKING:
    from lmkd.train.runtime import Runtime


class Callback(Plugin):
    """Training event hooks (begin/step/eval/save/end); override only what you need; lower ``order`` runs first."""
    order: int = 50   # lower runs first

    def on_train_begin(self, rt: Any) -> None: ...
    def on_step_begin(self, rt: Any) -> None: ...
    def on_step_end(self, rt: Any, out: Any) -> None: ...
    def on_eval(self, rt: Any, metrics: dict) -> None: ...
    def on_save(self, rt: Any, path: Path) -> None: ...
    def on_train_end(self, rt: Any) -> None: ...


class CallbackList:
    """Dispatches every event to all callbacks in ``order``."""
    def __init__(self, cbs: list[Callback]):
        self.cbs = sorted(cbs, key=lambda c: c.order)

    def __getattr__(self, event: str) -> Callable[..., None]:
        def call(*args: Any, **kwargs: Any) -> None:
            for c in self.cbs:
                getattr(c, event)(*args, **kwargs)
        return call


def _scalar(v: Any) -> float:
    return float(v.item()) if isinstance(v, torch.Tensor) else float(v)


@CALLBACKS.register("console_logger")
class ConsoleLogger(Callback):
    """Prints and appends metrics to ``<output_dir>/metrics.jsonl`` every ``train.log_every`` steps.

    Sources: ``out.metrics`` (step strategy) + ``rt.state.pop_metrics()`` (anything logged with
    ``state.log(key, value)`` by producers, hooks, callbacks) (K4). Averaged over the window, then
    averaged across ranks with ONE all-reduce over the UNION of the ranks' key sets (gathered first,
    so ranks logging different keys cannot desync the collective; keys missing on a rank count as 0)."""

    order = 90

    class Config(BaseConfig):
        jsonl: bool = True

    def on_train_begin(self, rt):
        self._acc: dict[str, float] = {}
        self._n = 0
        self._t = time.time()
        self.path = Path(rt.config.train.output_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _reduce(self, rt: Runtime, avg: dict[str, float]) -> dict[str, float]:
        eng = rt.engine
        if eng.world_size == 1:
            return avg
        keys = sorted(set().union(*eng.all_gather_object(sorted(avg))))
        if not keys:
            return avg
        t = torch.tensor([avg.get(k, 0.0) for k in keys], device=eng.device, dtype=torch.float64)
        t = eng.all_reduce_sum(t)
        ws = float(eng.world_size)
        out = {}
        for k, v in zip(keys, t.tolist()):
            # Loss metrics are already globally normalized (divided by global Norms Z across all ranks).
            # Summing across ranks gives the true global loss; dividing by world_size would divide twice.
            is_loss = k.startswith("loss/") or k == "loss_fn" or k == "sam/loss_phase1"
            out[k] = v if is_loss else v / ws
        return out

    def on_step_end(self, rt, out):
        merged = dict(out.metrics)
        merged.update(rt.state.pop_metrics())
        for k, v in merged.items():
            if isinstance(v, torch.Tensor):
                self._acc[k] = self._acc.get(k, 0.0) + float(v.detach().item())
            else:
                self._acc[k] = self._acc.get(k, 0.0) + float(v)
        self._n += 1
        if rt.state.step % rt.config.train.log_every == 0 or rt.state.done:
            float_acc = {k: _scalar(v) / max(1, self._n) for k, v in self._acc.items()}
            avg = self._reduce(rt, float_acc)
            dt = time.time() - self._t
            rec = {"step": rt.state.step, "epoch": round(rt.state.epoch, 4), "sec": round(dt, 2), **avg}
            if rt.engine.is_main:
                msg = " ".join(f"{k}={v:.4g}" for k, v in rec.items() if isinstance(v, float))
                print(f"[step {rt.state.step}/{rt.state.total_steps}] {msg}", flush=True)
                if self.cfg.jsonl:
                    with open(self.path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
            self._acc, self._n, self._t = {}, 0, time.time()


@CALLBACKS.register("nan_guard")
class NaNGuard(Callback):
    """Stops training after ``patience`` consecutive non-finite steps."""

    order = 10

    class Config(BaseConfig):
        patience: int = 5

    def on_train_begin(self, rt):
        self._bad = 0

    def on_step_end(self, rt, out):
        if out.skipped:
            self._bad += 1
            if self._bad >= self.cfg.patience:
                raise FloatingPointError(f"{self._bad} consecutive non-finite steps at step {rt.state.step}")
        else:
            self._bad = 0


@CALLBACKS.register("evaluate")
class EvaluateCallback(Callback):
    """Periodic evaluation callback (order 70, runs before checkpoint 80).

    Runs all rt.evaluators, updates rt.state.extra["eval"], and writes metrics to
    <output_dir>/eval.jsonl on the main rank. Add `callbacks: [evaluate]` to your recipe.
    """

    order = 70

    class Config(BaseConfig):
        every: int | str = "epoch"   # int steps | "epoch" | "end"
        at_start: bool = False       # evaluate before training begins
        at_end: bool = True

    def __init__(self, cfg: Any = None) -> None:
        super().__init__(cfg)
        self._last_eval_step: int | None = None

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        evaluators = getattr(rt, "evaluators", [])
        if not evaluators:
            import logging
            logging.getLogger(__name__).warning("evaluate callback registered but rt.evaluators is empty.")

    def _should_eval_step(self, rt: Any) -> bool:
        if self.cfg.every == "end":
            return False
        if isinstance(self.cfg.every, int):
            return self.cfg.every > 0 and rt.state.step % self.cfg.every == 0
        if self.cfg.every == "epoch":
            spe = getattr(rt.producer, "steps_per_epoch", lambda: None)()
            return bool(spe and rt.state.step % spe == 0)
        return False

    def _run_eval(self, rt: Any) -> None:
        metrics = rt.evaluate()
        self._last_eval_step = getattr(rt.state, "step", 0)
        if rt.engine.is_main:
            out_dir = Path(getattr(rt.config.train, "output_dir", "."))
            out_file = out_dir / "eval.jsonl"
            record = {"step": getattr(rt.state, "step", 0), **metrics}
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def on_train_begin(self, rt: Any) -> None:
        if self.cfg.at_start:
            self._run_eval(rt)

    def on_step_end(self, rt: Any, out: Any) -> None:
        if self._should_eval_step(rt):
            self._run_eval(rt)

    def on_train_end(self, rt: Any) -> None:
        if self.cfg.at_end and self._last_eval_step != getattr(rt.state, "step", 0):
            self._run_eval(rt)


@CALLBACKS.register("checkpoint")
class Checkpoint(Callback):
    """Full checkpoint (``train/checkpointing.py``: weights, aux modules, optimizer, schedulers,
    producer, RNG per rank, COMPLETE marker) every ``every`` steps and/or at the end; resume with
    ``train.resume_from``. Rotation of ``checkpoint-<step>`` dirs (never ``final`` / other tags):
    keep the ``keep_last`` newest plus the ``keep_best`` best by ``rt.state.extra["eval"][metric]``."""

    order = 80

    class Config(BaseConfig):
        every: int | None = None      # steps; None = only at the end
        at_end: bool = True
        keep_last: int | None = 3     # None = keep every step checkpoint
        keep_best: int = 0            # >0: also keep the N best step checkpoints by `metric`
        metric: str = "eval/loss"     # read from rt.state.extra["eval"][metric] (written by T07)
        mode: Literal["min", "max"] = "min"

    def save(self, rt: Runtime, tag: str) -> Path:
        from lmkd.train.checkpointing import save_checkpoint

        path = save_checkpoint(rt, tag)   # collective: every rank
        rt.callbacks.on_save(rt, path)
        if rt.engine.is_main:
            self.rotate(Path(rt.config.train.output_dir))
        rt.engine.barrier()
        return path

    def rotate(self, root: Path) -> None:
        """Delete step checkpoints neither among the newest ``keep_last`` nor the best ``keep_best``.
        Stateless: steps and metrics are read back from each checkpoint's train_state.json."""
        from lmkd.train.checkpointing import COMPLETE

        if self.cfg.keep_last is None:
            return
        ckpts = []   # (step, metric | None, path)
        for p in root.glob("checkpoint-*"):
            if not (p.is_dir() and p.name[len("checkpoint-"):].isdigit() and (p / COMPLETE).exists()):
                continue
            st = json.loads((p / "train_state.json").read_text())
            val = (st.get("extra") or {}).get("eval", {}).get(self.cfg.metric)
            ckpts.append((int(st["step"]), val, p))
        ckpts.sort(key=lambda c: c[0])
        keep = {c[2] for c in ckpts[max(0, len(ckpts) - self.cfg.keep_last):]} if self.cfg.keep_last > 0 else set()
        if self.cfg.keep_best > 0:
            scored = [c for c in ckpts if isinstance(c[1], (int, float)) and math.isfinite(c[1])]
            scored.sort(key=lambda c: c[1], reverse=self.cfg.mode == "max")
            keep |= {c[2] for c in scored[: self.cfg.keep_best]}
        for _, _, p in ckpts:
            if p not in keep:
                shutil.rmtree(p, ignore_errors=True)

    def on_step_end(self, rt, out):
        if self.cfg.every and rt.state.step % self.cfg.every == 0:
            self.save(rt, str(rt.state.step))

    def on_train_end(self, rt):
        if self.cfg.at_end:
            self.save(rt, "final")


@CALLBACKS.register("param_schedule")
class ParamSchedule(Callback):
    """Applies ``loss.schedules`` (and its own ``schedules``) at the START of every step.

    Criteria must read their configuration fields dynamically during ``forward`` (not cached in __init__)
    to be schedulable. Criteria may declare ``schedulable_fields: ClassVar[tuple[str,...] | None] = None``
    to whitelist fields that can be safely scheduled.
    """

    order = 20

    class Config(BaseConfig):
        schedules: dict[str, dict] = {}  # merged with rt.config.loss.schedules (own entries win)

    def setup(self, rt: Any) -> None:
        merged = dict(getattr(rt.config.loss, "schedules", {}))
        merged.update(self.cfg.schedules)
        self._schedules = merged
        self._alias_to_crit = {e.alias: e.criterion for e in rt.criteria}

        for key in self._schedules:
            target = key[len("loss."):] if key.startswith("loss.") else key
            if target.count(".") != 1:
                raise ValueError(f"unsupported target '{key}': expected '<alias>.<field>' or '<alias>.weight'")
            alias, prop = target.split(".", 1)
            if alias not in self._alias_to_crit:
                raise ValueError(f"unknown loss alias '{alias}' in schedule target '{key}'")
            crit = self._alias_to_crit[alias]
            if prop == "weight":
                if alias not in rt.loss_weights:
                    raise ValueError(f"unknown loss alias '{alias}' in schedule target '{key}'")
            else:
                model_fields = type(crit.cfg).model_fields
                if prop not in model_fields:
                    raise ValueError(f"field '{prop}' not in {type(crit.cfg).__name__} for alias '{alias}'")
                cur_val = getattr(crit.cfg, prop)
                if not isinstance(cur_val, (int, float)) and model_fields[prop].annotation not in (
                    int, float, int | float, float | int, float | None, int | None
                ):
                    raise ValueError(f"field '{prop}' in {type(crit.cfg).__name__} must be float or int")
                schedulable = getattr(crit, "schedulable_fields", None)
                if schedulable is not None and prop not in schedulable:
                    raise ValueError(f"criterion '{alias}' caches field '{prop}'; not schedulable")

    def on_step_begin(self, rt: Any) -> None:
        from lmkd.train.schedules import schedule_value

        step = rt.state.step
        total = rt.state.total_steps
        for key, spec in self._schedules.items():
            val = schedule_value(spec, step, total, rt)
            target = key[len("loss."):] if key.startswith("loss.") else key
            alias, prop = target.split(".", 1)
            if prop == "weight":
                rt.loss_weights[alias] = float(val)
            else:
                crit = self._alias_to_crit[alias]
                field_info = type(crit.cfg).model_fields.get(prop)
                if field_info and field_info.annotation is int:
                    val = int(round(val))
                else:
                    val = float(val)
                setattr(crit.cfg, prop, val)
            rt.state.log(f"sched/{key}", val)


@CALLBACKS.register("rollout_weight_sync")
class RolloutWeightSync(Callback):
    """Mark the rollout engine's weights stale every ``every`` optimizer steps (sync itself is lazy:
    the engine pushes weights only when ``generate`` is called)."""

    order = 60

    class Config(BaseConfig):
        every: int = 1

    def setup(self, rt):
        if rt.rollout is not None and hasattr(rt.rollout, "auto_stale"):
            rt.rollout.auto_stale = False       # `every` decides when weights become stale

    def on_step_end(self, rt, out):
        if rt.rollout is not None and rt.state.step % self.cfg.every == 0:
            mark = getattr(rt.rollout, "mark_stale", None)
            if mark:
                mark()



