"""Build a Runtime from a RecipeConfig. Order matters; keep it linear and explicit."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from lmkd.core import registry as R
from lmkd.core.config import RecipeConfig
from lmkd.data.collator import Collator
from lmkd.data.context import DataContext
from lmkd.data.dataset import PreparedDataset
from lmkd.data.transforms.base import run_transforms
from lmkd.losses.base import ModelPairInfo, PrepareContext
from lmkd.models.adapters import HiddenForward, resolve_adapter
from lmkd.models.loading import load_model, load_tokenizer
from lmkd.models.modules import ModuleRegistry
from lmkd.train.callbacks import CallbackList
from lmkd.train.grad import ClipNorm, ParamWithGroup
from lmkd.train.optim import NamedParam, build_param_groups
from lmkd.train.runtime import CriterionEntry, Runtime


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_data(rt: Runtime, train_records: list[dict] | None = None) -> None:
    """``train_records`` (Python injection): raw records that go through ``data.transforms``
    exactly like records read from ``data.train``."""
    cfg = rt.config
    rt.data_ctx = DataContext(student_tokenizer=rt.student_tokenizer, teacher_tokenizer=rt.teacher_tokenizer,
                              max_length=cfg.data.max_length, artifacts=rt.artifacts)
    transforms = [R.TRANSFORMS.build(t) for t in cfg.data.transforms]
    for t in transforms:   # K11: transforms get the runtime (tokenizers, artifacts, ...)
        t.setup(rt)
    samples = None
    if train_records is not None:
        samples = run_transforms(train_records, transforms, rt.data_ctx)
    elif cfg.pipeline:
        from lmkd.pipeline.runner import run_pipeline

        samples = run_pipeline(rt)   # None if no stage produces the training samples (K10)
    if samples is None:
        if cfg.data.train is None:
            raise ValueError("no training data: set data.train, a pipeline stage with dataset: true, "
                             "or train_records= in lmkd.build")
        source = R.SOURCES.build(cfg.data.train)
        samples = run_transforms(source.records(), transforms, rt.data_ctx)
    if not samples:
        raise ValueError("no training samples left after transforms")
    rt.train_dataset = PreparedDataset(samples, transforms, rt.data_ctx)
    pads = {"student": rt.student_tokenizer.pad_token_id}
    if rt.teacher_tokenizer is not None:
        pads["teacher"] = rt.teacher_tokenizer.pad_token_id
    fixed = {"student": cfg.data.max_length} if cfg.data.padding == "max_length" else None
    rt.collator = Collator(pads, pad_to_multiple_of=cfg.data.pad_to_multiple_of, fixed_length=fixed)


def build_objective(rt: Runtime) -> None:
    cfg = rt.config
    rt.teacher = R.TEACHERS.build(cfg.teacher.provider)
    rt.teacher_input = R.TEACHER_INPUTS.build(cfg.teacher.input)
    rt.aggregator = R.AGGREGATORS.build(cfg.loss.aggregator)
    rt.aggregator.setup(rt)   # K11
    seen = set()
    for term in cfg.loss.terms:
        crit = R.CRITERIONS.build({"type": term.criterion, **term.params})
        alias = term.alias or term.criterion
        if alias in seen:
            raise ValueError(f"duplicate loss alias '{alias}': set `alias:` explicitly")
        seen.add(alias)
        crit.alias = alias
        rt.criteria.append(CriterionEntry(alias, crit, term.weight))
        rt.loss_weights[alias] = term.weight


def build_modules(rt: Runtime) -> None:
    sa, s = rt.student_adapter, rt.student_model
    info = ModelPairInfo(
        student_hidden=sa.hidden_size(s), student_layers=sa.num_layers(s), student_vocab=sa.vocab_size(s),
    )
    if rt.teacher_model is not None:
        ta, t = rt.teacher_adapter, rt.teacher_model
        assert ta is not None
        info.teacher_hidden, info.teacher_layers, info.teacher_vocab = ta.hidden_size(t), ta.num_layers(t), ta.vocab_size(t)
    rt.extra["model_info"] = info
    rt.modules = ModuleRegistry()
    for e in rt.criteria:
        e.criterion.build_modules(info, rt.modules)
    rt.modules.to(rt.engine.device)
    for e in rt.criteria:
        e.criterion.to(rt.engine.device)


def build_optim(rt: Runtime) -> None:
    cfg = rt.config
    named = [NamedParam(n, p, "student") for n, p in rt.student_model.named_parameters()]
    named += [NamedParam(n, p, "modules", g) for n, p, g in rt.modules.named_params_with_group()]
    if not any(p.requires_grad for p in (n.param for n in named)):
        raise ValueError("nothing to train: student is frozen (models.student.trainable=false) and no aux module")
    groups = build_param_groups(named, cfg.optim.groups, default_lr=cfg.optim.lr,
                                default_wd=cfg.optim.weight_decay,
                                num_layers=rt.student_adapter.num_layers(rt.student_model))
    unknown = set(groups) - set(cfg.optim.optimizers)
    if unknown:
        raise ValueError(f"optim.groups reference undefined optimizers {unknown}")
    rt.optimizers = {name: R.OPTIMIZERS.build(cfg.optim.optimizers[name]).build(g) for name, g in groups.items()}
    for opt_name, gs in groups.items():
        for g in gs:
            for n, p in zip(g["names"], g["params"]):
                rt.trainable_params.append(ParamWithGroup(n, p, g["group_name"], opt_name))


def resolve_total_steps(rt: Runtime) -> int:
    tc = rt.config.train
    if tc.max_steps:
        return tc.max_steps
    spe = rt.producer.steps_per_epoch()
    if spe is None or tc.epochs is None:
        raise ValueError("set train.max_steps (producer has no notion of epoch)")
    return max(1, int(round(spe * tc.epochs)))


def build_models(rt: Runtime, student_model: torch.nn.Module | None = None, student_tokenizer: Any = None,
                 teacher_model: torch.nn.Module | None = None, teacher_tokenizer: Any = None) -> None:
    """Student (+ optional frozen teacher). Objects passed in win over ``models.*`` of the recipe."""
    cfg, engine = rt.config, rt.engine
    rt.student_tokenizer = student_tokenizer if student_tokenizer is not None else load_tokenizer(cfg.models.student)
    if student_model is None:
        student_model = load_model(cfg.models.student, trainable=cfg.models.student.trainable)
    elif not cfg.models.student.trainable:
        student_model.requires_grad_(False)
    rt.student_model = student_model.to(engine.device)
    rt.student_adapter = resolve_adapter(cfg.models.student.adapter)
    rt.student_module = HiddenForward(rt.student_model, rt.student_adapter)

    tspec = cfg.models.teacher
    if tspec is None and teacher_model is None:
        return
    if teacher_tokenizer is None:
        teacher_tokenizer = load_tokenizer(tspec) if tspec else rt.student_tokenizer
    rt.teacher_tokenizer = teacher_tokenizer
    if teacher_model is None:
        assert tspec is not None   # (tspec is None and teacher_model is None) returned above
        teacher_model = load_model(tspec, trainable=False)
    else:
        teacher_model.requires_grad_(False).eval()
    rt.teacher_model = engine.place_frozen(teacher_model, tspec.placement if tspec else "same_device")
    rt.teacher_adapter = resolve_adapter(tspec.adapter if tspec else None)


def build_grad(rt: Runtime) -> None:
    """Gradient hooks and transforms (+ the default clip-norm unless the engine clips itself, K5)."""
    cfg = rt.config
    rt.grad_hooks = [R.GRAD_HOOKS.build(h) for h in cfg.grad.hooks]
    rt.grad_transforms = [R.GRAD_TRANSFORMS.build(t) for t in cfg.grad.transforms]
    if (cfg.grad.clip_norm and not rt.engine.handles_clipping
            and not any(isinstance(t, ClipNorm) for t in rt.grad_transforms)):
        rt.grad_transforms.append(ClipNorm(ClipNorm.Config(max_norm=cfg.grad.clip_norm)))


def build_callbacks(rt: Runtime) -> None:
    """Recipe callbacks + the always-on ones (console_logger, nan_guard, checkpoint, param_schedule if used)."""
    cfg = rt.config
    cbs = [R.CALLBACKS.build(c) for c in cfg.callbacks]
    names = {type(c).registry_name for c in cbs}
    for default in ("console_logger", "nan_guard", "checkpoint"):
        if default not in names:
            cbs.append(R.CALLBACKS.build(default))
    if cfg.loss.schedules and "param_schedule" not in names:
        cbs.append(R.CALLBACKS.build("param_schedule"))
    rt.callbacks = CallbackList(cbs)


def setup_components(rt: Runtime) -> None:
    """``setup(rt)`` of every plugin; things other plugins depend on come first."""
    for comp in [rt.teacher, rt.teacher_input, rt.rollout, rt.producer, rt.algorithm, rt.step_strategy,
                 *rt.grad_hooks, *rt.grad_transforms, *rt.callbacks.cbs, *rt.evaluators]:
        if comp is not None and hasattr(comp, "setup"):
            comp.setup(rt)


def build_schedulers(rt: Runtime) -> None:
    rt.state.total_steps = resolve_total_steps(rt)
    for sc in rt.config.optim.schedulers:
        b = R.SCHEDULERS.build(sc)
        rt.schedulers += [b.build(opt, rt.state.total_steps) for opt in rt.optimizers.values()]


def build_runtime(
    cfg: RecipeConfig,
    *,
    student_model: torch.nn.Module | None = None,
    student_tokenizer: Any = None,
    teacher_model: torch.nn.Module | None = None,
    teacher_tokenizer: Any = None,
    train_records: list[dict] | None = None,
    loss_fn: Any = None,
) -> Runtime:
    """Build every component of a run, in dependency order (each step is a function above).

    Keyword arguments inject Python objects instead of building them from the recipe (library mode,
    ``lmkd.build``). ``loss_fn(batch, rt)`` adds a tier-0 loss (see ``rt.loss_fn``)."""
    from lmkd.plugins import load_plugins

    load_plugins(cfg.imports)
    seed_everything(cfg.train.seed)
    engine = R.ENGINES.build(cfg.engine)
    rt = Runtime(config=cfg, engine=engine)
    rt.loss_fn = loss_fn

    build_models(rt, student_model, student_tokenizer, teacher_model, teacher_tokenizer)
    build_data(rt, train_records)
    build_objective(rt)
    build_modules(rt)
    rt.student_module = engine.wrap_trainable(rt.student_module)   # K1: forward through the engine wrapper
    engine.setup(rt)
    if engine.prepare_before_optim:
        rt.student_module, rt.modules, _ = engine.prepare(rt.student_module, rt.modules, {})
        build_optim(rt)
    else:
        build_optim(rt)
        rt.student_module, rt.modules, rt.optimizers = engine.prepare(rt.student_module, rt.modules, rt.optimizers)

    rt.producer = R.PRODUCERS.build(cfg.producer)
    rt.algorithm = R.ALGORITHMS.build(cfg.algorithm)
    rt.step_strategy = R.STEP_STRATEGIES.build(cfg.step)
    if cfg.rollout is not None:
        rt.rollout = R.ROLLOUT.build(cfg.rollout)
    build_grad(rt)
    build_callbacks(rt)
    rt.evaluators = [R.EVALUATORS.build(e) for e in cfg.eval]
    setup_components(rt)
    build_schedulers(rt)

    pctx = PrepareContext(rt=rt)
    for entry in rt.criteria:
        entry.criterion.prepare(pctx)
    if cfg.train.resume_from:   # T05: after schedulers exist (LambdaLR lambdas are not saved)
        from lmkd.train.checkpointing import load_checkpoint

        load_checkpoint(rt, cfg.train.resume_from)
    return rt
