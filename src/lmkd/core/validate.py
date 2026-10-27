"""Static checks run by ``lmkd check`` before any model is loaded (DESIGN §13.3).

Checks:
  1. every plugin spec resolves to a registered, non-stub plugin with a valid Config;
  2. the ``requires``/``provides`` key graph is satisfied: each key required by a criterion /
     algorithm / teacher builder is provided upstream (stage, transform, producer, builder);
  3. teacher presence for criteria that need one; batch size divisibility;
  4. GradMag aggregator and multi-objective step configuration agree.
Teacher signal capability is checked at build time (Algorithm.setup) because it needs instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lmkd.core import registry as R
from lmkd.core.config import RecipeConfig


@dataclass
class CheckReport:
    """Result of ``lmkd check``: errors, warnings, the plugins resolved and optional preflight info."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"plugins: {len(self.plugins)}"]
        lines += [f"  - {p}" for p in self.plugins]
        if self.info:
            lines.append("info:")
            lines += [f"  {line}" for line in self.info]
        lines += [f"WARNING: {w}" for w in self.warnings]
        lines += [f"ERROR: {e}" for e in self.errors]
        lines.append("OK" if self.ok else f"FAILED ({len(self.errors)} errors)")
        return "\n".join(lines)


class _Graph:
    def __init__(self, rep: CheckReport):
        self.rep = rep
        self.provided: set[str] = set()

    def visit(self, reg: R.Registry, spec: Any, where: str, *, check_requires: bool = True) -> R.PluginMeta | None:
        try:
            meta, cfg = reg.validate(spec)
        except Exception as e:  # noqa: BLE001 - report everything
            self.rep.errors.append(f"{where}: {e}")
            return None
        self.rep.plugins.append(f"{where}: [{reg.name}] {meta.name} ({meta.status})")
        obj = meta.obj
        if check_requires and hasattr(obj, "required_keys"):
            for k in obj.required_keys(cfg):
                if k not in self.provided:
                    self.rep.errors.append(
                        f"{where}: '{meta.name}' requires batch key '{k}' but no upstream plugin provides it. "
                        f"Provided so far: {sorted(self.provided)}"
                    )
        if hasattr(obj, "provided_keys"):
            self.provided.update(obj.provided_keys(cfg))
        return meta


def check_recipe(cfg: RecipeConfig, *, injected: frozenset[str] = frozenset()) -> CheckReport:
    """``injected``: parts given as Python objects by ``lmkd.build`` (``"train_records"``,
    ``"loss_fn"``) so their recipe section may be absent.

    Order matters: ``_Graph`` accumulates the batch keys provided so far, so plugins are visited in
    data-flow order (sources -> pipeline -> transforms -> producer -> teacher -> losses -> the rest)."""
    from lmkd.plugins import load_plugins

    rep = CheckReport()
    try:
        load_plugins(cfg.imports)
    except Exception as exc:  # noqa: BLE001
        rep.errors.append(f"imports: {type(exc).__name__}: {exc}")
        return rep
    g = _Graph(rep)

    _check_data(cfg, g, injected)
    _check_pipeline(cfg, g)
    for i, t in enumerate(cfg.data.transforms):
        g.visit(R.TRANSFORMS, t, f"data.transforms[{i}]")
    _check_algorithm_parts(cfg, g)
    _check_nested_plugins(cfg, g)
    _check_loss(cfg, g, injected)
    _teacher_signal_check(cfg, rep)
    _check_packing(cfg, rep)
    _check_warnings(cfg, rep)
    _check_optimization_and_extras(cfg, g)
    _check_aggregator_step(cfg, rep)
    _check_rl_options(cfg, rep)
    if cfg.train.batch_size % cfg.train.micro_batch_size:
        rep.errors.append("train.batch_size must be divisible by train.micro_batch_size (x world size)")
    return rep


def _check_data(cfg: RecipeConfig, g: _Graph, injected: frozenset[str]) -> None:
    g.visit(R.ENGINES, cfg.engine, "engine")
    if cfg.data.train is not None:
        g.visit(R.SOURCES, cfg.data.train, "data.train")
    elif not cfg.pipeline and "train_records" not in injected:
        g.rep.errors.append("data.train is missing (give a source, a pipeline, or train_records= in lmkd.build)")


def _check_pipeline(cfg: RecipeConfig, g: _Graph) -> None:
    """Stage names are unique, inputs refer to EARLIER stages, every stage spec validates."""
    from lmkd.pipeline.runner import split_entry

    rep = g.rep
    names: set[str] = set()
    dataset_nodes: list[str] = []
    for i, entry in enumerate(cfg.pipeline):
        try:
            stype, name, inputs, _, dataset, params = split_entry(entry)
        except ValueError as exc:
            rep.errors.append(f"pipeline[{i}]: {exc}")
            continue
        if name in names:
            rep.errors.append(f"pipeline[{i}]: duplicate stage name '{name}'")
        for src in (inputs or {}).values():
            if src not in names:
                rep.errors.append(f"pipeline[{i}]: input '{src}' is not an earlier stage name")
        names.add(name)
        if dataset:
            dataset_nodes.append(name)
        g.visit(R.STAGES, {"type": stype, **params}, f"pipeline[{i}] ({name})")
    if len(dataset_nodes) > 1:
        rep.warnings.append(
            f"multiple stages marked dataset: true {dataset_nodes}; using last: '{dataset_nodes[-1]}'"
        )


def _check_algorithm_parts(cfg: RecipeConfig, g: _Graph) -> None:
    g.visit(R.PRODUCERS, cfg.producer, "producer")
    g.visit(R.TEACHER_INPUTS, cfg.teacher.input, "teacher.input")
    g.visit(R.TEACHERS, cfg.teacher.provider, "teacher.provider")
    g.visit(R.ALGORITHMS, cfg.algorithm, "algorithm")
    g.visit(R.AGGREGATORS, cfg.loss.aggregator, "loss.aggregator")


def _check_loss(cfg: RecipeConfig, g: _Graph, injected: frozenset[str]) -> None:
    """Loss source exists (criteria / compute_loss / loss_fn), criteria validate, teacher is present
    where needed."""
    rep = g.rep
    algo_name, algo_params = R.Registry.split_spec(cfg.algorithm)
    if algo_name == "compute_loss" and algo_params.get("fn") is not None:
        try:
            R.LOSS_FNS.meta(algo_params["fn"])
        except R.RegistryError as exc:
            rep.errors.append(f"algorithm.fn: {exc.args[0]} (list the file defining it under `imports:`)")
    if not cfg.loss.terms and algo_name != "compute_loss" and "loss_fn" not in injected:
        rep.errors.append("loss.terms is empty: add criteria, use `algorithm: {type: compute_loss, fn: ...}`, "
                          "or pass loss_fn= to lmkd.build / rt.train_step")
    no_teacher = cfg.teacher.provider in ("none", {"type": "none"})
    for i, term in enumerate(cfg.loss.terms):
        meta = g.visit(R.CRITERIONS, {"type": term.criterion, **term.params}, f"loss.terms[{i}]",
                       check_requires=term.weight != 0.0)
        if meta is not None and getattr(meta.obj, "needs_teacher", False) and no_teacher and term.weight != 0.0:
            rep.errors.append(f"loss.terms[{i}]: '{term.criterion}' needs a teacher but teacher.provider is 'none'")
    if not no_teacher and cfg.models.teacher is None and cfg.teacher.provider in ("hf_frozen", {"type": "hf_frozen"}):
        rep.errors.append("teacher.provider 'hf_frozen' needs models.teacher")


def _check_nested_plugins(cfg: RecipeConfig, g: _Graph) -> None:
    """Validate nested specs using their own registry, without loading models."""
    parents = ((R.PRODUCERS, cfg.producer, "producer", "verifier", R.REWARDS),
               (R.ALGORITHMS, cfg.algorithm, "algorithm", "advantage", R.ADVANTAGES),
               (R.TEACHERS, cfg.teacher.provider, "teacher.provider", "weight_policy", R.WEIGHT_POLICIES))
    for registry, spec, where, key, nested in parents:
        try:
            _, params = registry.validate(spec)
        except Exception:
            continue  # Parent visit already reports invalid configs.
        if hasattr(params, key):
            g.visit(nested, getattr(params, key), f"{where}.{key}")
        if where == "algorithm" and hasattr(params, "adv_transforms"):
            for i, transform in enumerate(params.adv_transforms):
                meta = g.visit(R.ADV_TRANSFORMS, transform, f"algorithm.adv_transforms[{i}]")
                if meta is not None and getattr(meta.obj, "needs_teacher_logp", False):
                    try:
                        teacher, _ = R.TEACHERS.validate(cfg.teacher.provider)
                    except Exception:
                        continue
                    if "sampled_logp" not in teacher.obj.capabilities:
                        g.rep.errors.append(f"algorithm.adv_transforms[{i}] requires a teacher with sampled_logp")


def _check_rl_options(cfg: RecipeConfig, rep: CheckReport) -> None:
    try:
        producer, pcfg = R.PRODUCERS.validate(cfg.producer)
        algorithm, acfg = R.ALGORITHMS.validate(cfg.algorithm)
        step, scfg = R.STEP_STRATEGIES.validate(cfg.step)
    except Exception:
        return
    if producer.name == "rollout_group":
        if cfg.rollout is None:
            rep.errors.append("producer.rollout_group requires rollout")
        if cfg.train.batch_size % pcfg.n:
            rep.errors.append("train.batch_size must be divisible by producer.n")
    if step.name == "ppo_epochs" and cfg.train.batch_size % scfg.mini_batches:
        rep.errors.append("train.batch_size must be divisible by step.mini_batches")
    if algorithm.name != "rl":
        return
    if acfg.rollout_is not in ("none", "token"):
        rep.errors.append("algorithm.rollout_is must be none or token")
    if not acfg.recompute_old_logp and (acfg.adv_transforms or acfg.rollout_is != "none"
            or getattr(scfg, "epochs", 1) != 1 or getattr(scfg, "mini_batches", 1) != 1):
        rep.errors.append("algorithm.recompute_old_logp=False requires one update without reweighting")
    for old, new in (("ppo_epochs", "epochs"), ("mini_batches", "mini_batches")):
        if old in acfg.model_fields_set and getattr(acfg, old) != getattr(scfg, new, 1):
            rep.errors.append(f"algorithm.{old} must match step.{new}")


def _check_packing(cfg: RecipeConfig, rep: CheckReport) -> None:
    if not cfg.data.packing:
        return
    for term in cfg.loss.terms:
        try:
            meta = R.CRITERIONS.meta(term.criterion)
        except R.RegistryError:
            continue   # reported by _check_loss
        if getattr(meta.obj, "supports_packing", True) is False:
            rep.errors.append(f"data.packing is enabled but criterion '{term.criterion}' has supports_packing=False")


def _check_warnings(cfg: RecipeConfig, rep: CheckReport) -> None:
    pname, _ = R.Registry.split_spec(cfg.producer)
    if cfg.rollout is not None and pname == "dataset" and not cfg.eval:
        rep.warnings.append("rollout configured but unused")
    tname, _ = R.Registry.split_spec(cfg.teacher.provider)
    if cfg.models.teacher is not None and tname == "none":
        rep.warnings.append("teacher model loaded but no teacher provider (wasted memory)")
    for term in cfg.loss.terms:
        if term.weight == 0.0:
            rep.warnings.append(f"term '{term.alias or term.criterion}' has weight 0 and is skipped")
    if cfg.train.max_steps is not None and cfg.train.epochs is not None:
        rep.warnings.append("max_steps overrides epochs")


def _check_optimization_and_extras(cfg: RecipeConfig, g: _Graph) -> None:
    g.visit(R.STEP_STRATEGIES, cfg.step, "step")
    for i, sched in enumerate(cfg.optim.schedulers):
        g.visit(R.SCHEDULERS, sched, f"optim.schedulers[{i}]")
    for name, opt in cfg.optim.optimizers.items():
        g.visit(R.OPTIMIZERS, opt, f"optim.optimizers.{name}")
    for i, hook in enumerate(cfg.grad.hooks):
        g.visit(R.GRAD_HOOKS, hook, f"grad.hooks[{i}]")
    for i, transform in enumerate(cfg.grad.transforms):
        g.visit(R.GRAD_TRANSFORMS, transform, f"grad.transforms[{i}]")
    for i, cb in enumerate(cfg.callbacks):
        g.visit(R.CALLBACKS, cb, f"callbacks[{i}]")
    for i, ev in enumerate(cfg.eval):
        g.visit(R.EVALUATORS, ev, f"eval[{i}]")
    if cfg.rollout is not None:
        g.visit(R.ROLLOUT, cfg.rollout, "rollout")


def _check_aggregator_step(cfg: RecipeConfig, rep: CheckReport) -> None:
    """Report GradMag pairings that the runtime would reject during setup."""
    try:
        aggregator, _ = R.Registry.split_spec(cfg.loss.aggregator)
        step, params = R.Registry.split_spec(cfg.step)
    except ValueError:
        return  # The corresponding registry visit has already reported the malformed spec.
    combine = params.get("combine", "gradmag")
    if aggregator == "gradmag" and (step != "multi_objective_grad" or combine != "gradmag"):
        rep.errors.append("loss.aggregator: gradmag requires step: multi_objective_grad with combine=gradmag")
    elif step == "multi_objective_grad" and combine == "gradmag" and aggregator != "gradmag":
        rep.errors.append("step: multi_objective_grad combine=gradmag requires loss.aggregator: gradmag")


def _teacher_signal_check(cfg: RecipeConfig, rep: CheckReport) -> None:
    from lmkd.models.forward import FeatureRequest

    req = FeatureRequest()
    needers: dict[str, list[str]] = {}
    for term in cfg.loss.terms:
        alias = term.alias or term.criterion
        is_active = (term.weight != 0.0) or (alias in cfg.loss.schedules)
        if not is_active:
            continue
        try:
            crit = R.CRITERIONS.build({"type": term.criterion, **term.params})
        except Exception:
            continue
        try:
            crit_req = crit.features()
        except Exception as exc:
            rep.errors.append(f"loss term {alias!r} features: {exc}")
            continue
        if crit_req.teacher_signal is not None and crit_req.teacher_signal.kind != "none":
            needers.setdefault(crit_req.teacher_signal.kind, []).append(alias)
        try:
            req = req.merge(crit_req)
        except ValueError as e:
            rep.errors.append(f"teacher signal conflict: {e}")
            return

    if req.teacher is None and req.teacher_signal is None:
        return

    try:
        name, _ = R.Registry.split_spec(cfg.teacher.provider)
    except Exception:
        return
    if name not in R.TEACHERS:
        return
    prov_cls = R.TEACHERS.get(name)
    spec = req.teacher_signal
    if spec is not None and spec.kind != "none" and spec.kind not in prov_cls.capabilities:
        rep.errors.append(
            f"teacher.provider '{name}' cannot produce signal '{spec.kind}' "
            f"(capabilities {sorted(prov_cls.capabilities)}); criteria need it: {needers.get(spec.kind, [])}"
        )
    if (req.teacher is not None and (req.teacher.hidden != "none" or req.teacher.attentions != "none")
            and "hidden" not in prov_cls.capabilities):
        rep.errors.append(
            f"teacher.provider '{name}' cannot produce hidden/attentions signals (capabilities {sorted(prov_cls.capabilities)})"
        )


def preflight(cfg: RecipeConfig, n_samples: int = 2) -> CheckReport:
    """Build the full runtime on a TINY data subset and run ONE forward+loss (no backward,
    no optimizer step). Reports the batch keys/shapes, feature request and loss terms."""
    import tempfile
    import traceback

    import torch

    from lmkd.train.builder import build_runtime

    rep = check_recipe(cfg)
    if not rep.ok:
        rep.errors.append(f"preflight: static check failed with {len(rep.errors)} error(s)")
        return rep

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg2 = cfg.model_copy(deep=True)
        cfg2.train.max_steps = 1
        batch_size, mbs, prompts = _preflight_sizes(cfg2)
        cfg2.train.batch_size = batch_size
        cfg2.train.micro_batch_size = mbs
        cfg2.train.output_dir = tmpdir

        stype = R.Registry.split_spec(cfg2.data.train)[0] if cfg2.data.train is not None else None
        if stype == "jsonl" and isinstance(cfg2.data.train, dict):
            cfg2.data.train["limit"] = max(n_samples, prompts)

        try:
            rt = build_runtime(cfg2)
            batch = rt.producer.next(rt.state)
            batch = rt.algorithm.augment(batch, rt)
            micro = rt.step_strategy.micro_batches(batch)[0]

            with torch.no_grad():
                outs = rt.algorithm.loss(micro, rt)
                norms = rt.step_strategy.global_norms([micro])
                loss, mets = rt.aggregator.combine(outs, norms)

            rep.info.append("micro-batch:")
            for k in sorted(micro):
                v = micro[k]
                shape = list(v.shape) if hasattr(v, "shape") else type(v).__name__
                rep.info.append(f"  {k}: {shape}")

            rep.info.append(f"request: {rt.algorithm.request!r}")

            rep.info.append("loss terms:")
            for k, v in sorted(mets.items()):
                val_str = f"{float(v):.6f}" if hasattr(v, "item") or isinstance(v, (int, float)) else str(v)
                rep.info.append(f"  {k}: {val_str}")

            group_counts: dict[str, int] = {}
            for pg in rt.trainable_params:
                group_counts[pg.group] = group_counts.get(pg.group, 0) + pg.param.numel()
            rep.info.append(f"trainable parameters: {group_counts}")

        except Exception as e:
            tb = traceback.format_exc().strip().splitlines()
            last_tb = "\n".join(tb[-5:])
            rep.errors.append(f"preflight: {type(e).__name__}: {e}\n{last_tb}")

    return rep


def _preflight_sizes(cfg: RecipeConfig) -> tuple[int, int, int]:
    """Preserve groups/minibatches when shrinking a rollout or listwise preflight."""
    import math
    import os

    producer, params = R.PRODUCERS.validate(cfg.producer)
    _, step = R.STEP_STRATEGIES.validate(cfg.step)
    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    micro = min(2, cfg.train.micro_batch_size)
    group = 1
    if producer.name == "rollout_group":
        group = params.n
    elif producer.name == "listwise":
        # Without explicit n the producer infers it from data. The original micro must contain whole groups.
        group = params.pad_groups_to or cfg.train.micro_batch_size
        micro = group
    rows_per_rank = math.lcm(group, micro * getattr(step, "mini_batches", 1))
    rows = rows_per_rank * world
    prompts = rows // group if producer.name in ("rollout_group", "listwise") else rows
    return rows, micro, prompts
