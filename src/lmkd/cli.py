"""Command line: ``lmkd list|check|config|train|pipeline|eval|new``."""
from __future__ import annotations

import argparse
import sys


def _load(args):
    """Recipe + ``--set`` overrides + ``--plugins`` (appended to the recipe's ``imports:``)."""
    from pathlib import Path

    from lmkd.core.config import load_recipe

    cfg = load_recipe(args.config, args.set)
    extra = [str(Path(p).resolve()) if p.endswith(".py") else p for p in getattr(args, "plugins", None) or []]
    if extra:
        cfg.imports = list(dict.fromkeys([*cfg.imports, *extra]))
    return cfg


def _cmd_list(args) -> int:
    from lmkd.core.registry import ALL_REGISTRIES
    from lmkd.plugins import load_plugins

    load_plugins(args.plugins)
    regs = [ALL_REGISTRIES[args.registry]] if args.registry else ALL_REGISTRIES.values()
    for reg in regs:
        print(f"[{reg.name}] ({len(reg)})")
        for name in reg:
            m = reg.meta(name)
            if args.status and m.status != args.status:
                continue
            extra = f"  -> {m.task}" if m.status == "stub" else ""
            print(f"  {name:28s} {m.status:12s} {m.description[:70]}{extra}")
            if args.verbose:
                if m.source:
                    print(f"    source: {m.source}")
                if m.paper:
                    print(f"    paper:  {m.paper}")
                if m.task:
                    print(f"    task:   {m.task}")
                cfg_cls = getattr(m.obj, "Config", None)
                if cfg_cls is not None and hasattr(cfg_cls, "model_fields"):
                    f_strs = []
                    for fname, finfo in cfg_cls.model_fields.items():
                        dflt = finfo.default if finfo.default is not ... else "<required>"
                        f_strs.append(f"{fname}={dflt}")
                    print(f"    config: {', '.join(f_strs)}")
    return 0


def _cmd_check(args) -> int:
    import json

    from lmkd.core.validate import check_recipe, preflight

    cfg = _load(args)
    rep = preflight(cfg) if args.preflight else check_recipe(cfg)
    if args.json:
        payload = {
            "ok": rep.ok,
            "errors": rep.errors,
            "warnings": rep.warnings,
        }
        if rep.info:
            payload["info"] = rep.info
        print(json.dumps(payload))
    else:
        print(rep.render())
    return 0 if rep.ok else 1


def _cmd_config(args) -> int:
    import yaml

    cfg = _load(args)
    print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return 0


def _cmd_train(args) -> int:
    from lmkd.core.validate import check_recipe
    from lmkd.train.trainer import train

    cfg = _load(args)
    rep = check_recipe(cfg)
    if not rep.ok:
        print(rep.render())
        return 1
    train(cfg)
    return 0


def _cmd_pipeline(args) -> int:
    from lmkd.core.validate import check_recipe
    from lmkd.pipeline.runner import build_runtime_for_pipeline, run_pipeline

    cfg = _load(args)
    rep = check_recipe(cfg)
    if not rep.ok:
        print(rep.render())
        return 1

    shard = (0, 1)
    if args.shard:
        parts = args.shard.split("/")
        if len(parts) == 2:
            shard = (int(parts[0]), int(parts[1]))
        else:
            raise ValueError(f"invalid --shard format '{args.shard}', expected 'i/n'")

    rt = build_runtime_for_pipeline(cfg)
    run_pipeline(
        rt,
        until=args.until,
        only=args.only,
        force=args.force,
        shard=shard,
        dry_run=args.dry_run,
    )
    return 0


def _cmd_eval(args) -> int:
    import json
    from pathlib import Path

    from lmkd.train.builder import build_runtime

    cfg = _load(args)
    ckpt = getattr(args, "ckpt", None)
    if ckpt:
        ckpt_p = Path(ckpt)
        student_p = ckpt_p / "student" if (ckpt_p / "student").exists() else ckpt_p
        cfg.models.student.path = str(student_p)
        if (student_p / "adapter_config.json").exists():
            cfg.models.student.lora = None

    cfg.train.max_steps = 1
    # TODO: optimize resource allocation by skipping optimizer building for pure evaluation
    rt = build_runtime(cfg)

    if ckpt:
        student_p = Path(ckpt) / "student" if (Path(ckpt) / "student").exists() else Path(ckpt)
        if (student_p / "adapter_config.json").exists() and not getattr(rt.rollout, "use_lora_request", False):
            from peft import PeftModel

            from lmkd.models.adapters import HiddenForward

            rt.student_model = PeftModel.from_pretrained(rt.student_model, str(student_p))
            wrapped = HiddenForward(rt.student_model, rt.student_adapter)
            rt.student_module = rt.engine.wrap_trainable(wrapped) if hasattr(rt.engine, "wrap_trainable") else wrapped

    metrics = rt.evaluate()
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_explain(args) -> int:
    """Print architectural breakdown, data flow, features, and parameter/memory summary."""
    cfg = _load(args)
    print("=" * 70)
    print(f"Recipe: {getattr(cfg, 'name', 'unnamed')}")
    print("=" * 70)
    print(f"Engine:     {cfg.engine}")
    print(f"Student:    {cfg.models.student.path}")
    if cfg.models.teacher is not None:
        print(f"Teacher:    {cfg.models.teacher.path}")
    print(f"Batch size: {cfg.train.batch_size} (micro: {cfg.train.micro_batch_size})")
    print(f"Max steps:  {cfg.train.max_steps}")
    print("\nLoss Terms:")
    for i, term in enumerate(cfg.loss.terms):
        crit = term.get("criterion", "unknown") if isinstance(term, dict) else getattr(term, "criterion", "unknown")
        alias = term.get("alias", crit) if isinstance(term, dict) else getattr(term, "alias", crit)
        weight = term.get("weight", 1.0) if isinstance(term, dict) else getattr(term, "weight", 1.0)
        print(f"  [{i}] {crit} (alias: {alias}, weight: {weight})")
    print("=" * 70)
    return 0


def _cmd_try(args) -> int:
    """Run a fast 1-step test with synthetic tensors on tiny models to verify loss & gradients in seconds."""
    from lmkd.core.registry import CRITERIONS
    from lmkd.plugins import load_plugins
    from lmkd.scaffold.probe import probe_gradient

    load_plugins(args.plugins)
    name = args.loss
    if name not in CRITERIONS:
        print(f"Error: loss '{name}' is not registered in CRITERIONS.")
        return 1
    meta = CRITERIONS.meta(name)
    print(f"Testing loss '{name}' (status: {meta.status})...")
    try:
        crit = CRITERIONS.build(name)
        out = probe_gradient(crit)
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}")
        print("This probe covers stateless synthetic inputs; criteria needing modules, alignment "
              "or prepare() require a Runtime smoke test.")
        return 1
    print("Success! Loss terms produced:")
    for key, term in out.terms.items():
        print(f"  - {key}: value={float(term.value.detach()):.4f}, reduction={term.reduction}")
    print("Finite student gradients verified.")
    return 0


def _cmd_new(args) -> int:
    from lmkd.scaffold import create_scaffold

    return create_scaffold(args.kind, args.name, domain=args.domain)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lmkd", description="LM-KD distillation framework")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common parent parser for recipe-driven commands
    recipe_parser = argparse.ArgumentParser(add_help=False)
    recipe_parser.add_argument("-c", "--config", required=True, help="path to YAML recipe")
    recipe_parser.add_argument("--set", nargs="*", default=[], action="extend", help="overrides a.b.c=value")
    recipe_parser.add_argument("--plugins", nargs="*", default=[], help="extra plugin files (.py) or modules")

    # list
    pl = sub.add_parser("list", help="list registered plugins")
    pl.add_argument("registry", nargs="?")
    pl.add_argument("--status", choices=["stable", "experimental", "stub"])
    pl.add_argument("-v", "--verbose", action="store_true", help="show source, paper, task, config fields")
    pl.add_argument("--plugins", nargs="*", default=[], help="extra plugin files (.py) or modules to import")
    pl.set_defaults(fn=_cmd_list)

    # check
    pc = sub.add_parser("check", parents=[recipe_parser], help="validate recipe config")
    pc.add_argument("--preflight", action="store_true", help="run 1 micro-batch preflight check")
    pc.add_argument("--json", action="store_true", help="output check result as JSON")
    pc.set_defaults(fn=_cmd_check)

    # config
    pcfg = sub.add_parser("config", parents=[recipe_parser], help="print resolved recipe config")
    pcfg.set_defaults(fn=_cmd_config)

    # train
    sp_tr = sub.add_parser("train", parents=[recipe_parser], help="train student model")
    sp_tr.set_defaults(fn=_cmd_train)

    # eval
    sp_ev = sub.add_parser("eval", parents=[recipe_parser], help="evaluate student checkpoint")
    sp_ev.add_argument("--ckpt", default=None, help="checkpoint path to evaluate")
    sp_ev.set_defaults(fn=_cmd_eval)

    # pipeline
    sp_pipe = sub.add_parser("pipeline", parents=[recipe_parser], help="run offline stages")
    sp_pipe.add_argument("--until", default=None, help="stop after running this stage name")
    sp_pipe.add_argument("--only", nargs="*", default=None, help="run only specified stage name(s)")
    sp_pipe.add_argument("--force", nargs="*", default=[], help="force re-run stage(s) or 'all'")
    sp_pipe.add_argument("--shard", default="0/1", help="shard in i/n format")
    sp_pipe.add_argument("--dry-run", action="store_true", help="show execution plan without running")
    sp_pipe.set_defaults(fn=_cmd_pipeline)

    # new
    pn = sub.add_parser("new", help="scaffold a new plugin, test, and recipe")
    pn.add_argument(
        "kind",
        choices=["criterion", "transform", "stage", "producer", "step_strategy", "grad_transform", "callback"],
    )
    pn.add_argument("name", help="plugin registry name")
    pn.add_argument("--domain", default="experimental", help="domain folder for recipe (criterions only)")
    pn.set_defaults(fn=_cmd_new)

    # explain
    pe = sub.add_parser("explain", parents=[recipe_parser], help="print architectural summary of a recipe")
    pe.set_defaults(fn=_cmd_explain)

    # try
    ptry = sub.add_parser("try", help="test a loss function in seconds with synthetic tensors")
    ptry.add_argument("loss", help="registered loss name to try")
    ptry.add_argument("--plugins", nargs="*", default=[])
    ptry.set_defaults(fn=_cmd_try)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
