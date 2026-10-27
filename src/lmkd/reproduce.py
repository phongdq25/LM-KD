"""Catalogue and reproducible launch plans for DESIGN §1 baselines.

Run ``python -m lmkd.reproduce --list`` from a source checkout. Training uses the
existing CLI/native trainer, or the explicit TRL KTO integration, never a new loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import yaml

from lmkd.core.config import load_recipe, recipe_from_dict
from lmkd.core.validate import check_recipe

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Baseline:
    """One named baseline, recipe and execution backend."""
    domain: str
    recipe: str
    backend: str = "native"


def _catalogue():
    groups = {
        "same_tokenizer": ["sft", "fkl", "rkl", "jsd", "tvd", "sfkl", "srkl", "csd", "gkd", "distillm", "csd_on_policy"],
        "cross_tokenizer": ["dskd", "dskdv2", "dskd_cma", "dskdv2_eta", "uld", "mined", "mlot", "escot", "alm", "sra", "mcw_kd", "mcw_kd_dual", "impact", "dskd_impact", "dskdv2_impact", "sra_impact", "alm_impact", "escot_impact", "residualkd_impact", "dskdv2_impact_first", "dskdv2_impact_last", "dskdv2_impact_ppl", "dskdv2_impact_random", "dskdv2_impact_railkd", "dskdv2_impact_alpkd"],
        "long_cot": ["ssft", "palign", "sgl_spectral", "sgl_spectral_pipeline", "iwc_original", "iwc_stable", "iwc_gated", "answer_gain", "region", "sarw", "tail_boost", "provenance_nll_dft", "provenance_nll_dft_j8", "trans", "sft_nll"],
        "self_distill": ["opsd", "opsd_tinker", "opsd_reason_first", "sdpo", "rlsd", "grpo"],
        "preference": ["vpd", "ppd", "dckd", "tvkd", "adpa", "ctpd", "dp3o", "pbsd", "dpo", "ipo", "simpo", "bpo", "tdpo", "beta_dpo", "epsilon_dpo", "ra_dpo", "adpo", "tbpo_a", "kto"],
        "embedding": ["talas_m", "talas_asam"],
    }
    result = {name: Baseline(domain, f"recipes/{domain}/{name}.yaml")
              for domain, names in groups.items() for name in names}
    result["residualkd"] = Baseline("cross_tokenizer", "recipes/cross_tokenizer/residualkd_stage2.yaml")
    result["kto"] = Baseline("preference", "recipes/preference/kto.yaml", "trl_kto")
    for name in ("tis_dpo", "ti_dpo", "tbpo_q", "logits_sam"):
        result[name] = Baseline("preference", f"recipes/preference/{name}.yaml", "po_reference")
    for domain, names in {"self_distill": ["opsd_topk"], "preference": ["vpd_pairwise", "pad"],
                          "cross_tokenizer": ["dskdv2_impact_ratio1to1", "dskdv2_impact_bi_ratio"]}.items():
        for name in names:
            result[name] = Baseline(domain, f"recipes/{domain}/{name}.yaml")
    for name in groups["cross_tokenizer"]:
        if name.startswith("dskdv2_impact_"):
            result[name] = Baseline("cross_tokenizer", f"recipes/cross_tokenizer/ablation/{name}.yaml")
    # Paper/method aliases name the complete pipeline, not a partial cached-input recipe.
    for alias, name in {"adaptive_sgo": "distillm", "mixed": "gkd",
                        "iwc": "iwc_original",
                        "dskd_cma_impact": "dskd_impact",
                        "dskdv2_eta_impact": "dskdv2_impact"}.items():
        result[alias] = result[name]
    return result


BASELINES = _catalogue()


def _sources(node, data=None):
    """Rebase local sources; replace training inputs while preserving artifact edges."""
    if isinstance(node, dict):
        if node.get("type") in {"jsonl", "csv", "parquet"} and "path" in node:
            node["path"] = str(Path(data).resolve()) if data else str((ROOT / node["path"]).resolve())
        else:
            for value in node.values():
                _sources(value, data)
    elif isinstance(node, list):
        for value in node:
            _sources(value, data)


def _configure(raw, output, *, student=None, teacher=None, reference=None, embedding_teacher=None,
               data=None, dev_data=None, stage1=False):
    model = teacher if stage1 else student
    if model:
        raw["models"]["student"]["path"] = model
    if teacher and raw["models"].get("teacher"):
        raw["models"]["teacher"]["path"] = teacher
    needs_reference = any(t["criterion"] in {"dpo", "po", "ctpd", "dp3o"} for t in raw["loss"]["terms"])
    if reference and raw["models"].get("teacher") and (needs_reference or raw["compat"].get("reproduction_backend")):
        raw["models"]["teacher"]["path"] = reference
    _sources(raw["data"], data)
    _sources(raw["eval"], dev_data)
    for stage in raw["pipeline"]:
        _sources(stage.get("source", {}), data)
        if student and stage.get("stage") in {"spectral_capture", "ig_attribution", "answer_gain", "palign_search", "generate"}:
            stage["model"] = student
        if teacher and stage.get("stage") in {"teacher_cache", "score"}:
            stage["model"] = teacher
        if reference and "reference_model" in stage:
            stage["reference_model"] = reference
        if embedding_teacher and stage.get("stage") == "embed_teacher":
            stage["model"] = embedding_teacher
    raw["train"]["output_dir"] = str(output)
    raw["artifacts_dir"] = str(output / "artifacts")
    return raw


def _isolate_caches(node, output):
    """Keep model-dependent vocabulary/attribution caches inside this experiment."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "cache_path" and isinstance(value, str):
                node[key] = str(output / "caches" / Path(value).name)
            else:
                _isolate_caches(value, output)
    elif isinstance(node, list):
        for item in node:
            _isolate_caches(item, output)


def resolve_jobs(name, output, *, overrides=None, **options):
    """Resolve all phases and paths without loading models or reading training data."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; use --list")
    output = Path(output).resolve()
    paths = [BASELINES[name].recipe]
    if name in {"residualkd", "residualkd_impact"}:
        paths.insert(0, "recipes/cross_tokenizer/residualkd_stage1.yaml")
    jobs = []
    for i, path in enumerate(paths):
        dest = output / f"stage{i + 1}" if len(paths) > 1 else output
        raw = load_recipe(ROOT / path).model_dump(mode="json")
        raw = _configure(raw, dest, stage1=len(paths) > 1 and i == 0, **options)
        _isolate_caches(raw, dest)
        cfg = recipe_from_dict(raw, overrides)
        jobs.append(cfg)
    if len(jobs) > 1:
        if jobs[0].train.output_dir == jobs[1].train.output_dir:
            raise ValueError("ResidualKD phases need distinct output directories; use --output-dir")
        # Phase 1 pretrains P_TA on exactly the phase-2 teacher, including CLI overrides.
        teacher_spec = jobs[1].models.teacher
        if teacher_spec is None:
            raise ValueError("ResidualKD phase 2 requires models.teacher")
        jobs[0].models.student = teacher_spec.model_copy(deep=True)
        jobs[0].models.student.trainable = False
        for term in jobs[1].loss.terms:
            if term.criterion == "residualkd":
                term.params["projector_ckpt"] = str(Path(jobs[0].train.output_dir) / "checkpoint-final/modules.pt")
    return jobs


def validate_jobs(name, jobs):
    """Static validation, with the external KTO trainer supplying its own objective."""
    injected = frozenset({"loss_fn"}) if BASELINES[name].backend != "native" else frozenset()
    reports = [check_recipe(cfg, injected=injected) for cfg in jobs]
    errors = [r.render() for r in reports if not r.ok]
    if errors:
        raise ValueError("\n".join(errors))
    return reports


def _input_hashes(node):
    paths = set()

    def collect(value):
        if isinstance(value, dict):
            if value.get("type") in {"jsonl", "csv", "parquet"} and "path" in value:
                paths.add(value["path"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(node)
    hashes: dict[str, str | None] = {}
    for path in sorted(paths):
        p = Path(path)
        if p.is_file():
            with p.open("rb") as stream:
                hashes[path] = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            hashes[path] = None
    return hashes


def write_plan(name, jobs, output):
    """Save exact configs, input hashes and launch commands for review/re-execution."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands, configs = [], []
    for i, cfg in enumerate(jobs):
        path = output / f"resolved-stage{i + 1}.yaml"
        payload = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
        path.write_text(payload)
        module = {"native": "lmkd.cli", "trl_kto": "lmkd.integrations.kto_runner",
                  "po_reference": "lmkd.integrations.po_reference_runner"}[BASELINES[name].backend]
        command = [sys.executable, "-m", module]
        if module == "lmkd.cli":
            command.append("train")
        commands.append([*command, "-c", str(path)])
        configs.append({"path": str(path), "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                        "inputs": _input_hashes(cfg.model_dump(mode="json"))})
    plan = {"baseline": name, "backend": BASELINES[name].backend, "cwd": str(ROOT),
            "environment": _environment(name, jobs),
            "commands": commands, "configs": configs, "status": "planned"}
    (output / "reproduction.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def _environment(name, jobs):
    """Record dependency versions and source content, including reference backend code."""
    versions = {}
    for package in ("lmkd", "torch", "transformers", "trl", "vllm", "peft", "deepspeed", "datasets"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    roots = [ROOT / "src/lmkd"]
    if BASELINES[name].backend == "po_reference":
        source_root = Path(jobs[0].compat.get("po_source_root", ROOT.parent / "Awesome-Human-AI-Alignment-main"))
        roots.append(source_root / "src/human_alignment")
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for root in roots for path in sorted(root.rglob("*.py"))}
    return {"python": sys.version, "packages": versions, "source_sha256": hashes}


def execute_plan(plan, output):
    """Run phases sequentially, preserving failed status and subprocess exit code."""
    target = Path(output).resolve() / "reproduction.json"
    plan["status"] = "running"
    target.write_text(json.dumps(plan, indent=2) + "\n")
    try:
        for command in plan["commands"]:
            subprocess.run(command, cwd=plan["cwd"], check=True)
    except subprocess.CalledProcessError as exc:
        plan.update(status="failed", returncode=exc.returncode)
        raise
    else:
        plan["status"] = "completed"
    finally:
        target.write_text(json.dumps(plan, indent=2) + "\n")


def main(argv=None):
    """List, validate, prepare or execute one baseline launch plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true", help="validate every catalogue entry (requires --check)")
    parser.add_argument("--check", action="store_true", help="static validation only; no model loading")
    parser.add_argument("--dry-run", action="store_true", help="save resolved configs/commands without training")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--domain", choices=sorted({b.domain for b in BASELINES.values()}))
    for key in ("student", "teacher", "reference", "embedding-teacher", "data", "dev-data"):
        parser.add_argument(f"--{key}")
    parser.add_argument("--set", nargs="*", default=[], action="extend")
    args = parser.parse_args(argv)
    if args.list:
        for name, baseline in BASELINES.items():
            if args.domain is None or baseline.domain == args.domain:
                print(f"{name:28} {baseline.domain:16} {baseline.backend:8} {baseline.recipe}")
        return 0
    if args.all and not args.check:
        parser.error("--all requires --check; select one baseline for training")
    names = list(BASELINES) if args.all else [args.baseline]
    if not args.all and args.baseline not in BASELINES:
        parser.error("choose a baseline from --list")
    options = {key: getattr(args, key) for key in ("student", "teacher", "reference", "embedding_teacher", "data", "dev_data")}
    try:
        for name in names:
            if args.domain and BASELINES[name].domain != args.domain:
                continue
            output = Path(args.output_dir or ROOT / "outputs/reproduction" / name)
            jobs = resolve_jobs(name, output, overrides=args.set, **options)
            validate_jobs(name, jobs)
            print(f"{name}: validated {len(jobs)} phase(s)")
            if not args.check:
                plan = write_plan(name, jobs, output)
                print(json.dumps(plan, indent=2))
                if not args.dry_run:
                    missing = [p for cfg in plan["configs"] for p, digest in cfg["inputs"].items() if digest is None]
                    if missing:
                        raise ValueError(f"Missing local data files: {sorted(set(missing))}")
                    execute_plan(plan, output)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
