"""Full checkpoint + deterministic resume (DESIGN §7.6, §11.2; task T05).

Layout of ``<output_dir>/checkpoint-<tag>/``::

    student/               engine.save_model (adapter-only for PEFT) + tokenizer       [rank0]
    modules.pt             saveable aux-module state (gathered under ZeRO-3)           [rank0]
    criterion_state.pt     {alias: criterion.extra_state()}                            [rank0]
    loop_state.pt          {"producer": ..., "engine": engine.state_dict()}            [rank0]
    optim/<name>.pt        engine.optimizer_state_dict(opt) (DeepSpeed: optim/ds/)
    schedulers.pt          [s.state_dict() for s in rt.schedulers]                     [rank0]
    rng/rank<r>.pt         python / numpy / torch / cuda RNG of every rank
    teacher/rank<r>.pt     provider state, including rank-local EMA shards
    train_state.json       rt.state.state_dict() without the per-step metric buffer    [rank0]
    resolved_config.json                                                               [rank0]
    COMPLETE               empty, written LAST after a barrier: marks a valid checkpoint

Every rank calls ``save_checkpoint`` / ``load_checkpoint`` (they contain collectives).
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

COMPLETE = "COMPLETE"


# ----------------------------------------------------------------------------- RNG
def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


# ----------------------------------------------------------------------------- discovery
def _ckpt_step(path: Path) -> int:
    try:
        return int(json.loads((path / "train_state.json").read_text())["step"])
    except (OSError, ValueError, KeyError):
        return -1


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Newest ``checkpoint-*`` with a COMPLETE marker (by saved step, then mtime)."""
    root = Path(output_dir)
    done = [p for p in root.glob("checkpoint-*") if p.is_dir() and (p / COMPLETE).exists()]
    if not done:
        return None
    return max(done, key=lambda p: (_ckpt_step(p), (p / COMPLETE).stat().st_mtime))


def resolve_checkpoint(path: str | Path) -> Path:
    p = Path(path)
    if (p / COMPLETE).exists():
        return p
    if p.name.startswith("checkpoint-") and p.is_dir():
        raise FileNotFoundError(f"{p} has no {COMPLETE} marker (interrupted save?)")
    latest = find_latest_checkpoint(p) if p.is_dir() else None
    if latest is None:
        raise FileNotFoundError(f"no complete checkpoint found in {p}")
    return latest


# ----------------------------------------------------------------------------- save
def save_checkpoint(rt: Any, tag: str) -> Path:
    eng = rt.engine
    path = Path(rt.config.train.output_dir) / f"checkpoint-{tag}"
    if eng.is_main and path.exists():
        shutil.rmtree(path)               # overwrite (e.g. "final") or stale partial save
    eng.barrier()
    path.mkdir(parents=True, exist_ok=True)

    eng.save_model(rt.student_model, path / "student", rt.student_tokenizer)
    if len(rt.modules.mods):
        full = dict(eng.iter_full_state_dict(rt.modules))   # collective under ZeRO-3
        if eng.is_main:
            keep = {k: v for k, v in full.items() if rt.modules.info[k.split(".")[1]].save}
            torch.save(keep, path / "modules.pt")
    eng.save_optimizers(rt.optimizers, path / "optim")
    if eng.is_main:
        torch.save({t.alias: t.criterion.extra_state() for t in rt.criteria}, path / "criterion_state.pt")
        torch.save({"producer": rt.producer.state_dict(), "engine": eng.state_dict()}, path / "loop_state.pt")
        torch.save([s.state_dict() for s in rt.schedulers], path / "schedulers.pt")
        state = rt.state.state_dict()
        state["extra"] = {k: v for k, v in state["extra"].items() if k != "metrics"}
        (path / "train_state.json").write_text(json.dumps(state, default=str))
        (path / "resolved_config.json").write_text(rt.config.model_dump_json(indent=2))
    (path / "rng").mkdir(exist_ok=True)
    torch.save(rng_state(), path / "rng" / f"rank{eng.rank}.pt")
    _save_teacher(rt, path)

    eng.barrier()
    if eng.is_main:
        (path / COMPLETE).touch()
    eng.barrier()
    return path


def _save_teacher(rt: Any, path: Path) -> None:
    """Each rank persists its own provider state; do not broadcast rank-zero EMA shards."""
    teacher = getattr(rt, "teacher", None)
    if teacher is not None:
        folder = path / "teacher"
        folder.mkdir(exist_ok=True)
        torch.save({"provider": rt.config.teacher.provider, "state": teacher.state_dict()},
                   folder / f"rank{rt.engine.rank}.pt")


def _load_teacher(rt: Any, path: Path) -> None:
    import warnings

    teacher = getattr(rt, "teacher", None)
    if teacher is None:
        return
    file = path / "teacher" / f"rank{rt.engine.rank}.pt"
    if not file.exists():
        if (path / "teacher").exists():
            raise FileNotFoundError(f"{file} missing (world size changed?)")
        if teacher.registry_name == "self":
            warnings.warn("legacy checkpoint has no teacher state; self-teacher resume may differ", stacklevel=2)
        return
    saved = torch.load(file, map_location="cpu", weights_only=False)
    if saved["provider"] != rt.config.teacher.provider:
        raise ValueError("checkpoint teacher provider/config differs from the current recipe")
    teacher.load_state_dict(saved["state"])


# ----------------------------------------------------------------------------- load
def _read_student_state(d: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    sd: dict[str, torch.Tensor] = {}
    index = next(iter(sorted(d.glob("*.safetensors.index.json"))), None) or \
        next(iter(sorted(d.glob("*.bin.index.json"))), None)
    if index is not None:
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = [f.name for f in sorted(d.glob("*.safetensors"))] or [f.name for f in sorted(d.glob("*.bin"))]
    if not files:
        raise FileNotFoundError(f"no weights in {d}")
    for f in files:
        sd.update(load_file(d / f) if f.endswith(".safetensors")
                  else torch.load(d / f, map_location="cpu", weights_only=True))
    return sd


def _load_student(rt: Any, d: Path) -> None:
    model = rt.student_model
    if hasattr(model, "peft_config"):            # PEFT: adapter-only checkpoint
        with rt.engine.gathered(list(model.parameters()), modifier_rank=0), torch.no_grad():
            if not rt.engine.is_main and rt.engine.is_partitioned:
                return
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(model, _read_student_state(d))
            return
    res = rt.engine.load_model_state_dict(model, _read_student_state(d))
    if res.unexpected_keys:
        raise RuntimeError(f"unexpected keys in {d}: {res.unexpected_keys[:5]}")
    # tied weights (e.g. lm_head = wte) are not written by save_pretrained
    tied = {n for n, _ in model.named_parameters(remove_duplicate=False)} - \
        {n for n, _ in model.named_parameters()}
    missing = [k for k in res.missing_keys if k not in tied]
    if missing:
        raise RuntimeError(f"missing keys in {d}: {missing[:5]}")


def load_checkpoint(rt: Any, path: str | Path) -> None:
    """Restore everything saved by ``save_checkpoint``. ``path`` = a checkpoint dir or an output dir
    (then the newest checkpoint with COMPLETE is used). Must run after schedulers are built."""
    eng = rt.engine
    path = resolve_checkpoint(path)

    _load_student(rt, path / "student")
    if (path / "modules.pt").exists():
        sd = torch.load(path / "modules.pt", map_location="cpu", weights_only=True)
        res = eng.load_model_state_dict(rt.modules, sd)
        if res.unexpected_keys:
            raise RuntimeError(f"unexpected aux-module keys: {res.unexpected_keys[:5]}")
    crit = torch.load(path / "criterion_state.pt", map_location="cpu", weights_only=False)
    for t in rt.criteria:
        if t.alias in crit:
            t.criterion.load_extra_state(crit[t.alias])
    eng.load_optimizers(rt.optimizers, path / "optim")
    scheds = torch.load(path / "schedulers.pt", map_location="cpu", weights_only=False)
    if len(scheds) != len(rt.schedulers):
        raise ValueError(f"checkpoint has {len(scheds)} schedulers, runtime has {len(rt.schedulers)}")
    for s, st in zip(rt.schedulers, scheds):
        s.load_state_dict(st)
    loop = torch.load(path / "loop_state.pt", map_location="cpu", weights_only=False)
    rt.producer.load_state_dict(loop["producer"])
    eng.load_state_dict(loop["engine"])

    total = rt.state.total_steps                     # the NEW run decides its length
    rt.state.load_state_dict(json.loads((path / "train_state.json").read_text()))
    rt.state.total_steps = total
    rt.state.done = rt.state.step >= total
    _load_teacher(rt, path)

    rng_file = path / "rng" / f"rank{eng.rank}.pt"
    if not rng_file.exists():
        raise FileNotFoundError(f"{rng_file} missing (world size changed?)")
    set_rng_state(torch.load(rng_file, map_location="cpu", weights_only=False))
    eng.barrier()
