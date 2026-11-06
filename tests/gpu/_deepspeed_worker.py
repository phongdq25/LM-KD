"""torchrun worker for test_deepspeed.py. argv: <spec.json>. Writes results next to the spec (rank 0)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import _ds_plugins  # noqa: E402,F401  (registers _t05_proj)

from lmkd.core.config import recipe_from_dict  # noqa: E402
from lmkd.train.builder import build_runtime  # noqa: E402
from lmkd.train.engine import AccelerateEngine, DeepSpeedForward  # noqa: E402
from lmkd.train.trainer import Trainer  # noqa: E402


def full_params(rt) -> dict[str, torch.Tensor]:
    eng = rt.engine
    out = {f"student.{n}": t for n, t in eng.iter_full_state_dict(rt.student_model)}
    out.update({n: t for n, t in eng.iter_full_state_dict(rt.modules)})
    return {k: v.detach().float().cpu() for k, v in out.items()}


def train(spec: dict, out: Path) -> None:
    rt = build_runtime(recipe_from_dict(spec["recipe"]))
    eng = rt.engine
    assert isinstance(eng, AccelerateEngine) and isinstance(rt.student_module, DeepSpeedForward)
    assert not any(type(t).__name__ == "ClipNorm" for t in rt.grad_transforms)
    assert eng.n_micro == spec["expect_ga"], eng.n_micro
    if rt.teacher_model is not None:
        assert id(rt.teacher_model) not in {id(m) for m in eng.ds_engine.module.modules()}
    start = rt.state.step  # > 0 when resumed
    calls = {"n": 0}
    eng.ds_engine.register_forward_pre_hook(lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    Trainer(rt).fit()
    steps = spec["recipe"]["train"]["max_steps"] - start
    assert calls["n"] == steps * eng.n_micro, calls  # K1: every student forward goes via ds_engine
    params = full_params(rt)
    # projector identical on every rank (DeepSpeed reduced the whole bundle)
    proj = {k: v for k, v in params.items() if k.startswith("mods.")}
    gathered = eng.all_gather_object({k: v.tolist() for k, v in proj.items()})
    same = all(g == gathered[0] for g in gathered)
    if spec.get("save_model"):
        eng.save_model(rt.student_model, out / "saved", rt.student_tokenizer)
    eng.barrier()
    if eng.is_main:
        torch.save(params, out / "params.pt")
        (out / "result.json").write_text(
            json.dumps({"proj_same_across_ranks": same, "world": eng.world_size, "zero_stage": eng.zero_stage})
        )


def fp16_skip(spec: dict, out: Path) -> None:
    rt = build_runtime(recipe_from_dict(spec["recipe"]))
    sch = rt.schedulers[0]
    lr0, ep0 = sch.get_last_lr(), sch.last_epoch
    w0 = full_params(rt)
    res = rt.step_strategy.run(rt.producer.next(rt.state), rt)
    w1 = full_params(rt)
    if rt.engine.is_main:
        (out / "result.json").write_text(
            json.dumps(
                {
                    "skipped": res.skipped,
                    "lr_same": sch.get_last_lr() == lr0,
                    "epoch_same": sch.last_epoch == ep0,
                    "params_same": all(torch.equal(w0[k], w1[k]) for k in w0),
                    "loss_finite": bool(torch.isfinite(res.metrics["loss/total"])),
                }
            )
        )


if __name__ == "__main__":
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    {"train": train, "fp16_skip": fp16_skip}[spec["mode"]](spec, spec_path.parent)
