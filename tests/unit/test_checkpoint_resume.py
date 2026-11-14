"""T05: full checkpoint (optimizer/scheduler/RNG/producer) + deterministic resume, COMPLETE, keep-last/best."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from lmkd.core.config import recipe_from_dict
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CALLBACKS
from lmkd.train.builder import build_runtime
from lmkd.train.callbacks import Callback
from lmkd.train.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from lmkd.train.trainer import Trainer


def _recipe(base_recipe, out: Path, *, max_steps: int, every: int | None, keep_last=None, **ckpt):
    base_recipe["train"].update(output_dir=str(out), max_steps=max_steps, log_every=1)
    base_recipe["optim"]["schedulers"] = [{"type": "constant"}]
    base_recipe["callbacks"] = [{"type": "checkpoint", "every": every, "keep_last": keep_last, **ckpt}]
    return recipe_from_dict(base_recipe)


def _losses(out: Path) -> dict[int, float]:
    rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    return {r["step"]: r["loss/total"] for r in rows}


def test_resume_deterministic(base_recipe, tmp_path):
    a = Trainer.from_config(_recipe(copy.deepcopy(base_recipe), tmp_path / "A", max_steps=6, every=3)).fit()
    Trainer.from_config(_recipe(copy.deepcopy(base_recipe), tmp_path / "B", max_steps=3, every=3)).fit()
    assert (tmp_path / "B" / "checkpoint-3" / "COMPLETE").exists()
    cfg_c = _recipe(copy.deepcopy(base_recipe), tmp_path / "C", max_steps=6, every=3)
    cfg_c.train.resume_from = str(tmp_path / "B" / "checkpoint-3")
    rt_c = build_runtime(cfg_c)
    assert rt_c.state.step == 3 and not rt_c.state.done and rt_c.state.total_steps == 6
    c = Trainer(rt_c).fit()

    pa = dict(a.student_model.named_parameters())
    for n, p in c.student_model.named_parameters():
        assert torch.allclose(p, pa[n], atol=1e-6), n
    la, lc = _losses(tmp_path / "A"), _losses(tmp_path / "C")
    assert sorted(lc) == [4, 5, 6]
    for s in (4, 5, 6):
        assert la[s] == pytest.approx(lc[s], abs=1e-6)


def test_complete_marker_and_latest(base_recipe, tmp_path):
    out = tmp_path / "run"
    Trainer.from_config(_recipe(base_recipe, out, max_steps=2, every=1, at_end=False)).fit()
    assert find_latest_checkpoint(out) == out / "checkpoint-2"
    # an interrupted (newer) save without COMPLETE must be ignored
    broken = out / "checkpoint-7"
    broken.mkdir()
    (broken / "train_state.json").write_text("{}")
    assert find_latest_checkpoint(out) == out / "checkpoint-2"

    rt = build_runtime(_recipe(base_recipe, tmp_path / "other", max_steps=5, every=None))
    load_checkpoint(rt, out)
    assert rt.state.step == 2
    with pytest.raises(FileNotFoundError):
        load_checkpoint(rt, broken)


def test_keep_last(base_recipe, tmp_path):
    out = tmp_path / "run"
    Trainer.from_config(_recipe(base_recipe, out, max_steps=5, every=1, keep_last=2)).fit()
    names = sorted(p.name for p in out.iterdir() if p.name.startswith("checkpoint-"))
    assert names == ["checkpoint-4", "checkpoint-5", "checkpoint-final"]


def test_keep_last_larger_than_number_of_checkpoints(base_recipe, tmp_path):
    """Regression: ckpts[len-keep_last:] used a negative start and deleted all but the newest."""
    out = tmp_path / "run"
    Trainer.from_config(_recipe(base_recipe, out, max_steps=4, every=2, keep_last=3)).fit()
    names = sorted(p.name for p in out.iterdir() if p.name.startswith("checkpoint-"))
    assert names == ["checkpoint-2", "checkpoint-4", "checkpoint-final"]


def test_keep_best(base_recipe, tmp_path):
    if "_test_fake_eval" not in CALLBACKS:

        @CALLBACKS.register("_test_fake_eval")
        class _FakeEval(Callback):
            order = 10

            class Config(BaseConfig):
                pass

            def on_step_end(self, rt, out):  # eval/loss: 5, 1, 4, 2, 3
                rt.state.extra["eval"] = {"eval/loss": [5.0, 1.0, 4.0, 2.0, 3.0][rt.state.step - 1]}

    out = tmp_path / "run"
    cfg = _recipe(base_recipe, out, max_steps=5, every=1, keep_last=1, keep_best=2, at_end=False)
    cfg.callbacks.append({"type": "_test_fake_eval"})
    Trainer.from_config(cfg).fit()
    names = sorted(p.name for p in out.iterdir() if p.name.startswith("checkpoint-"))
    assert names == ["checkpoint-2", "checkpoint-4", "checkpoint-5"]  # best two (1.0, 2.0) + last


def test_optimizer_and_scheduler_restored(base_recipe, tmp_path):
    base_recipe["train"].update(output_dir=str(tmp_path / "run"), max_steps=6)
    base_recipe["optim"]["schedulers"] = [{"type": "linear", "warmup_ratio": 0.5}]
    rt = build_runtime(recipe_from_dict(base_recipe))
    for _ in range(2):
        rt.step_strategy.run(rt.producer.next(rt.state), rt)
        rt.state.step += 1
    path = save_checkpoint(rt, "x")
    opt = next(iter(rt.optimizers.values()))
    saved = {id(p): opt.state[p]["exp_avg"].clone() for g in opt.param_groups for p in g["params"]}
    order = [id(p) for g in opt.param_groups for p in g["params"]]
    last_epoch, lr = rt.schedulers[0].last_epoch, rt.schedulers[0].get_last_lr()

    rt2 = build_runtime(recipe_from_dict(base_recipe))
    load_checkpoint(rt2, path)
    opt2 = next(iter(rt2.optimizers.values()))
    params2 = [p for g in opt2.param_groups for p in g["params"]]
    for pid, p2 in zip(order, params2):
        assert torch.equal(opt2.state[p2]["exp_avg"], saved[pid])
    assert rt2.schedulers[0].last_epoch == last_epoch == 2
    assert rt2.schedulers[0].get_last_lr() == lr
    assert rt2.state.step == 2
    for (n, p), (_, p2) in zip(rt.student_model.named_parameters(), rt2.student_model.named_parameters()):
        assert torch.equal(p, p2), n
    assert (path / "rng" / "rank0.pt").exists() and (path / "optim" / "default.pt").exists()
