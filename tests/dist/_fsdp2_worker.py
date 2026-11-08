"""FSDP2 worker: exercise the normal builder, step strategy and checkpoint API."""
import sys
from pathlib import Path

import torch

import lmkd
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.train.checkpointing import load_checkpoint, save_checkpoint


@CRITERIONS.register("fsdp_aux_test")
class AuxCriterion(Criterion):
    """Train an auxiliary head on final student hidden states."""

    def build_modules(self, info, modules):
        modules.add("aux.proj", torch.nn.Linear(info.student_hidden, 4))

    def features(self):
        return FeatureRequest(student=SideRequest(hidden="last"))

    def forward(self, ctx):
        values = ctx.modules.get("aux.proj")(ctx.student.last_hidden).square().mean(-1)
        return LossOutput({"aux": LossTerm((values * ctx.supervised_mask()).sum(), "token_sum")})


def main(model_path, data_path, output, optimizer):
    rt = lmkd.build({
        "engine": {"type": "fsdp2", "mixed_precision": "no"},
        "models": {"student": {"path": model_path, "dtype": "fp32"}},
        "data": {"train": {"type": "jsonl", "path": data_path},
                 "transforms": [{"type": "tokenize_sft", "use_chat_template": False}]},
        "loss": {"terms": [{"criterion": "nll"}, {"criterion": "fsdp_aux_test"}]},
        "optim": {"lr": 0.05 if optimizer == "sgd" else 0.001,
                  "optimizers": {"default": {"type": optimizer}, "aux": {"type": "adamw"}},
                  "groups": [{"name": "projector", "match": {"group": "projector"}, "optimizer": "aux"},
                             {"name": "default", "match": {}}]},
        "train": {"output_dir": output, "max_steps": 3, "batch_size": 4, "micro_batch_size": 1},
        "grad": {"clip_norm": None if optimizer == "sgd" else 1.0},
    })
    engine = rt.engine
    for module in rt.student_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    assert engine.world_size == 2
    expected_ids = {id(p.param) for p in rt.trainable_params}
    assert expected_ids == {id(p) for opt in rt.optimizers.values() for g in opt.param_groups for p in g["params"]}
    before = {n: p.to_local().detach().clone() for n, p in rt.modules.named_parameters()}
    rt.begin()
    rt.train_step()
    assert any(not torch.equal(before[n], p.to_local()) for n, p in rt.modules.named_parameters())
    student_state = dict(engine.iter_full_state_dict(rt.student_model))
    aux_state = dict(engine.iter_full_state_dict(rt.modules))
    if engine.is_main:
        torch.save({"student": student_state, "aux": aux_state}, Path(output) / "first_step.pt")
    checkpoint = save_checkpoint(rt, "probe")
    rt.train_step()
    expected = {n: p.to_local().detach().clone() for n, p in rt.student_module.named_parameters()}
    load_checkpoint(rt, checkpoint)
    rt.train_step()
    for name, param in rt.student_module.named_parameters():
        torch.testing.assert_close(param.to_local(), expected[name], rtol=0, atol=0)
    state = dict(engine.iter_full_state_dict(rt.student_model))
    if engine.is_main:
        assert state and all(v.device.type == "cpu" for v in state.values())
        (Path(output) / "verified").touch()
    engine.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(*sys.argv[1:])
