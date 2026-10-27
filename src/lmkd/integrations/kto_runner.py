"""Run true unpaired KTO with the pinned public TRL trainer API.

This is the external-trainer exception from CONTRIBUTING.md §5.5. KTO owns its
mismatched-completion KL baseline, objective and optimizer; no native DPO loss
is substituted. Use the dedicated recipe through scripts/reproduce.sh kto.
"""
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer

from lmkd.compat.hf_api import dtype_kwarg
from lmkd.core.config import load_recipe
from lmkd.data.sources import JsonlSource


def kto_records(records):
    """Validate the unpaired {prompt, completion, label: bool} dataset contract."""
    rows = []
    for row in records:
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
            raise ValueError("KTO requires rendered string prompt/completion fields")
        if not row["prompt"] or not row["completion"] or not isinstance(row.get("label"), bool):
            raise ValueError("KTO needs nonempty prompt/completion and a boolean label")
        rows.append({key: row[key] for key in ("prompt", "completion", "label")})
    if not rows:
        raise ValueError("KTO requires nonempty training data")
    return rows


def _arguments(cfg):
    if cfg.compat.get("reproduction_backend") != "trl_kto":
        raise ValueError("KTO runner requires the dedicated kto recipe")
    if cfg.pipeline or cfg.loss.terms or cfg.data.transforms or cfg.eval:
        raise ValueError("KTO bridge consumes raw unpaired records; native pipeline/loss/eval are unsupported")
    if cfg.engine not in ("local", {"type": "local"}) and (
        not isinstance(cfg.engine, dict) or cfg.engine.get("type") != "local"
    ):
        raise ValueError("KTO reproduction currently supports the local engine")
    if cfg.train.batch_size % cfg.train.micro_batch_size:
        raise ValueError("KTO batch_size must be divisible by micro_batch_size")
    cpu = isinstance(cfg.engine, dict) and cfg.engine.get("device") == "cpu"
    return KTOConfig(
        output_dir=cfg.train.output_dir, beta=cfg.compat.get("kto_beta", 0.1),
        max_length=cfg.data.max_length, per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=cfg.train.batch_size // cfg.train.micro_batch_size,
        learning_rate=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
        max_grad_norm=cfg.grad.clip_norm or 0.0, max_steps=cfg.train.max_steps or -1,
        num_train_epochs=cfg.train.epochs or 1, seed=cfg.train.seed,
        logging_steps=cfg.train.log_every, report_to="none", use_cpu=cpu,
        bf16=not cpu and cfg.models.student.dtype == "bf16",
        fp16=not cpu and cfg.models.student.dtype == "fp16",
        lr_scheduler_type="constant", save_strategy="no",
        gradient_checkpointing=cfg.models.student.gradient_checkpointing,
    )


def train_kto(cfg):
    """Build the public KTO trainer and train/export without adding a third loop."""
    args = _arguments(cfg)
    source = cfg.data.train
    if not source or source.get("type") != "jsonl":
        raise ValueError("KTO reproduction requires a JSONL source")
    records = kto_records(JsonlSource(JsonlSource.Config(path=source["path"])).records())
    spec = cfg.models.student
    tok = AutoTokenizer.from_pretrained(spec.tokenizer or spec.path, trust_remote_code=spec.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(spec.path, **dtype_kwarg(dtype[spec.dtype]),
                                               trust_remote_code=spec.trust_remote_code)
    ref = None
    if cfg.models.teacher is not None:
        teacher = cfg.models.teacher
        ref = AutoModelForCausalLM.from_pretrained(teacher.path, **dtype_kwarg(dtype[teacher.dtype]),
                                                 trust_remote_code=teacher.trust_remote_code)
    trainer = KTOTrainer(model=model, ref_model=ref, args=args, train_dataset=Dataset.from_list(records),
                         processing_class=tok, peft_config=LoraConfig(**spec.lora) if spec.lora else None)
    trainer.train(resume_from_checkpoint=cfg.train.resume_from)
    destination = Path(cfg.train.output_dir) / "checkpoint-final/student"
    trainer.save_model(str(destination))
    tok.save_pretrained(destination)


def main(argv=None):
    """Execute the resolved KTO recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args(argv)
    train_kto(load_recipe(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
