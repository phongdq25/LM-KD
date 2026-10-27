"""Explicit source-trainer bridge for the extended PO family.

TIS-DPO weight estimation, TI-DPO attribution/anchor, TBPO-Q learned baselines,
and logits-SAM use the audited original HF trainer. This is a reference backend,
not a claim that those four methods have native LM-KD plugins. No loop is copied.
"""
import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig

from lmkd.compat.hf_api import dtype_kwarg
from lmkd.core.config import load_recipe
from lmkd.data.sources import JsonlSource


def _upstream(cfg):
    root = Path(cfg.compat.get("po_source_root", Path(__file__).resolve().parents[4] / "Awesome-Human-AI-Alignment-main"))
    source = (root / "src").resolve()
    if not (source / "human_alignment/integrations/preference.py").is_file():
        raise FileNotFoundError(f"Extended PO needs the audited source checkout at {root}; set compat.po_source_root")
    sys.path.insert(0, str(source))
    config_module = importlib.import_module("human_alignment.config")
    backend = importlib.import_module("human_alignment.integrations.preference")
    if backend.__file__ is None or not Path(backend.__file__).resolve().is_relative_to(source):
        raise ValueError("A different human_alignment package is already loaded; use a fresh reproduction process")
    return config_module.PreferenceOptimizationConfig, backend.PreferenceBackend


def _configuration(cfg, config_class):
    if cfg.compat.get("reproduction_backend") != "po_reference" or cfg.pipeline or cfg.loss.terms:
        raise ValueError("Extended PO bridge requires its dedicated reference recipe")
    if cfg.train.batch_size % cfg.train.micro_batch_size:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    if not isinstance(cfg.engine, (str, dict)) or (cfg.engine if isinstance(cfg.engine, str) else cfg.engine.get("type")) != "local":
        raise ValueError("Extended PO reproduction supports local execution only")
    cpu = isinstance(cfg.engine, dict) and cfg.engine.get("device") == "cpu"
    extra = {"max_steps": cfg.train.max_steps or -1, "logging_steps": cfg.train.log_every,
             "use_cpu": cpu, "bf16": not cpu and cfg.models.student.dtype == "bf16",
             "fp16": not cpu and cfg.models.student.dtype == "fp16", "save_strategy": "no",
             "weight_decay": cfg.optim.weight_decay, "max_grad_norm": cfg.grad.clip_norm or 0.0,
             "gradient_checkpointing": cfg.models.student.gradient_checkpointing,
             "lr_scheduler_type": "constant"}
    return config_class(output_dir=str(Path(cfg.train.output_dir) / "checkpoint-final/student"),
                        learning_rate=cfg.optim.lr, epochs=cfg.train.epochs or 1,
                        batch_size=cfg.train.micro_batch_size,
                        gradient_accumulation_steps=cfg.train.batch_size // cfg.train.micro_batch_size,
                        max_length=cfg.data.max_length, seed=cfg.train.seed, extra_args=extra,
                        resume_from_checkpoint=cfg.train.resume_from, **cfg.compat.get("po_options", {}))


def train_reference_po(cfg):
    """Delegate to the source backend, keeping attribution, optimizer and export semantics."""
    config_class, backend_class = _upstream(cfg)
    config = _configuration(cfg, config_class)
    source = cfg.data.train
    if not source or source.get("type") != "jsonl":
        raise ValueError("Extended PO requires a JSONL preference source")
    rows = list(JsonlSource(JsonlSource.Config(path=source["path"])).records())
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    spec = cfg.models.student
    options = {"tokenizer": spec.tokenizer or spec.path, "model_kwargs": dtype_kwarg(dtype[spec.dtype])}
    if cfg.models.teacher:
        options["reference_model"] = cfg.models.teacher.path
    if spec.lora:
        options["peft_config"] = LoraConfig(**spec.lora)
    options.update(cfg.compat.get("po_models", {}))
    return backend_class().train(method=cfg.compat["po_method"], model=spec.path,
                                 dataset=SimpleNamespace(data=rows), config=config, options=options)


def main(argv=None):
    """Execute one resolved reference-backend recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args(argv)
    train_reference_po(load_recipe(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
