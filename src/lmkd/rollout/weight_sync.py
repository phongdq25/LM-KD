"""Student -> rollout-engine weight transfer (T08).

``iter_student_weights``: stream ``(hf_name, tensor)`` for vLLM ``load_weights`` (full fine-tune, or LoRA
merged on the fly). ``save_lora_adapter``: LoRA hot-load path (adapter written to disk, no tensors moved).
Source: Self-Distill/OPSD-main/opsd_trainer.py:1175-1245.
"""
from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from lmkd.models.adapters import unwrap


def clean_peft_name(name: str, prefix: str = "lora_") -> str | None:
    """PEFT parameter name -> plain HF name of the merged base model; ``None`` = skip (adapter weight)."""
    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
    if prefix in name or "original_module" in name:
        return None
    return name.replace("modules_to_save.default.", "")


def _is_peft(m: torch.nn.Module) -> bool:
    return hasattr(m, "merge_adapter") and hasattr(m, "peft_config")


def iter_student_weights(rt, mode: str) -> Iterator[tuple[str, torch.Tensor]]:
    """``mode``: ``full`` (no PEFT) | ``merge`` (LoRA merged for the duration of the iteration).

    Always reads ``rt.student_model`` (raw model), never ``rt.student_module`` (wrapper prefixes names).
    """
    model = rt.student_model
    engine = rt.engine
    if mode == "merge":
        m: Any = unwrap(model)   # PEFT model: merge_adapter / unmerge_adapter
        if getattr(engine, "is_partitioned", False):
            raise NotImplementedError("lora_mode='merge' with ZeRO-3 (sharded parameters) is not supported; "
                                      "use lora_mode='hot_load' or ZeRO-1/2")
        if not _is_peft(m):
            raise ValueError("lora_mode='merge' needs a PEFT student (models.student.lora)")
        m.merge_adapter()
        try:
            prefix = getattr(m, "prefix", "lora_")
            for raw_name, p in m.named_parameters():
                name = clean_peft_name(raw_name, prefix)
                if name is not None:
                    yield name, p.data
        finally:
            m.unmerge_adapter()                      # never leave the training model merged
        return
    if mode != "full":
        raise ValueError(f"unknown weight sync mode '{mode}' (full | merge)")
    if getattr(engine, "is_partitioned", False):     # ZeRO-3: parameters are sharded
        if not hasattr(engine, "iter_full_state_dict"):
            raise NotImplementedError("needs T05 iter_full_state_dict")
        yield from engine.iter_full_state_dict(model)
        return
    for n, p in unwrap(model).named_parameters():
        yield n, p.data


def save_lora_adapter(rt, root: Path, step: int, keep: int = 2) -> Path:
    """Write the current adapter to ``root/lora_<step>`` and drop all but the newest ``keep`` dirs."""
    root = Path(root)
    path = root / f"lora_{step}"
    adapter_model: Any = unwrap(rt.student_model)
    adapter_model.save_pretrained(str(path))
    old = sorted((d for d in root.glob("lora_*") if d != path), key=lambda d: d.stat().st_mtime)
    for d in old[: max(0, len(old) - (keep - 1))]:
        shutil.rmtree(d, ignore_errors=True)
    return path
