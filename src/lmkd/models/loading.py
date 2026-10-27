"""Load student / teacher models and tokenizers from ModelSpec."""
from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
import torch.nn as nn

from lmkd.compat import hf_api
from lmkd.core.config import ModelSpec

logger = logging.getLogger(__name__)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.tokenizer or spec.path, trust_remote_code=spec.trust_remote_code)
    if tok.pad_token_id is None:
        # NOTE: pad may equal eos afterwards. Never infer lengths from `== pad_token_id` (DESIGN §4.2).
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    return tok


def _load_quantized_model(cls: Any, spec: ModelSpec) -> Any:
    """Load a frozen model with fp8 (torchao) or int8/nf4 (bitsandbytes)."""
    if spec.dtype == "fp8":
        if not torch.cuda.is_available():
            raise RuntimeError("fp8 quantization requires a CUDA GPU")
        cap = torch.cuda.get_device_capability()
        if cap < (8, 9):
            raise RuntimeError(f"fp8 quantization requires GPU sm89+ (got sm{cap[0]}{cap[1]})")
        from torchao.quantization import float8_weight_only, quantize_  # lazy (extra: torchao)

        kwargs = {**hf_api.dtype_kwarg(torch.bfloat16), "trust_remote_code": spec.trust_remote_code}
        if spec.attn_implementation:
            kwargs["attn_implementation"] = spec.attn_implementation
        model: Any = cls.from_pretrained(spec.path, **kwargs)
        quantize_(model, float8_weight_only())
        return model

    from transformers import BitsAndBytesConfig

    if spec.placement.startswith("device:"):
        device = spec.placement.split(":", 1)[1]
    elif spec.placement == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if spec.dtype == "int8":
        q_cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:  # nf4
        q_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

    kwargs = {
        "quantization_config": q_cfg,
        "device_map": {"": device},
        "trust_remote_code": spec.trust_remote_code,
    }
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation
    return cls.from_pretrained(spec.path, **kwargs)


def _check_tied_lora_targets(model: Any, lora_dict: dict[str, Any]) -> None:
    """Warn users when applying LoRA to tied word embeddings or lm_head."""
    tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    if not tied:
        return
    raw_targets = lora_dict.get("target_modules")
    if raw_targets is None:
        return
    targets: list[str] = [raw_targets] if isinstance(raw_targets, str) else list(raw_targets)
    tied_names = {"lm_head", "embed_tokens", "wte", "word_embeddings", "transformer.wte", "shared"}
    found = [t for t in targets if any(k in t for k in tied_names)]
    if found:
        msg = (
            f"Model has `tie_word_embeddings=True`, but LoRA target_modules includes tied modules {found}. "
            "Applying LoRA to lm_head or word embeddings with tied weights is rarely recommended: "
            "it can cause weight synchronization issues when saving/merging adapters (peft#2018), "
            "and forces SelfTeacher/TrustRegion to materialize full vocabulary logits instead of interpolating "
            "hidden states, resulting in significantly higher VRAM usage. "
            f"Recommendation: remove {found} from target_modules and apply LoRA only to attention and MLP layers."
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)


def load_model(spec: ModelSpec, *, trainable: bool) -> nn.Module:
    from transformers import AutoModel, AutoModelForCausalLM

    cls = AutoModelForCausalLM if spec.task == "causal_lm" else AutoModel
    model: Any

    if spec.dtype in ("fp8", "int8", "nf4"):
        if trainable:
            raise ValueError(f"dtype '{spec.dtype}' is only supported for frozen models (models.teacher or trainable=False)")
        model = _load_quantized_model(cls, spec)
    else:
        if spec.dtype not in _DTYPES:
            raise ValueError(f"unknown dtype '{spec.dtype}', supported: {sorted(_DTYPES) + ['fp8', 'int8', 'nf4']}")
        kwargs: dict[str, Any] = {**hf_api.dtype_kwarg(_DTYPES[spec.dtype]), "trust_remote_code": spec.trust_remote_code}
        if spec.attn_implementation:
            kwargs["attn_implementation"] = spec.attn_implementation
        model = cls.from_pretrained(spec.path, **kwargs)

    if spec.gradient_checkpointing and trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if spec.lora is not None:
        if not trainable:
            raise ValueError("lora on a frozen teacher: merge it offline first")
        from peft import LoraConfig, get_peft_model  # lazy (extra: peft)

        _check_tied_lora_targets(model, spec.lora)
        model = get_peft_model(model, LoraConfig(**spec.lora))
        if spec.gradient_checkpointing:
            model.enable_input_require_grads()
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model
