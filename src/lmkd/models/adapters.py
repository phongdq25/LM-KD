"""ModelAdapter: the single place that knows architecture-specific attribute paths (DESIGN §5.1).

Layer index convention (same as HF ``output_hidden_states``):
  hidden(0) = embedding output, hidden(i) = output of block i (1..L). Negative indices allowed.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

from lmkd.compat import hf_api
from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import ADAPTERS


def _getattr_path(obj: Any, path: str) -> Any:
    cur = obj
    for p in path.split("."):
        cur = getattr(cur, p, None)
        if cur is None:
            return None
    return cur


def unwrap(model: nn.Module) -> nn.Module:
    """Strip DDP/DeepSpeed (`.module`) wrappers. PEFT models are kept (they forward correctly)."""
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model


def get_base_head(head: nn.Module | None) -> nn.Module | None:
    """Extract the underlying frozen base module from an adapted (LoRA) head."""
    if head is None:
        return None
    curr: Any = head
    while hasattr(curr, "base_layer") or hasattr(curr, "original_module"):
        if hasattr(curr, "base_layer"):
            curr = curr.base_layer
        elif hasattr(curr, "original_module"):
            curr = curr.original_module
    return curr


def _base(model: nn.Module) -> Any:   # Any: HF model classes differ (get_input_embeddings, config, ...)
    """PEFT -> underlying HF model."""
    m: Any = unwrap(model)   # Any: PEFT / HF classes expose different methods
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m


@ADAPTERS.register("hf_auto")
class HFAutoAdapter(Plugin):
    """Works for GPT-2, OPT, LLaMA/Qwen/Mistral, GPT-NeoX, BERT-like encoders."""

    class Config(BaseConfig):
        pass

    LAYER_PATHS = ("model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers",
                   "encoder.layer", "bert.encoder.layer", "layers", "h")
    BACKBONE_PATHS = ("model", "transformer", "gpt_neox", "bert", "")
    HEAD_PATHS = ("lm_head", "embed_out", "cls.predictions.decoder")

    def layers(self, model: nn.Module) -> nn.ModuleList:
        m = _base(model)
        for p in self.LAYER_PATHS:
            v = _getattr_path(m, p)
            if isinstance(v, nn.ModuleList):
                return v
        raise RuntimeError(f"cannot find transformer layers in {type(m).__name__}")

    def num_layers(self, model: nn.Module) -> int:
        return len(self.layers(model))

    def hidden_size(self, model: nn.Module) -> int:
        cfg = _base(model).config
        for k in ("hidden_size", "n_embd", "d_model", "word_embed_proj_dim"):
            if hasattr(cfg, k):
                return int(getattr(cfg, k))
        raise RuntimeError("cannot infer hidden size")

    def vocab_size(self, model: nn.Module) -> int:
        return int(self.embed_tokens(model).weight.shape[0])

    def embed_tokens(self, model: nn.Module) -> nn.Embedding:
        return _base(model).get_input_embeddings()

    def lm_head(self, model: nn.Module) -> nn.Module | None:
        m = _base(model)
        out = m.get_output_embeddings() if hasattr(m, "get_output_embeddings") else None
        if out is not None:
            return out
        for p in self.HEAD_PATHS:
            v = _getattr_path(m, p)
            if isinstance(v, nn.Module):
                return v
        return None

    def backbone(self, model: nn.Module) -> nn.Module:
        """Module returning ``last_hidden_state`` already passed through the final norm
        (true for HF LLaMA/Qwen `model.model`, GPT-2 `transformer`, OPT `model.decoder` via `model`).
        Calling the backbone instead of the full LM avoids materializing [B,T,V] logits."""
        m = _base(model)
        if self.lm_head(model) is None:  # encoder without head: the model itself is the backbone
            return m
        for p in self.BACKBONE_PATHS:
            if not p:
                continue
            v = _getattr_path(m, p)
            if isinstance(v, nn.Module):
                return v
        raise RuntimeError(f"cannot find backbone of {type(m).__name__}")

    @contextlib.contextmanager
    def adapters_disabled(self, model: nn.Module) -> Iterator[None]:
        """Run the base model with LoRA adapters off (zero-memory 'frozen base' teacher)."""
        m: Any = unwrap(model)
        if hasattr(m, "disable_adapter"):
            with m.disable_adapter():
                yield
        else:
            yield

    @torch.no_grad()
    def last_row_attention(self, model, layer, hidden_in, attention_mask_2d, position_ids):
        """Sum heads of softmax(q_last K.T), detached (SRA sra_span_utils.py:256).

        Supports GPT-2 and LLaMA/Qwen2/Qwen3/Mistral without full attention matrices.
        """
        import math
        m = _base(model)
        block = self.layers(model)[layer - 1]
        B, T, _ = hidden_in.shape
        kind = m.config.model_type
        if attention_mask_2d is not None:
            mask = (attention_mask_2d != 0)
            positions = torch.arange(T, device=hidden_in.device)
            valid_positions = torch.where(mask, positions, -1)
            last_idx = valid_positions.max(dim=-1).values.clamp_min(0)
        else:
            last_idx = torch.full((B,), T - 1, device=hidden_in.device, dtype=torch.long)
        if kind == "gpt2":
            att = block.attn
            x = block.ln_1(hidden_in)
            q, k, _ = att.c_attn(x).split(att.split_size, dim=2)
            heads, hd = att.num_heads, att.head_dim
            q = q.view(B, T, heads, hd).transpose(1, 2)
            k = k.view(B, T, heads, hd).transpose(1, 2)
            q_last = q.gather(2, last_idx.view(B, 1, 1, 1).expand(B, heads, 1, hd))
            scores = q_last.float() @ k.float().transpose(-1, -2)
            if att.scale_attn_weights:
                scores /= math.sqrt(hd)
            if att.scale_attn_by_inverse_layer_idx:
                scores /= att.layer_idx + 1
        elif kind in hf_api.ROTARY_MODEL_TYPES:
            att = block.self_attn
            x = block.input_layernorm(hidden_in)
            hd = att.head_dim
            q = att.q_proj(x).view(B, T, -1, hd)
            k = att.k_proj(x).view(B, T, -1, hd)
            if hasattr(att, "q_norm"):
                q, k = att.q_norm(q), att.k_norm(k)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            if position_ids is None:
                position_ids = torch.arange(T, device=x.device).unsqueeze(0)
            cos, sin = self.backbone(model).rotary_emb(x, position_ids)  # type: ignore[operator]  # nn.Module attr
            apply_rotary = hf_api.rotary_apply_fn(att)
            q, k = apply_rotary(q, k, cos, sin)
            k = k.repeat_interleave(att.num_key_value_groups, dim=1)
            q_last = q.gather(2, last_idx.view(B, 1, 1, 1).expand(B, q.shape[1], 1, hd))
            scores = (q_last.float() @ k.float().transpose(-1, -2)) * att.scaling
            sliding = getattr(att, "sliding_window", None)
            if sliding is not None:
                col_pos = torch.arange(T, device=x.device).view(1, 1, 1, T)
                scores = scores.masked_fill(col_pos <= (last_idx.view(B, 1, 1, 1) - sliding), torch.finfo(scores.dtype).min)
        else:
            raise NotImplementedError(f"last_row_attention unsupported architecture: {kind}")
        col_pos = torch.arange(T, device=hidden_in.device).view(1, 1, 1, T)
        scores = scores.masked_fill(col_pos > last_idx.view(B, 1, 1, 1), torch.finfo(scores.dtype).min)
        if attention_mask_2d is not None:
            scores = scores.masked_fill(attention_mask_2d[:, None, None, :] == 0, torch.finfo(scores.dtype).min)
        return scores.softmax(-1).to(q.dtype).sum(1).squeeze(1)

    def forward_hidden(
        self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None = None, output_hidden_states: bool = False,
        output_attentions: bool = False, inputs_embeds: torch.Tensor | None = None,
    ) -> Any:
        """Run the backbone. Returns the HF BaseModelOutput (last_hidden_state, hidden_states...)."""
        bb = self.backbone(model)
        kwargs: dict[str, Any] = {"attention_mask": attention_mask, "output_hidden_states": output_hidden_states,
                                      "output_attentions": output_attentions, "return_dict": True}
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if self.lm_head(model) is not None:
            kwargs["use_cache"] = False
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
        else:
            kwargs["input_ids"] = input_ids
        return bb(**kwargs)


class HiddenForward(nn.Module):
    """Trainable forward module: ``forward(**kw)`` runs ``adapter.forward_hidden(model, ...)``.

    The engine wraps THIS module, so the wrapper's ``forward`` is always on the call
    path (gradient sync hooks fire) while we still avoid computing full-vocab logits (K1).
    ``rt.student_model`` stays the raw HF/PEFT model (save_pretrained, adapter queries, lm_head).
    """

    def __init__(self, model: nn.Module, adapter: HFAutoAdapter):
        super().__init__()
        self.model = model
        self._adapter = adapter  # plain object, not a submodule

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, output_hidden_states=False,
                output_attentions=False, inputs_embeds=None):
        return self._adapter.forward_hidden(
            self.model, input_ids, attention_mask, position_ids, output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, inputs_embeds=inputs_embeds,
        )


def resolve_adapter(name: str | None) -> HFAutoAdapter:
    return ADAPTERS.build(name or "hf_auto")
