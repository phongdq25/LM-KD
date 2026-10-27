"""Offline teacher sentence embeddings pipeline stage (TALAS).

Source: embedding-distill/TALAS-M-main/tools/prepare_embedding.py:82-120
Paper: TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import STAGES
from lmkd.pipeline.io import load_samples_dir
from lmkd.pipeline.stage import Stage, StageContext


@STAGES.register(
    "embed_teacher",
    source="embedding-distill/TALAS-M-main/tools/prepare_embedding.py:82-120",
    paper="TALAS: Task-Agnostic Layer-Adaptive Self-Distillation for Sentence Embeddings",
)
class EmbedTeacher(Stage):
    """Offline teacher sentence embeddings.

    Loads texts from a dataset artifact, passes them through the teacher model,
    pools representations (cls, last, or mean), and saves:
      - embeddings.safetensors ('emb': [N, D_t] in fp32)
      - ids.json (ordered list of meta.id strings)
    """

    inputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    outputs: ClassVar[dict[str, str]] = {"emb": "embeddings"}

    class Config(BaseConfig):
        model: str
        pooling: str = "cls"  # cls | last (left padding) | mean
        max_len: int = 256
        batch_size: int = 64
        dtype: str = "fp16"
        normalize: bool = False
        instruction: str = ""
        text_field: str = "text"

    def run(self, ctx: StageContext) -> dict:
        from safetensors.torch import save_file

        samples = load_samples_dir(ctx.inputs["data"].path) if "data" in ctx.inputs else []
        ids, texts = self._ids_and_texts(samples)
        device = self._device(ctx)
        tokenizer, model = self._load_encoder(device)
        emb = self._encode(tokenizer, model, texts, device)
        if emb is None:
            emb = torch.empty((0, getattr(model.config, "hidden_size", 0)), dtype=torch.float32)

        assert ctx.output is not None
        out_path = Path(ctx.output.path)
        out_path.mkdir(parents=True, exist_ok=True)
        save_file({"emb": emb}, str(out_path / "embeddings.safetensors"))
        (out_path / "ids.json").write_text(json.dumps(ids))
        return {"dim": int(emb.shape[1]) if emb.numel() > 0 else 0, "n": len(ids),
                "model": self.cfg.model, "pooling": self.cfg.pooling}

    # ------------------------------------------------------------------ steps
    def _ids_and_texts(self, samples: list[dict]) -> tuple[list[str], list[str]]:
        """Ordered ``meta.id`` strings and (instruction-prefixed) texts of the dataset artifact."""
        ids, texts = [], []
        for idx, sample in enumerate(samples):
            ids.append(str(sample.get("meta.id", sample.get("id", idx))))
            txt = sample.get(self.cfg.text_field)
            if txt is None:
                txt = sample.get(f"meta.{self.cfg.text_field}")
            if txt is None:
                txt = sample.get("meta.text", sample.get("text", ""))
            texts.append(str(self.cfg.instruction + str(txt) if self.cfg.instruction else txt))
        return ids, texts

    @staticmethod
    def _device(ctx: StageContext) -> str:
        device = getattr(getattr(ctx.rt, "engine", None), "device", "cuda" if torch.cuda.is_available() else "cpu")
        return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else str(device)

    def _load_encoder(self, device: str):
        from transformers import AutoModel, AutoTokenizer

        from lmkd.compat import hf_api

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model, **({"padding_side": "left"} if self.cfg.pooling == "last" else {}))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.cfg.dtype, torch.float32)
        model = AutoModel.from_pretrained(self.cfg.model, **hf_api.dtype_kwarg(dtype))
        return tokenizer, model.to(device).eval()

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "last":     # left padding: the last position is the last real token
            return hidden[:, -1, :]
        if self.cfg.pooling == "cls":
            return hidden[:, 0, :]
        if self.cfg.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        raise ValueError(f"unknown pooling {self.cfg.pooling!r}")

    @torch.no_grad()
    def _encode(self, tokenizer, model, texts: list[str], device: str) -> torch.Tensor | None:
        """[N, D] fp32 CPU embeddings (None when there are no texts)."""
        bs = max(1, self.cfg.batch_size)
        vecs = []
        for i in range(0, len(texts), bs):
            enc = tokenizer(texts[i : i + bs], padding=True, truncation=True, max_length=self.cfg.max_len,
                            return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            v = self._pool(model(**enc).last_hidden_state, enc["attention_mask"])
            if self.cfg.normalize:
                v = F.normalize(v, p=2, dim=-1)
            vecs.append(v.to(torch.float32).cpu())
        return torch.cat(vecs, dim=0) if vecs else None
