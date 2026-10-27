"""Offline generation and P-ALIGN verification stages."""
from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import torch

from lmkd.compat import vllm_api
from lmkd.core.batch import FIELDS, FieldSpec
from lmkd.core.config import ModelSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import GRADERS, ROLLOUT, STAGES
from lmkd.models.loading import load_model, load_tokenizer
from lmkd.pipeline.io import ResumableWriter, read_jsonl, shard_of, write_jsonl
from lmkd.pipeline.stage import Stage, StageContext
from lmkd.pipeline.stages.longcot import _chat_ids, _input_rows, _merge_jsonl, _model_path
from lmkd.rollout.base import SamplingSpec, sampling_from_cfg
from lmkd.rollout.vllm_colocate import make_llm

FIELDS.register(FieldSpec("meta.source_id", "meta"))

CONTINUE_TEMPLATE = (
    "Please continue from the draft and solve the problem step by step, "
    "and put your final answer within \\boxed{{}}. "
    "I will provide you with some prior knowledge as a draft to assist you in solving the question."
    "*Question*:{question}\n*Prefix*:{prefix}"
)
PALIGN_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def assemble_palign(prefix: str, continuation: str) -> str:
    """Join the teacher prefix and verified student continuation exactly as P-ALIGN.

    Source: SGL/src/sgl/palign/build.py:22-23.
    """
    return f"<Begin_of_Prefix>{prefix}<End_of_Prefix>\n{continuation.strip()}"


def _engine_name(value: dict | str) -> tuple[str, dict]:
    if isinstance(value, str):
        name, options = value, {}
    else:
        spec = dict(value)
        name, options = str(spec.pop("type")), spec
    if name not in ("hf", "vllm", "vllm_colocate"):
        raise ValueError(f"unknown generation engine {name!r}")
    return name, options


@STAGES.register("generate", source="SGL/src/sgl/palign/align.py:23-91")
class Generate(Stage):
    """Generate text continuations from tokenized prompts using HF or vLLM.

    Source: SGL palign/align.py:23-91 and P-ALIGN/src/prefix-alignment.py.
    """

    supports_shards = True
    outputs: ClassVar[dict[str, str]] = {"records": "text"}

    class Config(BaseConfig):
        model: str = ""
        prompt_template: str = "{prompt}"
        chat_template: bool = True
        enable_thinking: bool | None = None
        sampling: dict = {}
        output_field: str = "output"
        n: int = 1
        engine: dict | str = "vllm_colocate"
        input_kind: str = "records"

    def _prompts(self, rows: list[dict], tokenizer: Any) -> list[list[int]]:
        prompts = []
        for row in rows:
            values = dict(row)
            values.setdefault("prompt", row.get("meta.prompt_text", row.get("question", "")))
            text = self.cfg.prompt_template.format_map(values)
            ids = (_chat_ids(tokenizer, text, self.cfg.enable_thinking) if self.cfg.chat_template
                   else list(tokenizer(text, add_special_tokens=False)["input_ids"]))
            prompts.append(ids)
        return prompts

    def _generate_hf(self, prompts: list[list[int]], spec: SamplingSpec,
                     tokenizer: Any, options: dict) -> list[list[str]]:
        dtype = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else ("fp16" if torch.cuda.is_available() else "fp32")
        model = load_model(ModelSpec(path=self._resolved_model, dtype=dtype), trainable=False)
        if torch.cuda.is_available():
            model = model.to("cuda")
        runtime = SimpleNamespace(student_model=model, student_tokenizer=tokenizer)
        engine = ROLLOUT.build({"type": "hf", **options})
        engine.setup(runtime)
        try:
            return [[r.text for r in group] for group in engine.generate(prompts, spec)]
        finally:
            del model, engine

    def _generate_vllm(self, prompts: list[list[int]], spec: SamplingSpec,
                       options: dict) -> list[list[str]]:
        allowed = {"gpu_memory_utilization", "max_model_len", "dtype", "enable_prefix_caching",
                   "enforce_eager", "tensor_parallel_size", "seed"}
        kwargs = {key: value for key, value in options.items() if key in allowed}
        llm = make_llm(self._resolved_model, **kwargs)
        params = vllm_api.sampling_params(n=spec.n, temperature=spec.temperature, top_p=spec.top_p,
                                          top_k=spec.top_k, repetition_penalty=spec.repetition_penalty,
                                          max_tokens=spec.max_new_tokens, stop=spec.stop or None,
                                          stop_token_ids=spec.stop_token_ids or None, seed=spec.seed)
        try:
            outputs = llm.generate([vllm_api.tokens_prompt(p) for p in prompts], params, use_tqdm=False)
            return [[choice.text for choice in out.outputs] for out in outputs]
        finally:
            del llm
            gc.collect()

    def run(self, ctx: StageContext) -> dict:
        if self.cfg.input_kind not in ("records", "dataset"):
            raise ValueError(f"unknown input_kind {self.cfg.input_kind}")
        rows = _input_rows(ctx, "records", "data", "prev")
        self._resolved_model = _model_path(self.cfg.model, ctx)
        assert ctx.output is not None
        path = ctx.output.path / "records.jsonl"
        shard_rows = shard_of(rows, *ctx.shard)
        tokenizer = load_tokenizer(ModelSpec(path=self._resolved_model))
        spec = sampling_from_cfg({**self.cfg.sampling, "n": self.cfg.n, "return_logprobs": False})
        name, options = _engine_name(self.cfg.engine)
        with ResumableWriter(path) as writer:
            pending = [r for r in shard_rows if not all(writer.done(
                str(r["meta.id"]) if self.cfg.n == 1 else f"{r['meta.id']}#{j}")
                for j in range(self.cfg.n))]
            if pending:
                prompts = self._prompts(pending, tokenizer)
                generated = (self._generate_hf(prompts, spec, tokenizer, options) if name == "hf"
                             else self._generate_vllm(prompts, spec, options))
                for row, completions in zip(pending, generated):
                    for index, completion in enumerate(completions):
                        sid = str(row["meta.id"]) if self.cfg.n == 1 else f"{row['meta.id']}#{index}"
                        if not writer.done(sid):
                            writer.write({**row, "meta.id": sid, "meta.source_id": str(row["meta.id"]),
                                          self.cfg.output_field: completion})
        return {"n": len(shard_rows) * self.cfg.n}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        # n>1 produces several ids per source id, so preserve source order and candidate index.
        source = _input_rows(ctx, "records", "data", "prev")
        if self.cfg.n == 1:
            rows = _merge_jsonl(ctx, shard_dirs, "records.jsonl", source)
        else:
            by_id = {str(r["meta.id"]): r for folder in shard_dirs for r in read_jsonl(folder / "records.jsonl")}
            ids = [f"{r['meta.id']}#{j}" for r in source for j in range(self.cfg.n)]
            rows = [by_id[sid] for sid in ids if sid in by_id]
            assert ctx.output is not None
            write_jsonl(ctx.output.path / "records.jsonl", rows)
        return {"n": len(rows)}


@STAGES.register("verify_filter", source="SGL/src/sgl/palign/build.py:22-62")
class VerifyFilter(Stage):
    """Keep mathematically correct continuations and optionally assemble P-ALIGN text.

    Source: SGL palign/build.py:22-62.
    """

    supports_shards = True
    outputs: ClassVar[dict[str, str]] = {"records": "text"}

    class Config(BaseConfig):
        grader: str = "math_verify"
        pred_field: str = "output"
        gold_field: str = "answer"
        assemble: str | None = "palign"
        keep_unverified: bool = False

    def _filter(self, rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
        grader = GRADERS.build(self.cfg.grader)
        kept, dropped = [], {"no_gold": 0, "wrong": 0}
        for row in rows:
            gold = str(row.get(self.cfg.gold_field, ""))
            if not self.cfg.keep_unverified:
                if not gold:
                    dropped["no_gold"] += 1
                    continue
                if not grader(str(row[self.cfg.pred_field]), gold):
                    dropped["wrong"] += 1
                    continue
            output = dict(row)
            if self.cfg.assemble == "palign":
                output["instruction"] = PALIGN_INSTRUCTION
                output["input"] = str(row.get("question", ""))
                output["output"] = assemble_palign(str(row["prefix"]), str(row[self.cfg.pred_field]))
            elif self.cfg.assemble is not None:
                raise ValueError(f"unknown assembly {self.cfg.assemble}")
            kept.append(output)
        return kept, dropped

    def run(self, ctx: StageContext) -> dict:
        rows = shard_of(_input_rows(ctx, "records", "data", "prev"), *ctx.shard)
        kept, dropped = self._filter(rows)
        assert ctx.output is not None
        path = ctx.output.path / "records.jsonl"
        with ResumableWriter(path) as writer:
            for row in kept:
                if not writer.done(str(row["meta.id"])):
                    writer.write(row)
        return {"kept": len(kept), "dropped": dropped}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = _merge_jsonl(ctx, shard_dirs, "records.jsonl", _input_rows(ctx, "records", "data", "prev"))
        dropped = {"no_gold": 0, "wrong": 0}
        for folder in shard_dirs:
            import json

            info = json.loads((folder / "manifest.json").read_text()).get("info", {})
            for key in dropped:
                dropped[key] += int(info.get("dropped", {}).get(key, 0))
        return {"kept": len(rows), "dropped": dropped}
