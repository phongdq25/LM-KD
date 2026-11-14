"""P-ALIGN pipeline stages: search, generate, verify_filter, and closed-loop PAlignStage."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import torch

from baselines.longCoT.common import (
    chat_ids,
    engine_name,
    input_rows,
    merge_jsonl,
    model_path,
    pending,
    resolve_source_rows,
)
from baselines.longCoT.palign.search import (
    CONTINUE_TEMPLATE,
    JUDGE_TEMPLATE,
    PALIGN_INSTRUCTION,
    PrefixSearch,
    assemble_palign,
    run_searches,
)
from baselines.longCoT.segmenters import SEGMENTERS
from lmkd.compat import vllm_api
from lmkd.core.batch import FIELDS, FieldSpec
from lmkd.core.config import ModelSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import GRADERS, ROLLOUT, STAGES, TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.models.loading import load_model, load_tokenizer
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import ResumableWriter, load_samples_dir, read_jsonl, save_samples, shard_of, write_jsonl
from lmkd.pipeline.stage import Stage, StageContext
from lmkd.rollout.base import SamplingSpec, sampling_from_cfg
from lmkd.rollout.vllm_colocate import make_llm

FIELDS.register(FieldSpec("meta.source_id", "meta"))


@STAGES.register("palign_search")
class PAlignSearch(Stage):
    """Find the shortest useful CoT prefix with a lockstep student judge.

    """

    supports_shards = True
    outputs: ClassVar[dict[str, str]] = {"records": "text"}

    class Config(BaseConfig):
        model: str = ""
        judge_template: str = JUDGE_TEMPLATE
        question_field: str = "question"
        cot_field: str = "deepseek_thinking_trajectory"
        batch_size: int = 512
        max_new_tokens: int = 256
        engine: str = "vllm"
        max_rounds: int = 20

    def _judge(self, prompts: list[str], model: Any, tokenizer: Any) -> list[str]:
        ids = [chat_ids(tokenizer, prompt) for prompt in prompts]
        if self.cfg.engine == "vllm":
            params = vllm_api.sampling_params(max_tokens=self.cfg.max_new_tokens, temperature=0.0)
            outputs = model.generate([vllm_api.tokens_prompt(p) for p in ids], params, use_tqdm=False)
            return [out.outputs[0].text for out in outputs]
        if self.cfg.engine != "hf":
            raise ValueError(f"unknown judge engine {self.cfg.engine}")
        device = next(model.parameters()).device
        width = max(map(len, ids))
        padded = torch.full((len(ids), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(padded)
        for index, prompt in enumerate(ids):
            padded[index, -len(prompt) :] = torch.tensor(prompt, dtype=torch.long, device=device)
            mask[index, -len(prompt) :] = 1
        with torch.no_grad():
            generated = model.generate(
                padded,
                attention_mask=mask,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        return [tokenizer.decode(row[width:], skip_special_tokens=True) for row in generated]

    def search_rows(self, rows: list[dict], resolved_model: str) -> list[dict]:
        """Perform prefix search on in-memory rows using the configured judge model."""
        self._resolved_model = resolved_model
        tokenizer = load_tokenizer(ModelSpec(path=resolved_model))
        segmenter = SEGMENTERS.build("palign_sentences")
        searches = [
            PrefixSearch(
                str(row.get(self.cfg.question_field, row.get("question", row.get("problem", "")))),
                segmenter(
                    str(row.get(self.cfg.cot_field, row.get("response", row.get("solution", row.get("cot", "")))))
                ),
            )
            for row in rows
        ]
        model = (
            make_llm(resolved_model, enable_prefix_caching=True, enforce_eager=True)
            if self.cfg.engine == "vllm"
            else load_model(ModelSpec(path=resolved_model), trainable=False)
        )
        if self.cfg.engine == "hf" and torch.cuda.is_available():
            model = model.to("cuda")
        judge_model = model
        judge_tok = tokenizer
        try:
            def judge_fn(p: list[str]) -> list[str]:
                return self._judge(p, judge_model, judge_tok)

            run_searches(
                searches,
                judge_fn,
                self.cfg.batch_size,
                self.cfg.max_rounds,
                self.cfg.judge_template,
            )
        finally:
            del model
            gc.collect()

        results = []
        for row, search in zip(rows, searches):
            prefix, n_prefix, sufficient = search.result()
            results.append(
                {
                    **row,
                    "meta.id": str(row["meta.id"]),
                    "question": search.question,
                    "answer": row.get("answer", row.get("meta.answer", "")),
                    "prefix": prefix,
                    "n_prefix": n_prefix,
                    "n_sentences": len(search.sentences),
                    "sufficient": sufficient,
                    "rounds": search.rounds,
                }
            )
        return results

    def run(self, ctx: StageContext) -> dict:
        source = input_rows(ctx, "data", "prev")
        path, rows = pending(ctx, "records.jsonl", source)
        self._resolved_model = model_path(self.cfg.model, ctx)
        with ResumableWriter(path) as writer:
            pending_rows = [row for row in rows if not writer.done(str(row["meta.id"]))]
            if pending_rows:
                searched = self.search_rows(pending_rows, self._resolved_model)
                for rec in searched:
                    writer.write(rec)
        return {"n": len(rows)}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = merge_jsonl(ctx, shard_dirs, "records.jsonl", input_rows(ctx, "data", "prev"))
        return {"n": len(rows)}


@STAGES.register("generate")
class Generate(Stage):
    """Generate text continuations from tokenized prompts using HF or vLLM.

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
            question = str(row.get("question", row.get("problem", "")))
            values.setdefault("question", question)
            values.setdefault("prefix", str(row.get("prefix", "")))
            values.setdefault("prompt", row.get("meta.prompt_text", question))
            text = self.cfg.prompt_template.format_map(values)
            ids = (
                chat_ids(tokenizer, text, self.cfg.enable_thinking)
                if self.cfg.chat_template
                else list(tokenizer(text, add_special_tokens=False)["input_ids"])
            )
            prompts.append(ids)
        return prompts

    def _generate_hf(
        self, prompts: list[list[int]], spec: SamplingSpec, tokenizer: Any, options: dict
    ) -> list[list[str]]:
        dtype = (
            "bf16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else ("fp16" if torch.cuda.is_available() else "fp32")
        )
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

    def _generate_vllm(self, prompts: list[list[int]], spec: SamplingSpec, options: dict) -> list[list[str]]:
        allowed = {
            "gpu_memory_utilization",
            "max_model_len",
            "dtype",
            "enable_prefix_caching",
            "enforce_eager",
            "tensor_parallel_size",
            "seed",
        }
        kwargs = {k: v for k, v in options.items() if k in allowed}
        llm = make_llm(self._resolved_model, **kwargs)
        params = vllm_api.sampling_params(
            n=spec.n,
            temperature=spec.temperature,
            top_p=spec.top_p,
            top_k=spec.top_k,
            repetition_penalty=spec.repetition_penalty,
            max_tokens=spec.max_new_tokens,
            stop=spec.stop or None,
            stop_token_ids=spec.stop_token_ids or None,
            seed=spec.seed,
        )
        try:
            outputs = llm.generate([vllm_api.tokens_prompt(p) for p in prompts], params, use_tqdm=False)
            return [[out.text for out in item.outputs] for item in outputs]
        finally:
            del llm

    def generate_rows(self, rows: list[dict], resolved_model: str) -> list[dict]:
        """Generate continuations for in-memory rows using the configured rollout engine."""
        self._resolved_model = resolved_model
        tokenizer = load_tokenizer(ModelSpec(path=self._resolved_model))
        spec = sampling_from_cfg({**self.cfg.sampling, "n": self.cfg.n, "return_logprobs": False})
        name, options = engine_name(self.cfg.engine)
        prompts = self._prompts(rows, tokenizer)
        generated = (
            self._generate_hf(prompts, spec, tokenizer, options)
            if name == "hf"
            else self._generate_vllm(prompts, spec, options)
        )
        results = []
        for row, completions in zip(rows, generated):
            for index, completion in enumerate(completions):
                sid = str(row["meta.id"]) if self.cfg.n == 1 else f"{row['meta.id']}#{index}"
                results.append(
                    {
                        **row,
                        "meta.id": sid,
                        "meta.source_id": str(row["meta.id"]),
                        self.cfg.output_field: completion,
                    }
                )
        return results

    def run(self, ctx: StageContext) -> dict:
        if self.cfg.input_kind not in ("records", "dataset"):
            raise ValueError(f"unknown input_kind {self.cfg.input_kind}")
        rows = input_rows(ctx, "records", "data", "prev")
        self._resolved_model = model_path(self.cfg.model, ctx)
        assert ctx.output is not None
        path = ctx.output.path / "records.jsonl"
        shard_rows = shard_of(rows, *ctx.shard)
        with ResumableWriter(path) as writer:
            pending_rows = [
                r
                for r in shard_rows
                if not all(
                    writer.done(str(r["meta.id"]) if self.cfg.n == 1 else f"{r['meta.id']}#{j}")
                    for j in range(self.cfg.n)
                )
            ]
            if pending_rows:
                generated = self.generate_rows(pending_rows, self._resolved_model)
                for rec in generated:
                    if not writer.done(str(rec["meta.id"])):
                        writer.write(rec)
        return {"n": len(shard_rows) * self.cfg.n}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        source = input_rows(ctx, "records", "data", "prev")
        if self.cfg.n == 1:
            rows = merge_jsonl(ctx, shard_dirs, "records.jsonl", source)
        else:
            by_id = {str(r["meta.id"]): r for folder in shard_dirs for r in read_jsonl(folder / "records.jsonl")}
            ids = [f"{r['meta.id']}#{j}" for r in source for j in range(self.cfg.n)]
            rows = [by_id[sid] for sid in ids if sid in by_id]
            assert ctx.output is not None
            write_jsonl(ctx.output.path / "records.jsonl", rows)
        return {"n": len(rows)}


@STAGES.register("verify_filter")
class VerifyFilter(Stage):
    """Keep mathematically correct continuations and optionally assemble P-ALIGN text.

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
            gold = str(row.get(self.cfg.gold_field, row.get("meta.answer", "")))
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
                q = str(row.get("question", row.get("problem", "")))
                output["input"] = q
                output["problem"] = q
                palign_out = (
                    str(row["output"])
                    if ("<Begin_of_Prefix>" in str(row.get("output", "")))
                    else assemble_palign(str(row["prefix"]), str(row[self.cfg.pred_field]))
                )
                output["output"] = palign_out
                output["response"] = palign_out
            elif self.cfg.assemble is not None:
                raise ValueError(f"unknown assembly {self.cfg.assemble}")
            kept.append(output)
        return kept, dropped

    def run(self, ctx: StageContext) -> dict:
        rows = shard_of(input_rows(ctx, "records", "data", "prev"), *ctx.shard)
        kept, dropped = self._filter(rows)
        assert ctx.output is not None
        path = ctx.output.path / "records.jsonl"
        with ResumableWriter(path) as writer:
            for row in kept:
                if not writer.done(str(row["meta.id"])):
                    writer.write(row)
        return {"kept": len(kept), "dropped": dropped}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        rows = merge_jsonl(ctx, shard_dirs, "records.jsonl", input_rows(ctx, "records", "data", "prev"))
        dropped = {"no_gold": 0, "wrong": 0}
        for folder in shard_dirs:
            info = json.loads((folder / "manifest.json").read_text()).get("info", {})
            for key in dropped:
                dropped[key] += int(info.get("dropped", {}).get(key, 0))
        return {"kept": len(rows), "dropped": dropped}


@STAGES.register("palign")
class PAlignStage(Stage):
    """Self-contained end-to-end Prefix-Search Alignment (P-ALIGN) stage.

    Composes PAlignSearch -> Generate -> VerifyFilter -> Tokenize in an integrated flow.
    """

    inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {"data": "dataset"}
    supports_shards = True

    class Config(BaseConfig):
        source: dict | str | None = None
        model: str = ""
        engine: str = "vllm"
        generation_engine: str | None = None
        question_field: str = "question"
        cot_field: str = "deepseek_thinking_trajectory"
        max_rounds: int = 10
        batch_size: int = 64
        max_new_tokens: int = 256
        judge_template: str = JUDGE_TEMPLATE
        prompt_template: str = CONTINUE_TEMPLATE
        grader: str = "math_verify"
        keep_unverified: bool = False
        temperature: float = 0.6
        top_p: float = 0.9
        repetition_penalty: float = 1.05
        max_generate_tokens: int = 32768
        chat_template: bool = True
        tokenize_verify: bool = True
        n: int = 1

    @classmethod
    def provided_keys(cls, cfg: Any) -> tuple[str, ...]:
        return ("student.input_ids", "seq.loss_mask", "seq.response_span")

    def run(self, ctx: StageContext) -> dict:
        source_rows = resolve_source_rows(ctx, self.cfg.source)
        self._resolved_model = model_path(self.cfg.model, ctx)
        tokenizer = load_tokenizer(ModelSpec(path=self._resolved_model))

        assert ctx.output is not None
        samples_path = ctx.output.path / "samples.jsonl"
        records_path = ctx.output.path / "records.jsonl"
        shard_rows = shard_of(source_rows, *ctx.shard)

        data_ctx = getattr(ctx.rt, "data_ctx", None) or DataContext(
            student_tokenizer=tokenizer,
            max_length=getattr(getattr(ctx.rt, "config", None), "data", None)
            and ctx.rt.config.data.max_length
            or 32768,
        )
        render_tf = TRANSFORMS.build({"type": "render_cot", "chat_template": self.cfg.chat_template})
        tokenize_tf = TRANSFORMS.build({"type": "tokenize_cot", "verify": self.cfg.tokenize_verify})
        if hasattr(render_tf, "setup"):
            render_tf.setup(ctx.rt)
        if hasattr(tokenize_tf, "setup"):
            tokenize_tf.setup(ctx.rt)

        with ResumableWriter(samples_path) as s_writer, ResumableWriter(records_path) as r_writer:
            pending_rows = [row for row in shard_rows if not s_writer.done(str(row["meta.id"]))]
            if pending_rows:
                # Flow step 1: Search prefix (delegate to PAlignSearch)
                needed_search = [r for r in pending_rows if "prefix" not in r]
                if needed_search:
                    searcher = PAlignSearch(
                        PAlignSearch.Config(
                            model=self.cfg.model,
                            engine=self.cfg.engine,
                            question_field=self.cfg.question_field,
                            cot_field=self.cfg.cot_field,
                            max_rounds=self.cfg.max_rounds,
                            batch_size=self.cfg.batch_size,
                            max_new_tokens=self.cfg.max_new_tokens,
                            judge_template=self.cfg.judge_template,
                        )
                    )
                    searched = searcher.search_rows(needed_search, self._resolved_model)
                    search_map = {str(r["meta.id"]): r for r in searched}
                    for r in pending_rows:
                        if str(r["meta.id"]) in search_map:
                            res = search_map[str(r["meta.id"])]
                            r["prefix"] = res["prefix"]
                            r["n_prefix"] = res["n_prefix"]
                            r["sufficient"] = res["sufficient"]

                # Flow step 2: Generate continuation (delegate to Generate)
                needed_gen = [
                    r for r in pending_rows if not (r.get("output") and "<Begin_of_Prefix>" in str(r.get("output")))
                ]
                gen_by_source: dict[str, list[dict]] = {}
                if needed_gen:
                    gen_engine = self.cfg.generation_engine or self.cfg.engine
                    generator = Generate(
                        Generate.Config(
                            model=self.cfg.model,
                            prompt_template=self.cfg.prompt_template,
                            chat_template=self.cfg.chat_template,
                            sampling={
                                "temperature": self.cfg.temperature,
                                "top_p": self.cfg.top_p,
                                "repetition_penalty": self.cfg.repetition_penalty,
                                "max_new_tokens": self.cfg.max_generate_tokens,
                            },
                            output_field="continuation",
                            engine=gen_engine,
                            n=self.cfg.n,
                        )
                    )
                    generated = generator.generate_rows(needed_gen, self._resolved_model)
                    for g in generated:
                        src_id = str(g.get("meta.source_id", str(g["meta.id"]).split("#")[0]))
                        gen_by_source.setdefault(src_id, []).append(g)
                    for r in needed_gen:
                        cands = gen_by_source.get(str(r["meta.id"]), [])
                        if cands:
                            r["continuation"] = cands[0].get("continuation", "")

                # Flow step 3: Verify and assemble (delegate to VerifyFilter)
                to_verify = []
                for r in pending_rows:
                    cands_opt = gen_by_source.get(str(r["meta.id"]))
                    if cands_opt and self.cfg.n > 1:
                        for cand in cands_opt:
                            row_copy = dict(cand)
                            if not (row_copy.get("output") and "<Begin_of_Prefix>" in str(row_copy.get("output"))):
                                row_copy["output"] = str(row_copy.get("continuation", ""))
                            if "answer" not in row_copy:
                                row_copy["answer"] = str(row_copy.get("meta.answer", ""))
                            to_verify.append(row_copy)
                    else:
                        row_copy = dict(r)
                        if not (r.get("output") and "<Begin_of_Prefix>" in str(r.get("output"))):
                            row_copy["output"] = str(r.get("continuation", ""))
                        if "answer" not in row_copy:
                            row_copy["answer"] = str(row_copy.get("meta.answer", ""))
                        to_verify.append(row_copy)

                verifier = VerifyFilter(
                    VerifyFilter.Config(
                        grader=self.cfg.grader,
                        pred_field="output",
                        gold_field="answer",
                        assemble="palign",
                        keep_unverified=self.cfg.keep_unverified,
                    )
                )
                verified_rows, _ = verifier._filter(to_verify)

                # Flow step 4: Tokenize & write samples and records
                for rec in verified_rows:
                    r_writer.write(rec)
                    rendered = render_tf(dict(rec), data_ctx)
                    if rendered is not None:
                        tokenized = tokenize_tf(rendered, data_ctx)
                        if tokenized is not None:
                            if "seq.loss_mask" not in tokenized:
                                tokenized["seq.loss_mask"] = [
                                    1 if lb != -100 else 0 for lb in tokenized["student.labels"]
                                ]
                            s_writer.write(tokenized)

        all_samples = read_jsonl(samples_path)
        save_samples(ctx.output.path, all_samples)
        return {"n": len(all_samples)}

    def merge(self, ctx: StageContext, shard_dirs: list[Path]) -> dict:
        source_rows = resolve_source_rows(ctx, self.cfg.source)
        merge_jsonl(ctx, shard_dirs, "records.jsonl", source_rows)
        merged_samples = merge_jsonl(ctx, shard_dirs, "samples.jsonl", source_rows)
        assert ctx.output is not None
        save_samples(ctx.output.path, merged_samples)
        return {"n": len(merged_samples)}

    def load_samples(self, art: Artifact) -> list[dict]:
        return load_samples_dir(art.path)
