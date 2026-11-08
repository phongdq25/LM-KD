"""Common pipeline utilities for Long-CoT stages (sharding, I/O, tokenization, model loading)."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from lmkd.compat import vllm_api
from lmkd.core.config import ModelSpec
from lmkd.core.registry import ROLLOUT, SOURCES, TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import run_transforms
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.loading import load_model, load_tokenizer
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import read_jsonl, shard_of, write_jsonl
from lmkd.pipeline.stage import StageContext
from lmkd.rollout.base import SamplingSpec
from lmkd.rollout.vllm_colocate import make_llm


def input_art(ctx: StageContext, *names: str) -> Artifact:
    for name in names:
        if name in ctx.inputs:
            return ctx.inputs[name]
    raise ValueError(f"missing input artifact: expected one of {names}")


def input_rows(ctx: StageContext, *names: str) -> list[dict]:
    art = input_art(ctx, *names)
    for filename in ("samples.jsonl", "records.jsonl"):
        if (art.path / filename).exists():
            return read_jsonl(art.path / filename)
    raise FileNotFoundError(f"no samples.jsonl or records.jsonl in {art.path}")


def pending(ctx: StageContext, filename: str, rows: list[dict]) -> tuple[Path, list[dict]]:
    assert ctx.output is not None
    path = ctx.output.path / filename
    return path, shard_of(rows, *ctx.shard)


def merge_jsonl(ctx: StageContext, shard_dirs: list[Path], filename: str, source_rows: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    by_source: dict[str, list[dict]] = {}
    for folder in shard_dirs:
        for row in read_jsonl(folder / filename):
            sid = str(row["meta.id"])
            if sid in by_id:
                raise ValueError(f"duplicate shard id {sid}")
            by_id[sid] = row
            src_id = str(row.get("meta.source_id", sid.split("#")[0]))
            by_source.setdefault(src_id, []).append(row)
    ordered: list[dict] = []
    for row in source_rows:
        rid = str(row["meta.id"])
        if rid in by_id:
            ordered.append(by_id[rid])
        elif rid in by_source:
            ordered.extend(by_source[rid])
    assert ctx.output is not None
    write_jsonl(ctx.output.path / filename, ordered)
    return ordered


def write_signals_parquet(path: Path) -> None:
    try:
        import pandas as pd

        pd.DataFrame(read_jsonl(path)).to_parquet(path.with_suffix(".parquet"), index=False)
    except ImportError:
        pass


def load_frozen(path: str, dtype: str = "bf16", device: str = "auto") -> tuple[Any, Any, HFAutoAdapter]:
    spec = ModelSpec(path=path, dtype=dtype)
    model = load_model(spec, trainable=False)
    target = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model.to(target)
    return model, load_tokenizer(spec), HFAutoAdapter(HFAutoAdapter.Config())


def model_path(configured: str, ctx: StageContext) -> str:
    if configured:
        return configured
    spec = getattr(getattr(ctx.rt, "config", None), "models", None)
    student = getattr(spec, "student", None)
    if student is None:
        raise ValueError("stage model is empty and runtime has no models.student")
    return str(student.path)


def span(sample: dict) -> tuple[int, int]:
    start, end = map(int, sample["seq.response_span"])
    if start < 1 or end <= start or end > len(sample["student.input_ids"]):
        raise ValueError(f"invalid response span for {sample.get('meta.id')}: {(start, end)}")
    return start, end


def chat_ids(tokenizer: Any, prompt: str, enable_thinking: bool | None = False) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            encoded = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            encoded = tokenizer.apply_chat_template(messages, **kwargs)
        return list(encoded["input_ids"] if isinstance(encoded, dict) else encoded)
    return list(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def engine_name(value: dict | str) -> tuple[str, dict]:
    if isinstance(value, str):
        name, options = value, {}
    else:
        spec = dict(value)
        name, options = str(spec.pop("type")), spec
    if name not in ("hf", "vllm", "vllm_colocate"):
        raise ValueError(f"unknown generation engine {name!r}")
    return name, options


def generate_texts(
    model_path_str: str,
    prompts: list[str],
    tokenizer: Any,
    engine_spec: dict | str,
    spec: SamplingSpec,
    chat_template: bool = True,
    enable_thinking: bool | None = None,
) -> list[str]:
    name, options = engine_name(engine_spec)
    prompt_ids = []
    for p in prompts:
        ids = (
            chat_ids(tokenizer, p, enable_thinking)
            if chat_template
            else list(tokenizer(p, add_special_tokens=False)["input_ids"])
        )
        prompt_ids.append(ids)

    if name in ("vllm", "vllm_colocate"):
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
        llm = make_llm(model_path_str, **kwargs)
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
            outputs = llm.generate([vllm_api.tokens_prompt(p) for p in prompt_ids], params, use_tqdm=False)
            return [out.outputs[0].text for out in outputs]
        finally:
            del llm
            gc.collect()
    else:
        dtype = (
            "bf16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else ("fp16" if torch.cuda.is_available() else "fp32")
        )
        model = load_model(ModelSpec(path=model_path_str, dtype=dtype), trainable=False)
        if torch.cuda.is_available():
            model = model.to("cuda")
        runtime = SimpleNamespace(student_model=model, student_tokenizer=tokenizer)
        rollout_engine = ROLLOUT.build({"type": "hf", **options})
        rollout_engine.setup(runtime)
        try:
            groups = rollout_engine.generate(prompt_ids, spec)
            return [group[0].text for group in groups]
        finally:
            del model, rollout_engine
            gc.collect()


def resolve_source_rows(ctx: StageContext, source_cfg: dict | str | None) -> list[dict]:
    if source_cfg is not None:
        spec = {"type": "jsonl", "path": source_cfg} if isinstance(source_cfg, str) else dict(source_cfg)
        src_obj = SOURCES.build(spec)
        if spec.get("type") == "artifact_records":
            node = spec["node"]
            art = ctx.inputs.get(node, ctx.inputs.get("prev"))
            if art is None:
                raise ValueError(f"artifact_records source needs pipeline input node {node!r}")
            src_obj.bind(art)
        rows = list(src_obj.records())
    else:
        rows = input_rows(ctx, "data", "records", "prev")
    for i, r in enumerate(rows):
        if "meta.id" not in r:
            r["meta.id"] = str(r.get("id", i))
    return rows


def ensure_segmented(
    rows: list[dict],
    ctx: StageContext,
    segmenter: str = "sentence",
    store_texts: bool = False,
    chat_template: bool = True,
    model_path: str = "",
    tokenize_verify: bool = True,
    **kwargs: Any,
) -> list[dict]:
    if (
        rows
        and "student.input_ids" in rows[0]
        and "seq.steps" in rows[0]
        and (not store_texts or "meta.segments" in rows[0])
    ):
        return rows
    resolved_path = model_path or kwargs.get("model_path_str", "")
    data_ctx = getattr(ctx.rt, "data_ctx", None)
    if data_ctx is None:
        tok = getattr(ctx.rt, "student_tokenizer", None) if ctx.rt is not None else None
        if tok is None and hasattr(ctx.rt, "config") and getattr(ctx.rt.config.models, "student", None):
            tok = load_tokenizer(ctx.rt.config.models.student)
        elif tok is None and resolved_path:
            tok = load_tokenizer(ModelSpec(path=resolved_path))
        data_ctx = DataContext(
            student_tokenizer=tok,
            teacher_tokenizer=getattr(ctx.rt, "teacher_tokenizer", None) if ctx.rt is not None else None,
            max_length=getattr(getattr(ctx.rt, "config", None), "data", None)
            and ctx.rt.config.data.max_length
            or 32768,
            artifacts=getattr(ctx.rt, "artifacts", None) if ctx.rt is not None else None,
        )
    transforms = [
        TRANSFORMS.build({"type": "render_cot", "chat_template": chat_template}),
        TRANSFORMS.build({"type": "tokenize_cot", "verify": tokenize_verify}),
        TRANSFORMS.build({"type": "segment", "segmenter": segmenter, "store_texts": store_texts}),
    ]
    for t in transforms:
        if hasattr(t, "setup"):
            t.setup(ctx.rt)
    return run_transforms(rows, transforms, data_ctx)


def sample_selection_stats(row: dict) -> dict[str, float]:
    spans = row.get("seq.steps", [])
    selected = row.get("meta.selected_steps", [])
    total_steps = len(spans)
    total_tokens = sum(end - start for start, end in spans)
    kept_tokens = sum(spans[i][1] - spans[i][0] for i in selected)
    step_retention = len(selected) / total_steps if total_steps else 0.0
    token_retention = kept_tokens / total_tokens if total_tokens else 0.0
    mask, weights = row.get("seq.loss_mask"), row.get("seq.token_weights")
    mass_ratio = (
        sum(float(m) * float(w) for m, w in zip(mask, weights)) / sum(mask)
        if mask is not None and weights is not None and sum(mask)
        else 1.0
    )
    return {
        "n_steps": float(total_steps),
        "n_steps_kept": float(len(selected)),
        "n_steps_dropped": float(total_steps - len(selected)),
        "step_retention": step_retention,
        "step_drop": 1.0 - step_retention if total_steps else 0.0,
        "n_tokens": float(total_tokens),
        "n_tokens_kept": float(kept_tokens),
        "n_tokens_dropped": float(total_tokens - kept_tokens),
        "token_retention": token_retention,
        "token_drop": 1.0 - token_retention if total_tokens else 0.0,
        "keeps_final_step": float(total_steps - 1 in selected) if total_steps else 0.0,
        "weight_mass_ratio_mean": mass_ratio,
    }


def selection_stats(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    stats = [sample_selection_stats(row) for row in rows]
    return {key: sum(item[key] for item in stats) / len(stats) for key in stats[0]}
