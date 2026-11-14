"""Standard evaluation plugins.

Implements:
- dev_loss: teacher-forced dev set loss under the current training objective.
- generation_rouge: open-ended generation scored with ROUGE-L and exact match.
- math_passk: multi-sample reasoning evaluation on math and MCQ benchmarks with pass@k.
"""

from __future__ import annotations

import copy
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import torch

from lmkd.core.registry import CRITERIONS, EVALUATORS, GRADERS, SOURCES, TRANSFORMS
from lmkd.data.dataset import PreparedDataset
from lmkd.data.transforms.base import run_transforms
from lmkd.eval.base import Evaluator
from lmkd.eval.generation import generate_for_records
from lmkd.eval.graders import rouge_metrics
from lmkd.eval.metrics import mean, score_generations
from lmkd.losses.aggregator import Norms, WeightedOutput
from lmkd.losses.base import LossContext
from lmkd.models.forward import ForwardRunner
from lmkd.models.modules import ModuleRegistry
from lmkd.rollout.base import sampling_from_cfg
from lmkd.train.algorithms import side_inputs
from lmkd.train.step_strategies import count_supervised

logger = logging.getLogger(__name__)


@EVALUATORS.register("dev_loss")
class DevLoss(Evaluator):
    """Teacher-forced dev set loss evaluated under the current training objective."""

    name_prefix = "dev_loss"

    class Config(Evaluator.Config):
        data: dict
        max_samples: int | None = 512
        batch_size: int = 8
        use_chat_template: bool | None = None
        chat_template_kwargs: dict | None = None
        system_prompt: str | None = None
        transforms: list[dict] | None = None

    def setup(self, rt: Any) -> None:
        if hasattr(self, "loader"):
            return
        super().setup(rt)

        source = SOURCES.build(self.cfg.data)
        try:
            records = list(source.records())
        except (FileNotFoundError, OSError):
            logger.warning("Data file for dev_loss not found: %s", self.cfg.data)
            records = []
        if self.cfg.max_samples is not None:
            records = records[: self.cfg.max_samples]

        transforms_cfg = list(self.cfg.transforms or getattr(rt.config.data, "transforms", []) or [])
        if not transforms_cfg and getattr(rt.config, "pipeline", None):
            stages = [stage if isinstance(stage, dict) else stage.model_dump() for stage in rt.config.pipeline]
            for st_dict in stages:
                if st_dict.get("transforms"):
                    transforms_cfg = st_dict["transforms"]
                    break
            if not transforms_cfg:
                for st_dict in stages:
                    if st_dict.get("stage") in ("sgl_spectral", "palign", "ssft"):
                        transforms_cfg = [
                            {"type": "render_cot", "chat_template": st_dict.get("chat_template", True)},
                            {"type": "tokenize_cot", "verify": False},
                        ]
                        break
        if not transforms_cfg and records and "student.input_ids" not in records[0]:
            if "chosen" in records[0] and "rejected" in records[0]:
                transforms_cfg = [{"type": "tokenize_listwise"}]
            else:
                transforms_cfg = [{"type": "tokenize_sft"}]

        if "render_cot" not in TRANSFORMS and any(
            (t.get("type") if isinstance(t, dict) else getattr(t, "type", None)) == "render_cot" for t in transforms_cfg
        ):
            import importlib

            importlib.import_module("baselines.longCoT.transforms")

        if (
            self.cfg.use_chat_template is not None
            or self.cfg.chat_template_kwargs is not None
            or self.cfg.system_prompt is not None
        ):
            normalized = []
            for t_item in copy.deepcopy(transforms_cfg):
                t_dict = t_item if isinstance(t_item, dict) else t_item.model_dump()
                if t_dict.get("type") in ("tokenize_sft", "tokenize_dpo", "tokenize_prompt"):
                    if self.cfg.use_chat_template is not None:
                        t_dict["use_chat_template"] = self.cfg.use_chat_template
                    if self.cfg.chat_template_kwargs is not None:
                        t_dict["chat_template_kwargs"] = self.cfg.chat_template_kwargs
                    if self.cfg.system_prompt is not None:
                        t_dict["system_prompt"] = self.cfg.system_prompt
                elif t_dict.get("type") == "render_cot" and self.cfg.use_chat_template is not None:
                    t_dict["chat_template"] = self.cfg.use_chat_template
                normalized.append(t_dict)
            transforms_cfg = normalized

        transforms = [TRANSFORMS.build(t) for t in transforms_cfg]
        for t in transforms:
            t.setup(rt)

        samples = run_transforms(records, transforms, rt.data_ctx) if transforms else records
        world_size = getattr(rt.engine, "world_size", 1)
        if world_size > 1 and not getattr(rt.engine, "is_partitioned", False):
            rank = getattr(rt.engine, "rank", 0)
            samples = samples[rank::world_size]
        self.dataset = PreparedDataset(samples, transforms, rt.data_ctx)
        is_cuda = getattr(getattr(rt.engine, "device", None), "type", None) == "cuda"
        collate_fn = getattr(rt.producer, "_flatten", rt.collator)
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=is_cuda,
        )

    def evaluate(self, rt: Any) -> dict[str, float]:
        if not hasattr(self, "loader"):
            self.setup(rt)

        prefix = self.cfg.name or getattr(self, "alias", None) or self.name_prefix
        world_size = getattr(rt.engine, "world_size", 1)
        if len(self.loader) == 0 and world_size <= 1:
            return {f"{prefix}/loss": 0.0}

        model = getattr(rt, "student_module", rt.student_model)
        was_training = model.training
        model.eval()

        device = rt.engine.device
        tot_loss_num, tot_z_tok = 0.0, 0.0
        term_nums: dict[str, float] = {}

        autocast_ctx = rt.engine.autocast() if hasattr(rt.engine, "autocast") else nullcontext()
        try:
            with torch.no_grad(), autocast_ctx:
                for batch in self.loader:
                    micro = batch.to(device)
                    if hasattr(rt.algorithm, "augment") and (
                        not hasattr(rt.algorithm, "estimator") or "rollout.reward" in micro
                    ):
                        micro = rt.algorithm.augment(micro, rt)
                    elif hasattr(rt, "teacher_input"):
                        micro = rt.teacher_input.build(micro, rt)
                    criteria = getattr(rt, "criteria", [])
                    loss_weights = getattr(rt, "loss_weights", {})
                    missing = [
                        t.alias
                        for t in criteria
                        if loss_weights.get(t.alias, 0.0) != 0.0
                        and not set(getattr(t.criterion, "requires", ())).issubset(micro.keys())
                    ]
                    if not missing:
                        outs = rt.algorithm.loss(micro, rt)
                    else:
                        if not getattr(self, "_warned_missing_reqs", False):
                            logger.warning(
                                "dev_loss data lacks pipeline-cached keys for criteria %s; "
                                "evaluating remaining/fallback supervised loss",
                                missing,
                            )
                            self._warned_missing_reqs = True
                        saved_weights = dict(loss_weights)
                        try:
                            for alias in missing:
                                rt.loss_weights[alias] = 0.0
                            has_active = any(w != 0.0 for w in rt.loss_weights.values()) or (
                                getattr(rt, "loss_fn", None) is not None
                            )
                            if has_active:
                                outs = rt.algorithm.loss(micro, rt)
                            else:
                                fallback_name = (
                                    "chosen_nll" if ("group.size" in micro and "group.index" in micro) else "nll"
                                )
                                if fallback_name == "chosen_nll" and "chosen_nll" not in CRITERIONS:
                                    fallback_name = "nll"
                                crit = CRITERIONS.build(fallback_name)
                                views = ForwardRunner.run(
                                    rt.student_model,
                                    rt.student_adapter,
                                    side_inputs(micro, "student"),
                                    crit.features().student,
                                    grad=False,
                                    module=getattr(rt, "student_module", rt.student_model),
                                )
                                rt_modules = getattr(rt, "modules", None)
                                ctx = LossContext(
                                    batch=micro,
                                    student=views[0],
                                    student_views=views,
                                    teacher=None,
                                    modules=ModuleRegistry() if rt_modules is None else rt_modules,
                                    state=rt.state,
                                    rt=rt,
                                )
                                outs = [WeightedOutput("nll", 1.0, crit(ctx))]
                        finally:
                            rt.loss_weights.update(saved_weights)
                    z_tok, z_seq = count_supervised(micro)
                    if z_tok == 0:
                        continue
                    loss, mets = rt.aggregator.combine(outs, Norms(z_tok, z_seq, 1))
                    tot_loss_num += float(loss.item()) * z_tok
                    tot_z_tok += z_tok
                    for k, v in mets.items():
                        term_nums[k] = term_nums.get(k, 0.0) + float(v) * z_tok

            if world_size > 1 and hasattr(rt.engine, "all_gather_object"):
                gathered_keys = rt.engine.all_gather_object(list(term_nums.keys()))
                keys = sorted({k for ks in gathered_keys for k in ks})
            else:
                keys = sorted(term_nums.keys())
            packed_vals = [tot_loss_num, tot_z_tok] + [term_nums.get(k, 0.0) for k in keys]
            reduced_tensor = rt.engine.all_reduce_sum(torch.tensor(packed_vals, device=device, dtype=torch.float64))
            reduced_vals = [float(x) for x in reduced_tensor.tolist()]
            tot_loss_num, tot_z_tok = reduced_vals[0], reduced_vals[1]

            results = {f"{prefix}/loss": tot_loss_num / tot_z_tok if tot_z_tok else 0.0}
            for i, k in enumerate(keys):
                results[f"{prefix}/{k}"] = reduced_vals[2 + i] / tot_z_tok if tot_z_tok else 0.0
            return results
        finally:
            if was_training:
                model.train()


def _gather_eval_results(
    rt: Any, preds: list[str], recs: list[dict], refs: list[list[str]], total: int, world_size: int
) -> tuple[list[str], list[dict], list[list[str]]]:
    all_preds = rt.engine.all_gather_object(preds)
    all_recs = rt.engine.all_gather_object(recs)
    all_refs = rt.engine.all_gather_object(refs)
    flat_preds: list[str] = [None] * total  # type: ignore
    flat_recs: list[dict] = [None] * total  # type: ignore
    flat_refs: list[list[str]] = [None] * total  # type: ignore
    for r_idx in range(world_size):
        for j, (p, r, rf) in enumerate(zip(all_preds[r_idx], all_recs[r_idx], all_refs[r_idx])):
            idx = r_idx + j * world_size
            if idx < total:
                flat_preds[idx], flat_recs[idx], flat_refs[idx] = p, r, rf
    return flat_preds, flat_recs, flat_refs


@EVALUATORS.register("generation_rouge")
class GenerationRouge(Evaluator):
    """Generates completions and scores them with exact match and ROUGE-L."""

    name_prefix = "generation_rouge"

    class Config(Evaluator.Config):
        data: dict
        sampling: dict = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 512}
        engine: dict | str | None = None
        max_samples: int | None = 500
        seeds: list[int] = [10]
        use_chat_template: bool = False
        chat_template_kwargs: dict = {}
        system_prompt: str | None = None

    def _load_eval_records(self) -> list[dict]:
        source = SOURCES.build(self.cfg.data)
        try:
            records = list(source.records())
        except (FileNotFoundError, OSError):
            logger.warning("Data file for dev_loss not found: %s", self.cfg.data)
            records = []
        if self.cfg.max_samples is not None:
            records = records[: self.cfg.max_samples]
        return records

    def _extract_references(self, records: list[dict]) -> list[list[str]]:
        refs: list[list[str]] = []
        for r in records:
            if "references" in r:
                ref = r["references"]
                refs.append(ref if isinstance(ref, list) else [str(ref)])
            elif "response" in r:
                refs.append([str(r["response"])])
            elif "output" in r:
                out_val = r["output"]
                refs.append([str(x) for x in out_val] if isinstance(out_val, list) else [str(out_val)])
            elif "gold" in r:
                refs.append([str(r["gold"])])
            elif "answer" in r:
                refs.append([str(r["answer"])])
            else:
                refs.append([])
        return refs

    def evaluate(self, rt: Any) -> dict[str, float]:
        records = self._load_eval_records()
        prefix = self.cfg.name or getattr(self, "alias", None) or self.name_prefix
        if not records:
            return {f"{prefix}/exact_match": 0.0, f"{prefix}/rougeL": 0.0}
        references = self._extract_references(records)
        raw_file = self.output_dir(rt) / "raw.jsonl"

        em_scores, rouge_scores = [], []
        world_size = getattr(rt.engine, "world_size", 1)
        rank = getattr(rt.engine, "rank", 0)

        for seed in self.cfg.seeds:
            spec = sampling_from_cfg({**self.cfg.sampling, "seed": seed})
            r_recs = records[rank::world_size] if world_size > 1 else records
            r_refs = references[rank::world_size] if world_size > 1 else references

            rollouts = generate_for_records(
                rt,
                r_recs,
                spec,
                self.cfg.engine,
                use_chat_template=self.cfg.use_chat_template,
                chat_template_kwargs=self.cfg.chat_template_kwargs,
                system_prompt=self.cfg.system_prompt,
            )
            r_preds = [r[0].text if r else "" for r in rollouts]

            if world_size > 1:
                preds, eval_recs, eval_refs = _gather_eval_results(
                    rt, r_preds, r_recs, r_refs, len(records), world_size
                )
            else:
                preds, eval_recs, eval_refs = r_preds, r_recs, r_refs

            if rt.engine.is_main:
                with open(raw_file, "a", encoding="utf-8") as f:
                    for rec, pred, ref in zip(eval_recs, preds, eval_refs):
                        row = {
                            "id": rec.get("id", rec.get("meta.id")),
                            "prompt": rec.get("prompt", ""),
                            "prediction": pred,
                            "references": ref,
                            "seed": seed,
                        }
                        f.write(json.dumps(row) + "\n")

            metrics = rouge_metrics(preds, eval_refs)
            em_scores.append(metrics["exact_match"])
            rouge_scores.append(metrics["rougeL"])

        return {f"{prefix}/exact_match": mean(em_scores), f"{prefix}/rougeL": mean(rouge_scores)}


@EVALUATORS.register("math_passk")
class MathPassK(Evaluator):
    """Multi-sample generation and pass@k evaluation on math and reasoning benchmarks."""

    name_prefix = "math_passk"

    class Config(Evaluator.Config):
        benchmarks: list[str]
        n: int = 8
        ks: list[int] = [1]
        sampling: dict = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 32768}
        grader: str = "math_verify"
        engine: dict | str | None = None
        batch_size: int | None = 64
        rescore_only: bool = False
        use_chat_template: bool = False
        chat_template_kwargs: dict = {}
        system_prompt: str | None = None

    def _load_bench_records(self, bench: str) -> list[dict]:
        p = Path(bench)
        if bench.endswith(".jsonl") or (p.exists() and p.is_file()):
            source = SOURCES.build({"type": "jsonl", "path": bench})
        else:
            source = SOURCES.build({"type": "benchmark", "name": bench})
        return list(source.records())

    def _generate_benchmark(
        self, rt: Any, records: list[dict], raw_file: Path, done_ids: set[Any], existing: list[dict]
    ) -> list[dict]:
        to_run = [r for r in records if r.get("id", r.get("meta.id")) not in done_ids]
        if not to_run:
            return list(existing)

        world_size = getattr(rt.engine, "world_size", 1)
        rank = getattr(rt.engine, "rank", 0)
        rank_to_run = to_run[rank::world_size] if world_size > 1 else to_run

        sampling_cfg = dict(self.cfg.sampling)
        sampling_cfg["n"] = self.cfg.n
        spec = sampling_from_cfg(sampling_cfg)
        bs = self.cfg.batch_size or 64

        rank_rows: list[dict] = []
        for i in range(0, len(rank_to_run), bs):
            chunk = rank_to_run[i : i + bs]
            rollouts = generate_for_records(
                rt,
                chunk,
                spec,
                self.cfg.engine,
                use_chat_template=self.cfg.use_chat_template,
                chat_template_kwargs=self.cfg.chat_template_kwargs,
                system_prompt=self.cfg.system_prompt,
            )
            for rec, r_list in zip(chunk, rollouts):
                row = {
                    "id": rec.get("id", rec.get("meta.id")),
                    "gold": str(rec.get("gold", rec.get("answer", ""))),
                    "task_type": rec.get("task_type", "math"),
                    "generations": [
                        {"text": r.text, "finish_reason": r.finish_reason, "n_tokens": len(r.response_ids)}
                        for r in r_list
                    ],
                }
                rank_rows.append(row)

        if world_size > 1:
            all_rank_rows = rt.engine.all_gather_object(rank_rows)
            flat_rows: list[dict] = [None] * len(to_run)  # type: ignore
            for r_idx in range(world_size):
                for j, item in enumerate(all_rank_rows[r_idx]):
                    idx = r_idx + j * world_size
                    if idx < len(to_run):
                        flat_rows[idx] = item
            new_rows = flat_rows
        else:
            new_rows = rank_rows

        if rt.engine.is_main:
            with open(raw_file, "a", encoding="utf-8") as f:
                for row in new_rows:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
        return list(existing) + new_rows

    def evaluate(self, rt: Any) -> dict[str, float]:
        out_dir = self.output_dir(rt)
        raw_dir = out_dir / "raw"
        if rt.engine.is_main:
            raw_dir.mkdir(parents=True, exist_ok=True)

        grader_spec = self.cfg.grader if isinstance(self.cfg.grader, dict) else {"type": self.cfg.grader}
        grader = GRADERS.build(grader_spec)
        grader.setup(rt)

        prefix = self.cfg.name or getattr(self, "alias", None) or self.name_prefix
        all_metrics: dict[str, float] = {}
        pass1_values: list[float] = []

        for bench in self.cfg.benchmarks:
            bench_name = Path(bench).stem if (bench.endswith(".jsonl") or "/" in bench) else bench
            raw_file = raw_dir / f"{bench_name}.jsonl"

            if self.cfg.rescore_only:
                if not raw_file.exists():
                    logger.warning("Rescore only requested but %s does not exist.", raw_file)
                    continue
                rows = [json.loads(line) for line in raw_file.read_text().splitlines() if line.strip()]
            else:
                existing_rows: list[dict] = []
                done_ids = set()
                if raw_file.exists():
                    for line in raw_file.read_text().splitlines():
                        if line.strip():
                            r = json.loads(line)
                            done_ids.add(r.get("id"))
                            existing_rows.append(r)
                records = self._load_bench_records(bench)
                rows = self._generate_benchmark(rt, records, raw_file, done_ids, existing_rows)

            summary = score_generations(rows, grader, ks=self.cfg.ks)
            if rt.engine.is_main:
                with open(out_dir / f"{bench_name}_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

            for k, v in summary.items():
                all_metrics[f"{prefix}/{bench_name}/{k}"] = v
            if "pass@1" in summary:
                pass1_values.append(summary["pass@1"])

        if pass1_values:
            all_metrics[f"{prefix}/avg/pass@1"] = mean(pass1_values)

        return all_metrics


@EVALUATORS.register("lm_eval", source="lm_eval:simple_evaluate")
class LMEval(Evaluator):
    """LM Evaluation Harness plugin running standard benchmarks (HellaSwag, ARC, MMLU, GSM8K, IFEval, etc.).
    Supports both HuggingFace (in-memory model) and vLLM backends (high throughput on GPU).
    """

    name_prefix = "lm_eval"

    class Config(Evaluator.Config):
        backend: Literal["auto", "vllm", "hf"] = "auto"
        tasks: list[str] = ["hellaswag", "arc_challenge", "truthfulqa_mc2", "winogrande", "gsm8k"]
        num_fewshot: int | None = None
        batch_size: int | str = "auto"
        max_batch_size: int | None = 64
        limit: int | float | None = None
        device: str | None = None
        tensor_parallel_size: int = 1
        gpu_memory_utilization: float = 0.8
        max_model_len: int | None = None
        apply_chat_template: bool = False
        system_instruction: str | None = None
        extra_args: dict[str, Any] = {}

    def evaluate(self, rt: Any) -> dict[str, float]:
        if hasattr(rt, "engine") and hasattr(rt.engine, "is_main") and not rt.engine.is_main:
            return {}

        import lm_eval

        model = rt.unwrap_model()
        tokenizer = getattr(rt, "student_tokenizer", None)

        device = self.cfg.device
        if device is None:
            dev_obj = getattr(getattr(rt, "engine", None), "device", None)
            if dev_obj is not None:
                device = str(dev_obj)
            elif hasattr(model, "device"):
                device = str(model.device)
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check for existing checkpoint path on disk
        ckpt_path = None
        cand = getattr(getattr(getattr(rt, "config", None), "models", None), "student", None)
        cand_path = getattr(cand, "path", None)
        if cand_path and Path(cand_path).exists():
            ckpt_path = str(cand_path)
        train_cfg = getattr(getattr(rt, "config", None), "train", None)
        if not ckpt_path and train_cfg:
            out_dir = Path(getattr(train_cfg, "output_dir", "."))
            for p in [out_dir / "checkpoint-final", out_dir / "final"]:
                if p.exists():
                    ckpt_path = str(p)
                    break

        use_vllm = False
        if self.cfg.backend == "vllm":
            use_vllm = True
            if not ckpt_path:
                temp_ckpt = self.output_dir(rt) / "model_weights"
                temp_ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(temp_ckpt)
                if tokenizer is not None:
                    tokenizer.save_pretrained(temp_ckpt)
                ckpt_path = str(temp_ckpt)
        elif self.cfg.backend == "auto" and ckpt_path and torch.cuda.is_available() and device.startswith("cuda"):
            # auto uses vllm if checkpoint is available on disk and cuda is active
            try:
                from lmkd.compat import vllm_api

                vllm_api.import_vllm()
                use_vllm = True
            except (ImportError, RuntimeError):
                use_vllm = False

        if use_vllm and ckpt_path:
            from lm_eval.models.vllm_causallms import VLLM

            tok_path = getattr(tokenizer, "name_or_path", None) or ckpt_path
            lm_obj = VLLM(
                pretrained=ckpt_path,
                tokenizer=tok_path,
                batch_size=self.cfg.batch_size,
                tensor_parallel_size=self.cfg.tensor_parallel_size,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                max_model_len=self.cfg.max_model_len,
                dtype="auto",
            )
        else:
            from lm_eval.models.huggingface import HFLM

            lm_obj = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=self.cfg.batch_size,
                max_batch_size=self.cfg.max_batch_size,
                device=device,
            )

        res = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=self.cfg.tasks,
            num_fewshot=self.cfg.num_fewshot,
            limit=self.cfg.limit,
            apply_chat_template=self.cfg.apply_chat_template,
            system_instruction=self.cfg.system_instruction,
            **self.cfg.extra_args,
        )

        prefix = self.cfg.name or getattr(self, "alias", None) or self.name_prefix
        metrics: dict[str, float] = {}
        primary_scores: list[float] = []

        if res and "results" in res:
            out_dir = self.output_dir(rt)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "lm_eval_summary.json", "w", encoding="utf-8") as f:
                json.dump(res.get("results", {}), f, indent=2)

            for task_name, task_dict in res["results"].items():
                if not isinstance(task_dict, dict):
                    continue
                val = None
                for candidate in (
                    "acc_norm,none",
                    "acc,none",
                    "exact_match,strict-match",
                    "exact_match,none",
                    "strict_acc,none",
                    "prompt_level_strict_acc,none",
                    "inst_level_strict_acc,none",
                ):
                    if candidate in task_dict and isinstance(task_dict[candidate], (int, float)):
                        val = float(task_dict[candidate])
                        break
                if val is None:
                    for k, v in task_dict.items():
                        if k.endswith(",none") and isinstance(v, (int, float)) and "stderr" not in k and k != "sample_len":
                            val = float(v)
                            break

                if val is not None:
                    metrics[f"{prefix}/{task_name}"] = val
                    primary_scores.append(val)

                for k, v in task_dict.items():
                    if isinstance(v, (int, float)) and "stderr" not in k and k != "sample_len":
                        clean_k = k.replace(",none", "")
                        metrics[f"{prefix}/{task_name}/{clean_k}"] = float(v)

        if primary_scores:
            metrics[f"{prefix}/avg"] = mean(primary_scores)

        return metrics

