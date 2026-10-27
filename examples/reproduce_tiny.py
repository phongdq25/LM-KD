"""Offline six-domain smoke reproduction: generate local tiny models/data and launch recipes.

Usage: .venv/bin/python examples/reproduce_tiny.py --output-dir outputs/tiny-reproduction
This checks executable workflows; random tiny models do not reproduce paper scores.
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertModel, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from lmkd.reproduce import execute_plan, resolve_jobs, validate_jobs, write_plan


def make_models(root):
    """Save tiny causal student, a different-tokenizer teacher and an encoder."""
    torch.manual_seed(42)
    words = [f"w{i}" for i in range(60)] + [".", "1", "<think>", "</think>"]
    paths = {key: root / key for key in ("student", "teacher", "encoder")}
    for key, path in paths.items():
        units = list("abcdefghijklmnopqrstuvwxyz0123456789 .") if key == "teacher" else words
        vocabulary = {word: i for i, word in enumerate(["<pad>", "<eos>", "<unk>", *units])}
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        tokenizer.pre_tokenizer = (pre_tokenizers.Split("", "isolated") if key == "teacher"
                                  else pre_tokenizers.WhitespaceSplit())
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>", pad_token="<pad>",
                                       unk_token="<unk>")
        fast.save_pretrained(path)
        if key == "encoder":
            config = BertConfig(vocab_size=len(fast), hidden_size=32, num_hidden_layers=3,
                                num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
            model = BertModel(config)
        else:
            config = GPT2Config(vocab_size=len(fast), n_positions=128, n_embd=48 if key == "teacher" else 32,
                                n_layer=3 if key == "teacher" else 2, n_head=2, bos_token_id=1,
                                eos_token_id=1, pad_token_id=0)
            model = GPT2LMHeadModel(config)
        model.save_pretrained(path)
    return {key: str(path) for key, path in paths.items()}


def make_data(root):
    """Write domain-specific data rather than silently coercing one schema into all domains."""
    sources = {}
    rows = [{"id": str(i), "prompt": "w1 w2 ", "response": "w3 w4 w5", "problem": "w1 w2",
             "solution": "w3 w4", "answer": "1", "chosen": "w3 w4", "rejected": "w5 w6"} for i in range(16)]
    for name in ("sft", "math", "preference", "cot"):
        path = root / f"{name}.jsonl"
        data = [{**row, "response": "w3 w4 .\n\nw5 w6 ."} for row in rows] if name == "cot" else rows
        path.write_text("\n".join(json.dumps(row) for row in data) + "\n")
        sources[name] = str(path)
    path = root / "embedding.csv"
    path.write_text("text\n" + "\n".join(f"w{i % 10} w{(i + 1) % 10}" for i in range(16)) + "\n")
    sources["embedding"] = str(path)
    return sources


def smoke_jobs(root, models_, data):
    """Use normal catalogue resolution, with explicit CPU/tiny-size overrides."""
    common = ["engine={type: local, device: cpu, mixed_precision: 'no'}", "models.student.dtype=fp32",
              "data.max_length=64", "train.max_steps=2", "train.batch_size=4", "train.micro_batch_size=2",
              "train.log_every=1", "eval=[]", "callbacks=[]"]
    specs = {
        "fkl": {"data": data["sft"], "teacher": models_["student"],
                "overrides": ["models.teacher.dtype=fp32"]},
        "uld": {"data": data["sft"], "teacher": models_["teacher"],
                "overrides": ["models.teacher.dtype=fp32"]},
        "sgl_spectral": {"data": data["cot"], "overrides": ["pipeline.0.transforms.0.chat_template=false",
                                                               "pipeline.0.transforms.1.verify=false",
                                                               "pipeline.1.device=cpu", "pipeline.1.dtype=fp32"]},
        "grpo": {"data": data["math"], "overrides": ["rollout={type: hf}", "producer.n=2",
                                                         "producer.sampling.max_new_tokens=4", "step.epochs=1",
                                                         "data.transforms.0.use_chat_template=false"]},
        "dpo": {"data": data["preference"], "reference": models_["student"],
                "overrides": ["models.teacher.dtype=fp32", "pipeline.0.transforms.0.max_length=64",
                              "pipeline.0.transforms.0.max_prompt_length=32",
                              "pipeline.0.transforms.0.format_plain_prompts=false"]},
        "talas_asam": {"student": models_["encoder"], "data": data["embedding"],
                       "embedding_teacher": models_["encoder"],
                       "overrides": [f"models.student.tokenizer={models_['encoder']}", "pipeline.1.pooling=cls",
                                     "pipeline.1.dtype=fp32", "loss.terms.1.params.teacher_dim=32"]},
    }
    for name, options in specs.items():
        overrides = common + options.pop("overrides")
        options.setdefault("student", models_["student"])
        jobs = resolve_jobs(name, root / name, overrides=overrides, **options)
        validate_jobs(name, jobs)
        yield name, jobs


def main(argv=None):
    """Generate inputs, run each representative and retain reproduction evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/tiny-reproduction")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_paths, data = make_models(root), make_data(root)
    for name, jobs in smoke_jobs(root, model_paths, data):
        destination = root / name
        plan = write_plan(name, jobs, destination)
        if not args.dry_run:
            execute_plan(plan, destination)
        print(f"{name}: {'planned' if args.dry_run else 'completed'} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
