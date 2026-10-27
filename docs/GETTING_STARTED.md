# Getting Started with LM-KD: A Beginner's Guide

Welcome to **LM-KD**! This guide walks you through everything you need to know to get started with knowledge
distillation for Large Language Models (LLMs) — from conceptual foundations to running your first distillation
experiment in under 5 minutes.

---

## 1. What is LM-KD?

### Knowledge Distillation in a Nutshell
In modern NLP, high-capability teacher models (e.g., 7B, 14B, or 70B parameters) are too computationally
expensive to serve with low latency on edge devices or cost-effective hardware. **Knowledge Distillation (KD)**
transfers the reasoning capability, distribution fidelity, and generative quality of a large **Teacher model**
into a compact **Student model** (e.g., 0.5B to 3B parameters).

### Why LM-KD?
Historically, distillation research was fragmented across dozens of disconnected, brittle codebases:
- White-box logit distillation (DistiLLM, GKD)
- Cross-tokenizer representation transfer (DSKD, ULD, MinED, E-SCOT, IMPACT)
- Long-Chain-of-Thought (Long-CoT) compression (P-ALIGN, SSFT, SGL-Spectral)
- On-policy & self-distillation (OPSD, SDPO, RLSD)
- Preference alignment distillation (DPO, IPO, SimPO, KTO, TDPO)
- Embedding distillation (TALAS, SimCSE)

**LM-KD unifies over 90 published baselines into a single, cohesive framework** with:
- **One canonical training loop** (`rt.train_step`) eliminating copy-pasting.
- **Memory efficiency**: Zero `[B, T, V]` FP32 tensor blowups via target-aligned chunked token views (`LogitsView`).
- **Flexible execution**: 100% self-contained YAML recipes, an intuitive CLI, and a Python library API (`lmkd.build`, `rt.fit()`).
- **Distributed scale**: Seamless execution on CPU, single GPU, DeepSpeed ZeRO-1/2/3, and native PyTorch FSDP2.

---

## 2. Installation & Verification

### Prerequisites
- Linux OS (Ubuntu 20.04+ recommended)
- Python 3.10, 3.11, or 3.12
- PyTorch 2.1+ with CUDA support (or CPU for testing)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-org/LM-KD.git
cd LM-KD/lm-kd

# 2. Create and activate a virtual environment
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 3. Install LM-KD with developer dependencies
pip install -e ".[dev]"
```

### Verifying Your Setup
Run the built-in fast offline smoke test (runs in ~20 seconds on CPU without any downloads):
```bash
bash scripts/run_smoke_tests.sh
```

Check the catalogue of available plugins:
```bash
lmkd list --status stable
```

---

## 3. Your First Distillation Run (30 Seconds)

You can run an end-to-end distillation loop immediately without downloading multi-gigabyte models from the web:

```bash
python examples/quickstart_cpu.py
```

This script:
1. Synthesizes a toy 1-layer GPT-2 student and a 2-layer GPT-2 teacher in RAM.
2. Tokenizes an in-memory batch.
3. Computes Forward KL (`fkl`) distillation combined with Cross-Entropy (`nll`).
4. Executes 5 optimizer updates and prints step metrics.

Output:
```text
[step 1/5] epoch=0.5 sec=0.13 loss/nll/nll=2.108 loss/kd/fkl=2.079 loss/total=2.093 lr/default=0.001
[step 2/5] epoch=1.0 sec=0.04 loss/nll/nll=2.099 loss/kd/fkl=2.080 loss/total=2.090 lr/default=0.001
...
Training completed! Final Step: 5
```

---

## 4. The 6 Orthogonal Axes of Distillation

LM-KD models every distillation method as a coordinate across six independent design choices:

```
                      ┌────────────────────────┐
                      │    One Training Loop   │
                      │    (rt.train_step)     │
                      └───────────┬────────────┘
         ┌──────────────┬─────────┴────────┬──────────────┐
         ▼              ▼                  ▼              ▼
    [Axis A]       [Axis B]           [Axis C]       [Axis D]
      Data         Producer           Teacher        Objective
  (Transforms)    (Batches)          (Signals)       (Losses)
         ▲              ▲                  ▲              ▲
         └──────────────┴─────────┬────────┴──────────────┘
                                  ▼
                        [Axis E]     [Axis F]
                          Step      Optimizer
                       (Strategy)   (Schedulers)
```

| Axis | Question | Config Section | Examples |
|---|---|---|---|
| **A: Data** | How are text samples preprocessed? | `data.transforms` | `tokenize_sft`, `tokenize_cot`, `align` |
| **B: Batches** | Where do batches come from at each step? | `producer` | `dataset` (static), `mixed` (on-policy), `rollout_group` (RL) |
| **C: Teacher** | What does the teacher see and compute? | `teacher` | `hf_frozen`, `self` (EMA / LoRA-disabled), `cache` |
| **D: Objective** | Which loss functions are optimized? | `loss.terms` | `nll`, `token_kd` (`fkl`, `rkl`, `csd`), `dpo` |
| **E: Step** | How does an optimizer step accumulate? | `step` | `standard`, `sam` (sharpness-aware), `ppo_epochs` |
| **F: Optimizer** | Which parameters update and at what rate? | `optim` | AdamW, Muon, Cosine decay, Warmup-Stable-Decay |

---

## 5. Preparing Your Dataset

LM-KD uses standard JSON Lines (`.jsonl`) files where each row is a JSON object.

### Standard Supervised Distillation (SFT / Off-Policy KD)
Create a file `data/train.jsonl`:
```json
{"id": "sample_1", "prompt": "What is the capital of France?", "response": "The capital of France is Paris."}
{"id": "sample_2", "prompt": "Solve: 2x + 4 = 12.", "response": "Subtracting 4 yields 2x = 8. Dividing by 2 gives x = 4."}
```

### Chain-of-Thought (CoT) Reasoning
For reasoning distillation (Long-CoT, P-ALIGN, SSFT, SGL):
```json
{"id": "math_1", "problem": "Find x if 3x - 9 = 0.", "solution": "Add 9 to get 3x = 9, so x = 3.", "answer": "3"}
```

### Preference Distillation (DPO / PO / KTO)
For alignment objectives:
```json
{"id": "pref_1", "prompt": "Write a haiku about code.", "chosen": "Silent screen aglow,\nLogic flows in quiet loops,\nBug becomes a feature.", "rejected": "Code is very nice and good."}
```

---

## 6. Understanding and Customizing Recipes

Recipes in LM-KD are declarative YAML files specifying all configuration parameters.
The master template [`examples/recipes/fkl_full.yaml`](../examples/recipes/fkl_full.yaml) documents every available field.

Here is a minimal practical recipe (`my_run.yaml`):

```yaml
name: qwen_distillation_fkl

models:
  student:
    path: Qwen/Qwen3-0.6B
    dtype: bf16
  teacher:
    path: Qwen/Qwen3-4B-Instruct-2507
    dtype: bf16

data:
  train:
    type: jsonl
    path: data/train.jsonl
  transforms:
    - type: tokenize_sft
      use_chat_template: true
  max_length: 1024
  padding: dynamic

teacher:
  provider: hf_frozen
  input: same

producer:
  type: dataset
  shuffle: true

loss:
  aggregator:
    type: kd_mix
    kd_ratio: 0.5            # 50% Cross-Entropy, 50% Forward KL
    ce_alias: nll
  terms:
    - criterion: nll
    - criterion: token_kd
      alias: kd
      params:
        divergence: fkl      # Forward KL divergence
        chunk_size: 1024     # Evaluated in chunks to avoid memory spikes

optim:
  lr: 2.0e-5
  optimizers:
    default:
      type: adamw
  schedulers:
    - type: cosine
      warmup_ratio: 0.03

engine:
  type: local
  mixed_precision: bf16

train:
  output_dir: outputs/my_first_distill
  epochs: 3
  batch_size: 16             # Effective global batch size
  micro_batch_size: 2        # Per-forward batch size (accumulates 8 steps)
  log_every: 10
```

### CLI Workflow

```bash
# 1. Static validation (checks schema, types, and plugins without loading weights):
lmkd check -c my_run.yaml

# 2. Preflight test (runs 1 real micro-batch on GPU to guarantee memory and compatibility):
lmkd check -c my_run.yaml --preflight

# 3. View the fully resolved configuration:
lmkd config -c my_run.yaml

# 4. Launch training:
CUDA_VISIBLE_DEVICES=0 lmkd train -c my_run.yaml

# 5. Override any parameter directly from the command line:
CUDA_VISIBLE_DEVICES=0 lmkd train -c my_run.yaml \
    --set train.max_steps=200 optim.lr=1e-5 train.batch_size=32
```

---

## 7. Scaling Up: Distributed & Multi-GPU Training

When your model or batch size exceeds single-GPU VRAM, transition to multi-GPU with zero code changes:

### DeepSpeed ZeRO-2 / ZeRO-3
Use the included launcher script:
```bash
NUM_GPUS=4 bash scripts/train_multigpu_deepspeed.sh recipes/same_tokenizer/fkl_zero2.yaml \
    --set train.batch_size=64 train.micro_batch_size=4
```

### PyTorch Native FSDP2
Launch with FSDP2 backend:
```bash
NUM_GPUS=4 bash scripts/train_multigpu_fsdp2.sh recipes/same_tokenizer/fkl.yaml \
    --set train.batch_size=64 train.micro_batch_size=4
```

---

## 8. Checkpoints and Evaluation

Checkpoints are saved automatically into `<train.output_dir>/checkpoint-<step>/` containing:
- Complete student weights and auxiliary module weights.
- Optimizer and scheduler states.
- Data producer progress (resumes deterministically at exact token boundaries).
- `COMPLETE` verification marker.

To evaluate a saved checkpoint:
```bash
lmkd eval -c my_run.yaml --ckpt outputs/my_first_distill/checkpoint-final
```

To resume an interrupted run:
```bash
lmkd train -c my_run.yaml --set train.resume_from=outputs/my_first_distill/checkpoint-500
```

---

## 9. Next Steps

- **Reproduce published papers**: Follow the [Baseline Reproduction Guide](BASELINE_REPRODUCTION.md) to replicate 90+ baselines.
- **Implement a custom method**: Read [Developing and Testing a New Method](NEW_METHOD_GUIDE.md) to write your own loss or aligner in minutes.
- **Deep-dive into architecture**: Check out [Architecture Overview in 15 Minutes](ARCHITECTURE.md) and [Design Specification](DESIGN.md).
- **Contributing**: Consult the [Contributor Guide](CONTRIBUTING.md) for architectural rules and conventions.

