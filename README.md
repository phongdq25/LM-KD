<div align="center">

# LM-KD

**A unified, modular toolkit for Language Model Knowledge Distillation**

Reproducible benchmarking and rapid prototyping of distillation and alignment methods for LLMs, from a single hardware-efficient runtime.

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-ee4c2c?logo=pytorch&logoColor=white)
![Backends](https://img.shields.io/badge/backends-DeepSpeed%20%7C%20FSDP2%20%7C%20vLLM-orange)
![Lint](https://img.shields.io/badge/code%20style-ruff-261230)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-Survey%20Paper-b31b1b.svg)](#citation)

[Documentation](official_docs/index.md) · [Quickstart](official_docs/getting-started/quickstart.md) · [Methods](official_docs/methods/index.md) · [Scenarios & Baselines](official_docs/scenarios/index.md) · [Results](official_docs/results/leaderboards.md) · [Contributing](official_docs/contributing/index.md)

</div>

---

## Table of Contents

<table width="100%">
<tr>
<td valign="top" width="33%">

- [Overview](#overview)
- [Key Features](#key-features)
- [Taxonomy & Supported Methods](#taxonomy--supported-methods)
- [Experimental Scenarios](#experimental-scenarios)
- [Installation](#installation)

</td>
<td valign="top" width="33%">

- [Quickstart](#quickstart)
- [Recipes & Configuration](#recipes--configuration)
- [Extending LM-KD](#extending-lm-kd)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)

</td>
<td valign="top" width="33%">

- [Evaluation](#evaluation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

</td>
</tr>
</table>

---

## Overview

> **Companion toolkit of a survey.** LM-KD is the open-source implementation and benchmarking toolkit that accompanies the survey ***Knowledge Distillation for Large Language Models: A Survey of Training Contexts and Supervision***. The survey introduces the taxonomy used in this README, and LM-KD provides the implementations of the surveyed methods together with the seven benchmark scenarios discussed in the paper (see [Experimental Scenarios](#experimental-scenarios)).

Knowledge distillation (KD) for LLMs has moved far beyond classical logit matching: it now covers cross-tokenizer transfer, long chain-of-thought compression, on-policy self-distillation, preference distillation and iterative self-play. Research in this space is often fragmented:

- **Disparate codebases.** Methods live in bespoke forks of TRL, LLaMA-Factory, verl, Megatron, etc., so it is unclear whether gains come from the algorithm or from hidden engineering details (padding masks, warmups, accumulation denominators).
- **GPU memory explosions.** Large vocabularies (V ≥ 128k) trigger OOM when logits are materialized over long contexts.
- **Incompatible interfaces.** Comparing token-level, sequence-level and representation-level losses usually means rewriting the training loop.

**LM-KD** addresses this with a single runtime built around one universal batch contract, a plugin registry, declarative YAML recipes and centralized, distributed-safe loss normalization. Every distillation run is decomposed along **six orthogonal axes**, so changing one component (e.g. the loss, the teacher, or the context source) does not require touching the others.

## Key Features

- **Unified plugin registry.** Implement a loss, teacher provider, dataset, aligner, optimizer or evaluator in a standalone Python file and register it with a single decorator, with no core changes.
- **Declarative recipes.** Experiments are YAML files with `_base_` inheritance, Pydantic schema validation and CLI dot-key overrides.
- **Target-aligned lazy chunking (`LogitsView`).** Divergences are computed over streamed sequence chunks, so the full `[Batch, Length, Vocab]` tensor is never materialized (documented peak-memory reduction of over 60%).
- **Centralized global token normalization.** Loss numerators are divided by the exact global count of supervised tokens (`num_items_in_batch`) across micro-batches and ranks, so gradients are numerically identical regardless of batch size, accumulation steps or GPU count.
- **Flexible teachers.** Local Hugging Face teachers (incl. 4/8-bit quantized), precomputed top-K logit caches, remote API teachers, and EMA / lagged self-teachers.
- **On-policy rollouts.** Colocated or server-mode **vLLM**, plus a Hugging Face generation fallback and Ray-based decoupled rollouts.
- **Distributed first.** DeepSpeed ZeRO-1/2/3, PyTorch FSDP2, FlashAttention-2 and attention-isolated sequence packing.
- **Cross-tokenizer tooling.** Built-in aligners (`dtw_edit`, `parent_groups`, `exact_token`), learnable projectors and optimal-transport losses.
- **PEFT support.** LoRA / QLoRA through Hugging Face PEFT.
- **Offline teacher caching.** Precompute teacher logits once; the docs report 3.5x to 8x faster training with zero teacher memory.
- **Developer experience.** `lmkd try`, `lmkd check`, `lmkd config`, `lmkd explain`, `lmkd list` and `lmkd new` for fast iteration and scaffolding.

## Taxonomy & Supported Methods

LM-KD organizes distillation methods by one question: **where does the training context come from?** In an autoregressive model every token is predicted from a prefix, and the teacher's signal can be applied to prefixes taken from a reference corpus, from an external model's answers, or from sequences generated by the student itself. Knowing the teacher or the loss does not tell you which contexts the student actually learns on, so the taxonomy classifies each training configuration by the **origin and version of its training context**.

```text
Taxonomy by Training Context Source
├── 1. External Supervision
│   ├── 1.1. Reference text only
│   │   ├── 1.1.1. Autoregressive generation on reference prefixes
│   │   └── 1.1.2. Text embedding and dense retrieval
│   └── 1.2. Text generated by external models
├── 2. Current-policy Supervision
│   ├── 2.1. Non-interactive rollouts
│   └── 2.2. Environment-interactive rollouts
├── 3. Lagged-policy Supervision
│   ├── 3.1. Data from a single student checkpoint
│   └── 3.2. Data from multiple student checkpoints
└── 4. Mixed-policy Supervision
    ├── 4.1. Mixing samples across sources
    ├── 4.2. Combining sources within the generation history
    └── 4.3. Staged source transitions
```

| Branch | Training context comes from | Methods in LM-KD |
| :--- | :--- | :--- |
| **1. External Supervision** | A reference corpus or text written by external models; no student rollouts | **Same-tokenizer logit KD:** FKL, RKL, JSD, TVD, SFKL, SRKL<br>**Cross-tokenizer KD:** DSKD, DSKDv2, DSKD-CMA, DSKDv2-ETA, ULD, MinED, MultiLevelOT, E-SCOT, ALM, SRA, MCW-KD, ResidualKD, DWA-KD<br>**Long-CoT reasoning:** SSFT, P-ALIGN, SGL-Spectral<br>**Preference distillation:** VPD (PAD), PPD, DCKD, TVKD, ADPA, CTPD, DP3O, PBSD<br>**Embedding distillation:** TALAS, DistillCSE, EMO, Jasper, MoL |
| **2. Current-policy Supervision** | Rollouts generated by the exact student version that is about to be updated | **On-policy self-distillation:** OPSD, SDPO, RLSD |
| **3. Lagged-policy Supervision** | Student-generated data that is used after the student's parameters have changed | **Self-play fine-tuning:** SPIN, T-SPIN, SPACE, SPPO, S-SPPO, RSPO |
| **4. Mixed-policy Supervision** | External text combined with student rollouts | CSD, DistiLLM, GKD, FDD, AMiD, MTA |

> **Classification unit.** The taxonomy classifies a concrete *training configuration*, not an algorithm name. The same method can fall into different branches depending on how it is run (for example, GKD can be external, current-policy or mixed depending on its mixing ratio). The table lists each method under the branch of the configuration evaluated in LM-KD.
>
> The taxonomy answers *"which contexts does the student learn on?"*, not *"which loss is optimized?"*. Methods with very different losses can share a branch, and one family can span several branches (for example, the same-tokenizer methods are split between External and Mixed). MTA is classified under Mixed, including when it is combined with a cross-tokenizer base method. Method families remain the unit of documentation: formulations, pseudocode and recipes are grouped by family in the [Methods Catalog](official_docs/methods/index.md), and the concepts are explained in [Taxonomy of LLM Knowledge Distillation](official_docs/concepts/taxonomy.md). For primary source publications, authors, and conference links across all 49+ baselines, see the [Scenarios & Baseline Registry](official_docs/scenarios/index.md#baseline-methods--paper-registry).

**Method families and their docs**

| Family | Methods and recipes |
| :--- | :--- |
| Same-Tokenizer KD | [Methods](official_docs/methods/same-tokenizer.md) · `recipes/same_tokenizer/` |
| Cross-Tokenizer KD | [Methods](official_docs/methods/cross-tokenizer.md) · `recipes/cross_tokenizer/` |
| Long-CoT Reasoning | [Methods](official_docs/methods/long-cot.md) · `recipes/long_cot/` |
| On-Policy Self-Distillation | [Methods](official_docs/methods/on-policy-self-distill.md) · `recipes/self_distill/` |
| Preference Distillation | [Methods](official_docs/methods/preference-distill.md) · `recipes/preference/` |
| Embedding Distillation | [Methods](official_docs/methods/embedding-distill.md) · `recipes/embedding/` |
| Self-Play Fine-Tuning | [Methods](official_docs/methods/self-play.md) · `recipes/self_play/` |

## Experimental Scenarios

Scenarios are a separate axis from the taxonomy. A **scenario** is a standardized benchmark setting: a fixed set of teacher/student pairs, training data, evaluation benchmarks and metrics shared by every method that competes in it. The taxonomy describes *how a method obtains its training context*; a scenario describes *what the method is evaluated on*. Each scenario page contains the full protocol and its results table; all tables are compiled in the [Master Leaderboards](official_docs/results/leaderboards.md).

| # | Scenario | Teacher → Student pairs | Training data → evaluation | Details |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **Sentence embedding distillation** | Qwen3-Embedding-0.6B → MiniLMv2-L6, BGE-M3 → MiniLMv2-L12, Qwen3-Embedding-4B → BERT-base | AllNLI, Wikipedia → 9 MTEB tasks (classification, pair classification, STS) | [Scenario 1](official_docs/scenarios/1-sentence-embedding.md) |
| 2 | **Same-tokenizer instruction distillation** | GPT-2 (1.5B → 120M), Qwen1.5 (1.8B → 0.5B), OPT (6.7B → 1.3B) | Dolly-15k → Dolly, Self-Instruct, Vicuna, Super-NaturalInstructions (ROUGE-L) | [Scenario 2](official_docs/scenarios/2-same-tokenizer.md) |
| 3 | **Cross-tokenizer distillation** | Qwen1.5-1.8B → GPT-2 120M and 0.4B; Qwen2.5-7B → GPT-2 1.5B and OPT-2.7B; Mistral-7B → TinyLLaMA-1.1B | Dolly-15k → Dolly, Self-Instruct, Vicuna, Super-NaturalInstructions (ROUGE-L) | [Scenario 3](official_docs/scenarios/3-cross-tokenizer.md) |
| 4 | **Long-CoT reasoning distillation** | DeepSeek-R1 → R1-Distill-Qwen-1.5B, Qwen2.5-7B-Instruct, Qwen3-8B | s1K-1.1 → AIME 2024/2025, AMC 12, MATH-500 (pass@1, pass@3) | [Scenario 4](official_docs/scenarios/4-long-cot.md) |
| 5 | **On-policy self-distillation for reasoning** | Qwen3-8B, Olmo3-7B-Think, Qwen3-4B (no external teacher) | GSM8K, MATH, NuminaMath-CoT → AIME 2025, AIME 2026, HMMT 2025 (accuracy) | [Scenario 5](official_docs/scenarios/5-opsd.md) |
| 6 | **Reward-model-free preference distillation** | Llama-3.1-8B → Llama-3.2-1B, Qwen3-8B → Qwen3-1.7B, Qwen3-1.7B → Qwen3-0.6B | UltraFeedback Binarized → HellaSwag, ARC, TruthfulQA, Winogrande, GSM8K | [Scenario 6](official_docs/scenarios/6-preference-distillation.md) |
| 7 | **Self-play fine-tuning** | Qwen2.5-1.5B, Gemma-2-2B, Qwen3-4B | UltraChat-200k, SPIN_iter0 → GSM8K, MATH, MMLU, MMLU-Pro, HellaSwag, Winogrande, IFEval | [Scenario 7](official_docs/scenarios/7-self-play-fine-tuning.md) |

Because the two axes are independent, a scenario can compare methods that sit in different taxonomy branches (for instance, Scenario 5 compares SFT, GRPO and OPSD/RLSD/SDPO), and a taxonomy branch can be evaluated across several scenarios.

## Installation

### Option A: `uv` (recommended)

```bash
git clone https://github.com/phongdq25/LM-KD.git
cd LM-KD

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Option B: Conda + pip

```bash
conda create -n lmkd python=3.11 -y
conda activate lmkd

pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

cd LM-KD
pip install -e ".[dev]"
```

### Optional acceleration backends

```bash
pip install flash-attn --no-build-isolation   # FlashAttention-2 (recommended for context length >= 2048)
```

### Verify

```bash
bash scripts/run_smoke_tests.sh
bash scripts/run_baseline.sh --all --check
```

## Quickstart

A first end-to-end run (Qwen3-0.6B student from Qwen3-4B-Instruct-2507 on Dolly-15k with Forward KL) takes roughly 10 minutes on a single GPU. Full walkthrough: [Quickstart](official_docs/getting-started/quickstart.md).

**1. Sanity-check a loss in seconds (no model or dataset download):**

```bash
lmkd try fkl
lmkd try rkl
```

**2. Inspect and validate a recipe without allocating GPUs:**

```bash
lmkd config -c recipes/same_tokenizer/fkl.yaml     # print the fully merged config
lmkd check  -c recipes/same_tokenizer/fkl.yaml     # schema, imports and vocab validation
```

**3. Train:**

```bash
lmkd train -c recipes/same_tokenizer/fkl.yaml \
    --set train.max_steps=50 \
          train.log_every=10 \
          train.output_dir=./outputs/quickstart_fkl
```

**4. Evaluate:**

```bash
lmkd eval -c recipes/same_tokenizer/fkl.yaml \
    --ckpt ./outputs/quickstart_fkl/checkpoint-50
```

### Python API

```python
import lmkd

rt = lmkd.quick(
    student="Qwen/Qwen2.5-0.5B",
    teacher="Qwen/Qwen2.5-1.5B",
    data="datasets/same_cross_tokenizer_data/dolly/train.jsonl",
    loss="fkl",
    max_steps=100,
    lr=2e-5,
    output_dir="./outputs/quick_fkl",
)
rt.fit()
```

For full control, build a `Runtime` and drive the loop yourself:

```python
rt = lmkd.build("recipes/same_tokenizer/fkl.yaml", overrides=["train.max_steps=500"])

rt.begin()
while not rt.done:
    out = rt.train_step()                 # micro-batches, all-reduce, optimizer step
    if rt.state.step % 50 == 0:
        print(rt.state.step, out.metrics.get("loss/total", 0.0), rt.evaluate())
rt.end()
rt.save("final_checkpoint")
```

### CLI reference

| Command | Purpose |
| :--- | :--- |
| `lmkd train -c <recipe>` | Run distillation |
| `lmkd eval -c <recipe> [--ckpt <path>]` | Evaluate a checkpoint on benchmark suites |
| `lmkd config -c <recipe>` | Print the resolved configuration |
| `lmkd check -c <recipe>` | Static validation of a recipe |
| `lmkd explain -c <recipe>` | Architectural summary of a recipe |
| `lmkd try <criterion>` | Dry-run a loss on synthetic tensors |
| `lmkd list [registry]` | List registered plugins and components |
| `lmkd pipeline -c <recipe>` | Offline pipelines (e.g. teacher logit caching) |
| `lmkd init <name>` | Initialize a complete baseline/method template directory (`dev/` or `src/baselines/`) |
| `lmkd new <kind> <name>` | Scaffold a single plugin, test, and recipe |

## Recipes & Configuration

A recipe is a single YAML file describing a whole experiment. Recipes can inherit from base templates via `_base_` and any value can be overridden from the CLI via `--set a.b.c=value`.

```yaml
# recipes/same_tokenizer/fkl.yaml
name: fkl_qwen3_0.6b_from_4b

models:
  student:
    path: Qwen/Qwen3-0.6B
    dtype: bf16
    task: causal_lm
  teacher:
    path: Qwen/Qwen3-4B-Instruct-2507
    dtype: bf16
    task: causal_lm
    placement: same_device

data:
  train:
    type: jsonl
    path: datasets/same_cross_tokenizer_data/dolly/train.jsonl
  eval:
    type: jsonl
    path: datasets/same_cross_tokenizer_data/dolly/dev.jsonl
  transforms:
    - type: tokenize_sft
      use_chat_template: true
  max_length: 1024

loss:
  aggregator:
    type: kd_mix
    kd_ratio: 1.0
    ce_alias: nll
  terms:
    - criterion: nll
    - criterion: token_kd
      alias: kd
      params:
        divergence: fkl
        chunk_size: 1024

train:
  batch_size: 16
  micro_batch_size: 4
  max_steps: 2000
  output_dir: ./outputs/qwen_fkl
  seed: 42

optim:
  lr: 2.0e-5
  optimizers:
    default:
      type: adamw
```

CLI overrides with `--set`:

```bash
lmkd train -c recipes/same_tokenizer/fkl.yaml \
    --set optim.lr=5e-5 \
          train.batch_size=8 \
          train.micro_batch_size=2
```

Recipes are organized by research domain under [`recipes/`](recipes/): `same_tokenizer/`, `cross_tokenizer/`, `embedding/`, `long_cot/`, `self_distill/`, `preference/`, `self_play/`, plus shared building blocks in `_base_/` (DeepSpeed engines, optimizers, evaluation suites). More in the [Configuration System](official_docs/architecture/configuration-system.md) guide.

## Extending LM-KD

Add a new loss in a standalone file and register it, with no changes to core code:

```python
from __future__ import annotations

import torch
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import CRITERIONS
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest


@CRITERIONS.register("temperature_mse", paper="arXiv:xxxx.xxxxx", status="stable")
class TemperatureMSELoss(Criterion):
    class Config(BaseConfig):
        temperature: float = 1.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.temp = config.temperature

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        s_logits = ctx.student_logits().full() / self.temp
        t_logits = (ctx.teacher.logits.full() if ctx.teacher else s_logits) / self.temp
        diff = (s_logits - t_logits).pow(2).sum()
        # Return an unnormalized numerator; the framework normalizes globally.
        return LossOutput(terms={"mse": LossTerm(value=diff, reduction="token_sum")})
```

Use it from a recipe:

```yaml
imports:
  - ./plugins/temperature_mse.py

loss:
  terms:
    - criterion: temperature_mse
      params:
        temperature: 2.0
```

Or generate boilerplate (implementation, unit test, minimal recipe) with `lmkd new criterion my_custom_kd`. See [Developing a Custom Loss](official_docs/guides/developing-custom-loss.md) and [Adding a New Baseline](official_docs/guides/adding-a-baseline.md).

> **Three rules for criterions:** (1) never shift logits manually; only `LogitsView` does, and every per-token tensor is target-aligned; (2) never call a model inside a criterion; (3) never divide by batch size or sequence length; return raw numerators and let the aggregator normalize.


## Architecture

Every distillation algorithm is decomposed into six independent axes:

| Axis | Responsibility |
| :--- | :--- |
| **A. Context** | Where prefixes come from: external corpus, current policy, lagged policy, or a mixed schedule (see the [taxonomy](#taxonomy--supported-methods)) |
| **B. Producer** | Batch collation, packing, multi-turn, preference pairs, cross-tokenizer aligners |
| **C. Teacher** | Local HF, vLLM, remote API, EMA checkpoint, or offline cache |
| **D. Criterion** | The divergence or objective (FKL, RKL, JSD, TVD, CSD, Sinkhorn, DPO-style, ...) |
| **E. Step** | Execution strategy: standard backprop, PPO/GRPO-style, dual-model sync, gradient hooks |
| **F. Optimizer** | AdamW, Muon, TALAS, SAM, schedules, clipping |

```
Recipe (YAML) ─▶ Pydantic RecipeConfig ─▶ Plugin Registries
                                              │
Raw data ─▶ Batch Producer ─▶ Unified Batch ──┤
                                  ┌───────────┴───────────┐
                         Student forward           Teacher Provider
                                  └───────────┬───────────┘
                                       Criterion ─▶ Loss Aggregator ─▶ Engine
                                                      (global Z_tok)   (ZeRO / FSDP2)
```

Learn more: [Core Abstractions](official_docs/architecture/core-abstractions.md), [Registry & Plugins](official_docs/architecture/registry-plugin-system.md), [Data Flow & Caching](official_docs/architecture/data-flow-and-caching.md), [Distributed Backends](official_docs/architecture/distributed-backends.md), and the [Taxonomy of LLM KD](official_docs/concepts/taxonomy.md).

## Repository Structure

```text
LM-KD/
├── src/
│   ├── lmkd/               # Core framework
│   │   ├── core/           # Batch, Registry, Config, State
│   │   ├── data/           # Collators, aligners, packing, producers
│   │   ├── losses/         # Criterions, divergences, aggregator
│   │   ├── teachers/       # Teacher providers, caches, weight policies
│   │   ├── rollout/        # HF / vLLM rollout engines, weight sync
│   │   ├── models/         # Adapters, LogitsView, ChunkedHead
│   │   ├── rl/             # Advantage estimators, verifiers, sandboxes
│   │   ├── train/          # Engine, trainer, optimizers, schedulers
│   │   ├── eval/           # Benchmarks, evaluators, graders, LLM-as-judge
│   │   ├── compat/         # DeepSpeed / FSDP / vLLM / TRL compatibility
│   │   ├── scaffold/       # `lmkd new` code generators
│   │   ├── cli.py          # Command-line entrypoint
│   │   └── api.py          # Public Python API
│   └── baselines/          # Self-contained baseline implementations
├── recipes/                # YAML experiment specifications (grouped by method family)
├── datasets/               # Instruction, reasoning and embedding datasets
├── examples/               # Standalone scripts, custom loops, and tutorials
├── packages/               # Internal packages (tokenkit)
├── scripts/                # Generation, conversion and evaluation scripts
├── official_docs/          # Documentation
├── tests/                  # Unit, regression and integration tests
├── LICENSE                 # Apache 2.0 License
└── pyproject.toml
```

## Evaluation

LM-KD includes a unified evaluation harness (`lmkd eval`) with registered evaluators, graders and a standardized protocol.

- **Math and reasoning:** `math500`, `aime24`, `aime25`, `amc23`, `amc12`, `minerva`, `olympiadbench`, `gpqa` (unbiased pass@k with `math_verify` grading).
- **Instruction following:** Dolly, Self-Instruct, Vicuna, Super-NaturalInstructions (ROUGE-L, exact match).
- **Sentence embeddings:** 9 MTEB-style tasks covering STS, pair classification and linear-probe classification.
- **LLM-as-a-judge:** pairwise win-rate with bidirectional position swap and length-bias reporting.
- **Protocol:** greedy decoding for deterministic benchmarks, 3-seed replication (42, 123, 456), and paired bootstrap significance testing.

See [Metrics Catalog](official_docs/evaluation/metrics-catalog.md), [Benchmark Suites](official_docs/evaluation/benchmarks.md), [Evaluation Protocol](official_docs/evaluation/eval-protocol.md) and the [Master Leaderboards](official_docs/results/leaderboards.md) for the full results of all 7 experimental scenarios.

## Documentation

| If you want to... | Start here |
| :--- | :--- |
| Run your first experiment | [Quickstart](official_docs/getting-started/quickstart.md) |
| Understand the concepts | [Concepts & Taxonomy](official_docs/concepts/index.md) |
| Reproduce a published method | [Methods](official_docs/methods/index.md), [Scenarios](official_docs/scenarios/index.md) |
| Look up baseline papers & links | [Scenarios & Baseline Registry](official_docs/scenarios/index.md#baseline-methods--paper-registry) |
| Implement a new method | [Architecture](official_docs/architecture/index.md), [Guides](official_docs/guides/index.md), [API Reference](official_docs/api/index.md) |
| Prepare data | [Data Schemas & Datasets](official_docs/data/index.md) |
| Debug OOM / NaN issues | [Troubleshooting](official_docs/getting-started/troubleshooting.md), [Debugging & Profiling](official_docs/guides/debugging-and-profiling.md) |
| Contribute | [Contributing Guide](official_docs/contributing/index.md) |

## Contributing

Contributions are welcome: new baselines, distributed backends (Megatron-LM, MLX, TPU, ROCm), fused Triton/CUDA kernels, and benchmark loaders.

```bash
git checkout -b feature/my-new-feature
ruff check --fix . && ruff format .
pytest tests/ -v
lmkd check -c recipes/<your_recipe>.yaml
```

Before opening a pull request, please read the [Code Style & Standards](official_docs/contributing/code-style.md) (Ruff, 120-char lines, Python 3.11 type hints) and the [PR Checklist](official_docs/contributing/pr-checklist.md).

## Citation

If you use LM-KD or the taxonomy in your research, please cite the survey:

```bibtex
@article{lmkd_survey,
  title  = {Knowledge Distillation for Large Language Models: A Survey of Training Contexts and Supervision},
  author = {},
  year   = {},
  url    = {}
}
```

and the software:

```bibtex
@software{lmkd,
  title  = {LM-KD: A Unified Toolkit for Language Model Knowledge Distillation},
  author = {},
  year   = {},
  url    = {https://github.com/phongdq25/LM-KD}
}
```

Please also cite the original papers of the baselines you use; complete bibliographies, authors, and links are cataloged in the [Baseline Methods & Paper Registry](official_docs/scenarios/index.md#baseline-methods--paper-registry) and on each method page.

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgements

LM-KD builds on the open-source ecosystem, including PyTorch, Hugging Face Transformers / Datasets / PEFT, DeepSpeed, vLLM, FlashAttention, and Ray. Thanks to the authors of all the baseline methods implemented in this toolkit.
