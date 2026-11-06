# Repository Structure & Layout

LM-KD is engineered with a strict separation between **declarative experiment recipes**, **pluggable algorithm implementations**, **reusable data pipelines**, and **distributed execution runtimes**.

This document outlines the organization of the codebase to help you quickly find the components relevant to your workflow.

---

## High-Level Directory Layout

```text
LM-KD/
├── lm-kd/
│   ├── src/
│   │   ├── lmkd/                 # Core framework source code
│   │   │   ├── core/             # Base abstractions (Batch, Registry, Config, State)
│   │   │   ├── data/             # Unified collators, aligners, packing, producers
│   │   │   ├── losses/           # Mathematical criteria, divergences, surrogate losses
│   │   │   ├── teachers/         # Teacher inference providers, caches, weight policies
│   │   │   ├── rollout/          # Real-time policy generation (HF, vLLM co-locate/server)
│   │   │   ├── models/           # Student/teacher adapters and architectures
│   │   │   ├── rl/               # Advantage estimators, reward verifiers, sandboxes
│   │   │   ├── train/            # Engine, trainer, optimizers, learning rate schedulers
│   │   │   ├── eval/             # Benchmarks, LLM-as-judge, evaluators, graders
│   │   │   ├── compat/           # DeepSpeed, FSDP, vLLM, and TRL compatibility layers
│   │   │   ├── scaffold/         # Automated code scaffolding tools (`lmkd new`)
│   │   │   ├── cli.py            # Unified command-line interface entrypoint
│   │   │   └── api.py            # Public programmatic Python API
│   │   └── baselines/            # Self-contained baseline algorithm implementations
│   ├── recipes/                  # Modular, composable YAML experiment specifications
│   │   ├── _base_/               # Base configurations (engines, optimizers, evaluation)
│   │   ├── same_tokenizer/       # Same-vocab KD (FKL, RKL, GKD, CSD, TVD, DistiLLM)
│   │   ├── cross_tokenizer/      # Cross-vocab KD (DSKD, DSKDv2, IMPACT, ALM, SRA, ULD)
│   │   ├── embedding/            # Dense embedding KD (DistillCSE, EMO, TALAS, Jasper)
│   │   ├── long_cot/             # Long-chain reasoning KD (SSFT, P-ALIGN, SGL-Spectral)
│   │   ├── self_distill/         # On-policy self-distillation (OPSD, SDPO, RLSD)
│   │   ├── preference/           # RM-free preference KD (PAD, PBSD, CTPD, DP3O, VPD)
│   │   └── self_play/            # Adversarial self-play fine-tuning (SPIN, SPPO, SPACE)
│   ├── datasets/                 # Curated instruction, reasoning, and embedding datasets
│   ├── scripts/                  # Offline generation, data conversion, evaluation scripts
│   ├── official_docs/            # Official documentation suite
│   ├── tests/                    # Unit, regression, and end-to-end integration tests
│   ├── pyproject.toml            # Project packaging specification
│   └── README.md                 # Project landing page
```

---

## Detailed Directory Breakdown

### 1. `src/lmkd/core/` (Foundational Engine)
- [`batch.py`](../../src/lmkd/core/batch.py): Unified tensor container [`Batch`](../../src/lmkd/core/batch.py) with dynamic [`FieldSpec`](../../src/lmkd/core/batch.py) registration, device migration, slicing, and pinning.
- [`registry.py`](../../src/lmkd/core/registry.py): Strict, metadata-rich plugin registry managing 20+ extension points (criterions, producers, teachers, optimizers, evaluators).
- [`config.py`](../../src/lmkd/core/config.py): Pydantic-based configuration system supporting multi-file `_base_` inheritance and dot-key CLI overrides.
- [`validate.py`](../../src/lmkd/core/validate.py): Static recipe validation preventing runtime errors before allocating GPU memory.

### 2. `src/lmkd/data/` (Data Processing & Alignment)
- [`dataset.py`](../../src/lmkd/data/dataset.py): Dataset loaders supporting Hugging Face datasets, JSONL, Parquet, and CSV.
- [`collator.py`](../../src/lmkd/data/collator.py): High-performance dynamic padding and batch collation.
- [`packing.py`](../../src/lmkd/data/packing.py): Sequence packing with cross-attention masks for zero-padding training efficiency.
- [`aligners/`](../../src/lmkd/data/aligners/): Cross-tokenizer alignment algorithms (Token-level Dynamic Time Warping, Subword-to-Subword alignment, Character-span mapping).

### 3. `src/lmkd/losses/` (Distillation Criteria & Divergences)
- [`base.py`](../../src/lmkd/losses/base.py): Base [`Criterion`](../../src/lmkd/losses/base.py), `LossOutput`, and `LossTerm` contracts.
- [`token_kd.py`](../../src/lmkd/losses/token_kd.py): Token-level distillation loss plugin wrapping divergences with temperatures and loss weights.
- [`functional/divergences.py`](../../src/lmkd/losses/functional/divergences.py): Core mathematical divergences: `forward_kl`, `reverse_kl`, `js_distance`, `tv_distance`, `skewed_forward_kl`, `skewed_reverse_kl`, and `csd`.
- [`cross_tokenizer/`](../../src/lmkd/losses/cross_tokenizer/): Cross-vocabulary losses: Universal Logit Distillation (`uld.py`), Multi-Level Optimal Transport (`mlot.py`), and Minimum Edit Distance (`mined.py`).
- [`embedding.py`](../../src/lmkd/losses/embedding.py): Sentence embedding losses: TALAS loss, SimCSE, TAMD relational similarity, and LASD.
- [`aggregator.py`](../../src/lmkd/losses/aggregator.py): Multi-objective loss combination and distributed token normalization (`Norms`).

### 4. `src/lmkd/teachers/` (Teacher Infrastructure)
- [`providers.py`](../../src/lmkd/teachers/providers.py): Hugging Face local teacher providers, offline precomputed logit readers, and API-based black-box teacher interfaces.
- [`signals.py`](../../src/lmkd/teachers/signals.py): Feature extractors (hidden states, attention maps, routing decisions).
- [`weight_policies.py`](../../src/lmkd/teachers/weight_policies.py): Exponential Moving Average (EMA) and lagged checkpoint policies for self-distillation.

### 5. `src/lmkd/rollout/` (On-Policy Rollout Generation)
- [`hf.py`](../../src/lmkd/rollout/hf.py): Standard Hugging Face autoregressive generation engine.
- [`vllm_colocate.py`](../../src/lmkd/rollout/vllm_colocate.py): Co-located vLLM rollout worker sharing GPU memory with PyTorch training processes.
- [`vllm_server.py`](../../src/lmkd/rollout/vllm_server.py): High-throughput decoupled rollout server communicating via IPC or gRPC.
- [`weight_sync.py`](../../src/lmkd/rollout/weight_sync.py): Zero-copy or fast IPC synchronization of student weights into the vLLM engine.

### 6. `recipes/` (Declarative Experiment Configurations)
Organized strictly by research domain:
- `_base_/`: Shared building blocks for DeepSpeed ZeRO configurations, optimizer presets (AdamW, Muon, TALAS), and evaluation suites.
- `same_tokenizer/`: Same-vocabulary distillation methods.
- `cross_tokenizer/`: Cross-vocabulary transfer recipes.
- `embedding/`: Sentence and token embedding distillation methods.
- `long_cot/`: Step-aware and reasoning-chain distillation.
- `self_distill/`: On-policy rollout self-distillation.
- `preference/`: Direct preference alignment without external reward models.
- `self_play/`: Iterative adversarial self-play.

---

## Where to Look for Common Tasks

| Task | Primary Directory / Files |
| :--- | :--- |
| **Run an existing baseline** | [`recipes/`](../../recipes/) |
| **Implement a new loss function** | [`src/lmkd/losses/`](../../src/lmkd/losses/) |
| **Add a new dataset / aligner** | [`src/lmkd/data/`](../../src/lmkd/data/) |
| **Configure distributed training** | [`recipes/_base_/deepspeed/`](../../recipes/_base_/deepspeed/) & [`src/lmkd/train/engine.py`](../../src/lmkd/train/engine.py) |
| **Add an evaluation benchmark** | [`src/lmkd/eval/`](../../src/lmkd/eval/) |

