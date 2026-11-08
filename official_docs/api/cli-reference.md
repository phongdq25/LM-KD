# CLI Reference: `lmkd` Command-Line Interface

`lmkd` is the unified command-line entrypoint for the LM-KD distillation framework. It exposes 10 subcommands covering the complete distillation workflow: registry exploration, instant loss probing, recipe validation, offline data pipelines, distributed training, evaluation, and code scaffolding.

```bash
lmkd [-h] <subcommand> [options]
```

---

## 1. Common Options for Recipe-Driven Commands

Commands that operate on YAML recipes (`train`, `eval`, `check`, `config`, `explain`, `pipeline`) inherit these common flags:

| Flag | Type | Description |
|---|---|---|
| `-c`, `--config` | `str` (repeatable) | Path to one or more YAML recipe files. Multiple recipes are deeply merged in order. |
| `--set` | `str` (repeatable) | Dynamic hyperparameter overrides in dotted syntax: `a.b.c=value`. |
| `--plugins` | `str` (repeatable) | Paths to custom Python plugin files (`.py`) or importable module paths. |

---

## 2. Command Index

The 10 subcommands are organized into three functional categories:

```
lmkd
├── Inspection & Validation
│   ├── lmkd list       # Inspect all registered plugins across all registries
│   ├── lmkd try        # 1-second gradient & math probe on synthetic tensors
│   ├── lmkd check      # Static configuration & live preflight validation
│   ├── lmkd config     # Print resolved & deeply merged YAML configuration
│   └── lmkd explain    # Print architectural summary (models, data flow, losses)
├── Execution & Evaluation
│   ├── lmkd train      # Execute distillation training
│   ├── lmkd eval       # Evaluate checkpoint against registered benchmark suites
│   └── lmkd pipeline   # Run offline data & signal preprocessing DAG stages
└── Scaffolding & Development
    ├── lmkd init       # Initialize complete method/baseline template directory
    └── lmkd new        # Scaffold a single plugin, test, and starter recipe
```

---

## 3. Inspection & Validation Commands

### `lmkd list`

List registered plugins, their operational status, documentation, and source references.

```bash
lmkd list [registry] [--status {stable,experimental,stub}] [-v] [--plugins [PLUGINS ...]]
```

#### Arguments & Flags
- `registry` *(optional)*: Limit output to a specific registry name (e.g. `criterions`, `transforms`, `teachers`, `graders`, `evaluators`, `optimizers`, `stages`). If omitted, lists all registries.
- `--status`: Filter components by status (`stable`, `experimental`, or `stub`).
- `-v`, `--verbose`: Print source repository pointers, paper citations, task IDs, and configurable Pydantic parameters.
- `--plugins`: Import extra plugin files (`.py`) or modules before listing.

#### Examples
```bash
# List all registered components across all registries
lmkd list

# List all loss functions (criterions)
lmkd list criterions

# Inspect detailed configuration schema and paper sources for stable aligners
lmkd list aligners --status stable -v
```

---

### `lmkd try`

Run a 1-second synthetic CPU sanity check verifying loss calculation and student parameter gradient backpropagation.

```bash
lmkd try <loss> [--plugins [PLUGINS ...]]
```

#### Arguments & Flags
- `loss` *(required)*: Registered name of the criterion to test (e.g. `fkl`, `rkl`, `token_kd`, `nll`, `jsd`).
- `--plugins`: Import extra plugin modules or `.py` files defining custom loss functions.

> [!NOTE]
> **Scope & Limitations of `lmkd try`**:
> `lmkd try` executes [`probe_gradient`](../../src/lmkd/scaffold/probe.py#L34-L49) with a minimal synthetic batch (`2x4` shape with `student.input_ids` and `student.labels`) and a mock `torch.nn.Identity()` model.
> - **Supported:** Stateless, logits-only divergence losses (e.g., `fkl`, `rkl`, `jsd`, `token_kd`, `nll`, `topk_kd`).
> - **Unsupported:** Criteria requiring auxiliary projection modules (`mta`, `mol`, `emo`), transformer layers (`dskd`), token alignment graphs (`alm`, `escot`), or preference/rollout structures (`adpa`, `luffy`).
> - For complex criteria, use **`lmkd check --preflight`** or the fast CPU tiny smoke test (`pytest tests/e2e_tiny/test_<name>_e2e.py`).

#### Examples
```bash
# Verify standard forward KL divergence
lmkd try fkl

# Verify custom loss defined in a standalone file
lmkd try my_custom_kd --plugins ./dev/my_loss.py
```

---

### `lmkd check`

Validate recipe configurations for structural consistency, dependency resolution, and runtime feasibility.

```bash
lmkd check -c <recipe.yaml> [--preflight] [--json] [--set ...] [--plugins ...]
```

#### Arguments & Flags
- `-c`, `--config` *(required)*: Path to the YAML recipe.
- `--preflight`: Instead of only static validation, instantiates the runtime and executes **1 live micro-batch forward and backward pass** with real models and data.
- `--json`: Output check results in structured JSON format (ideal for CI/CD pipelines).

#### Validation Checks Performed
1. **Static validation (default):**
   - Registry lookups for all declared components (`models`, `transforms`, `teacher`, `loss`, `optim`, `eval`).
   - Contract compatibility: ensures producer/transform outputs provide all batch fields required by the teacher input and criterion (e.g., `seq.loss_mask`, `student.offsets`).
   - Tokenizer and vocabulary consistency across student and teacher.
2. **Preflight validation (`--preflight`):**
   - Loads models into CPU/CUDA memory.
   - Runs one batch from `data.train` through data collators.
   - Executes student forward, teacher signal extraction, and loss computation.
   - Confirms backpropagation produces finite gradients on trainable parameters.

#### Examples
```bash
# Fast static validation
lmkd check -c recipes/same_tokenizer/distillm_mta.yaml

# Preflight test with 1 real micro-batch
lmkd check -c recipes/same_tokenizer/distillm_mta.yaml --preflight

# CI/CD check with JSON output
lmkd check -c recipes/same_tokenizer/fkl.yaml --json
```

---

### `lmkd config`

Resolve inheritance (`base:` recipes) and print the fully merged, final configuration in clean YAML.

```bash
lmkd config -c <recipe.yaml> [--set ...]
```

#### Examples
```bash
# View resolved configuration with merged defaults
lmkd config -c recipes/same_tokenizer/fkl.yaml

# Inspect how dynamic overrides affect the final recipe
lmkd config -c recipes/same_tokenizer/fkl.yaml --set train.batch_size=32 optim.lr=5e-5
```

---

### `lmkd explain`

Print a human-readable architectural summary of a recipe, including engine, student/teacher pair, batch layouts, and weighted loss objectives.

```bash
lmkd explain -c <recipe.yaml>
```

#### Examples
```bash
lmkd explain -c recipes/cross_tokenizer/dskdv2_mta.yaml
```

*Output summary:*
```text
======================================================================
Recipe: dskdv2_mta
======================================================================
Engine:     local
Student:    Qwen/Qwen3-0.6B
Teacher:    Qwen/Qwen3-4B-Instruct-2507
Batch size: 4 (micro: 2)
Max steps:  3

Loss Terms:
  [0] nll (alias: nll, weight: 0.2)
  [1] dskdv2 (alias: kd, weight: 0.8)
  [2] mta (alias: mta, weight: 1.0)
======================================================================
```

---

## 4. Training & Evaluation Commands

### `lmkd train`

Launch knowledge distillation training on single-GPU, multi-GPU (DDP), DeepSpeed (ZeRO-1/2/3), or PyTorch FSDP2.

```bash
lmkd train -c <recipe.yaml> [--set ...] [--plugins ...]
```

Before training starts, `lmkd train` automatically executes static recipe validation (`check_recipe`). If any schema or contract error is found, training aborts immediately with diagnostic feedback.

#### Common Parameter Overrides via `--set`
```bash
# Adjust steps, learning rate, and output directory
lmkd train -c recipes/same_tokenizer/fkl.yaml \
    --set train.max_steps=1000 \
          train.output_dir=./outputs/fkl_experiment \
          optim.lr=3e-5

# Quick debugging run with sample limit and frequent logging
lmkd train -c recipes/same_tokenizer/fkl.yaml \
    --set data.train.max_samples=100 \
          train.max_steps=10 \
          train.log_every=1
```

#### Distributed Training Launchers
To run distributed training with Accelerate or Torchrun, pass the `lmkd.cli` module to the launcher:

```bash
# Multi-GPU training via Torchrun
torchrun --nproc_per_node=4 -m lmkd.cli train -c recipes/same_tokenizer/fkl_fsdp2.yaml

# DeepSpeed ZeRO-3 via Accelerate
accelerate launch --config_file accelerate_zero3.yaml -m lmkd.cli train -c recipes/same_tokenizer/fkl.yaml
```

---

### `lmkd eval`

Evaluate a student checkpoint or base model against configured evaluation suites (`dev_loss`, `generation_rouge`, `math_passk`, `embedding_sts`, etc.).

```bash
lmkd eval -c <recipe.yaml> [--ckpt <path>] [--set ...] [--plugins ...]
```

#### Arguments & Flags
- `-c`, `--config` *(required)*: Path to evaluation or training recipe.
- `--ckpt`: Path to saved checkpoint directory. Automatically resolves Hugging Face weights, LoRA adapters (`adapter_config.json`), and student tokenizer.

#### Examples
```bash
# Evaluate checkpoint produced during training
lmkd eval -c recipes/same_tokenizer/distillm_mta.yaml \
    --ckpt ./outputs/distillm_mta/checkpoint-500

# Override evaluation benchmark on the fly
lmkd eval -c recipes/same_tokenizer/fkl.yaml \
    --set eval.0.data.limit=200
```

---

### `lmkd pipeline`

Execute offline DAG stages such as pretokenization, teacher logit caching, vocabulary mapping, or offline span alignment.

```bash
lmkd pipeline -c <pipeline.yaml> [--until <stage>] [--only [STAGES ...]] [--force [STAGES ...]] [--shard i/n] [--dry-run]
```

#### Arguments & Flags
- `--dry-run`: Display execution plan, input/output artifacts, and cache status without executing stages.
- `--until <stage>`: Execute pipeline up to and including the specified stage.
- `--only <stage1> <stage2>`: Run only designated stages.
- `--force [STAGES ...]`: Invalidate artifact cache and force re-running specified stages (or `all`).
- `--shard i/n`: Shard offline dataset processing across cluster workers (e.g. `--shard 0/4`, `--shard 1/4`).

#### Examples
```bash
# Preview pipeline DAG execution plan
lmkd pipeline -c recipes/cross_tokenizer/precompute_vocab.yaml --dry-run

# Run offline vocab map generation
lmkd pipeline -c recipes/cross_tokenizer/precompute_vocab.yaml

# Run sharded dataset alignment across worker 0 of 4
lmkd pipeline -c recipes/pipeline/align_corpus.yaml --shard 0/4
```

---

## 5. Scaffolding & Development Commands

### `lmkd init`

Initialize a self-contained, production-grade template directory for a novel method or baseline.

```bash
lmkd init <name> [-t {dev,baseline}] [-d <domain>] [-f]
```

#### Arguments & Flags
- `name` *(required)*: Identifier for the new method (e.g. `my_adaptive_kd`).
- `-t`, `--target`: Target destination:
  - `dev` (default, shortcut `--dev`): Generates in `dev/<name>/` for rapid prototyping without touching core repository code.
  - `baseline` (shortcut `--baseline`): Generates in `src/baselines/<name>/` for official framework baselines.
- `-d`, `--domain`: Optional category subfolder (e.g. `self_distill`, `cross_tokenizer`, `embedding`).
- `-f`, `--force`: Overwrite existing files if directory already exists.

#### Generated Files & Directory Structure

When running `lmkd init my_adaptive_kd --dev`:
```
dev/my_adaptive_kd/
├── __init__.py           # Plugin entrypoint registering all sub-components
├── config.py             # Pydantic configuration schema
├── loss.py               # Criterion subclass with forward() & build_modules()
├── data.py               # Custom dataset transform / producer / stage
├── teacher.py            # Custom teacher signal / input builder
├── train.py              # Custom training algorithm / step strategy / callback
├── eval.py               # Custom evaluator / metric function
├── recipe.yaml           # Method-local declarative recipe
└── README.md             # Theoretical background & usage instructions

recipes/dev/my_adaptive_kd.yaml   # Central recipe referencing the dev package
tests/unit/test_my_adaptive_kd.py # Unit test verifying plugin registration & gradient probe
```

When targeting `--baseline` (`--target baseline`), the package is created under `src/baselines/[domain/]<name>`, the central recipe under `recipes/[domain/]<name>.yaml`, and the unit test under `tests/unit/test_<name>.py`.

#### Examples
```bash
# Scaffold experimental method under dev/
lmkd init my_adaptive_kd --dev

# Scaffold official baseline under src/baselines/
lmkd init novel_align_kd --baseline --domain cross_tokenizer
```

---

### `lmkd new`

Scaffold a single isolated plugin file, companion unit test, and minimal recipe.

```bash
lmkd new <kind> <name> [--domain <domain>]
```

#### Arguments & Flags
- `kind` *(required)*: Plugin type. Choices:
  - `criterion`: Distillation divergence loss.
  - `transform`: Per-sample data transform.
  - `producer`: Batch provider / replay buffer.
  - `stage`: Offline pipeline processing stage.
  - `step_strategy`: Custom gradient accumulation or optimization step.
  - `grad_transform`: Gradient modification hook (e.g. PCGrad, norm clipping).
  - `callback`: Training lifecycle callback.
  - `baseline` / `method`: Full baseline alias.
- `name` *(required)*: Identifier name for the component.
- `--domain`: Subdirectory category for recipe (default: `experimental`).

#### Examples
```bash
# Scaffold a custom loss function
lmkd new criterion confident_kl

# Scaffold a custom data transform
lmkd new transform filter_low_entropy
```

---

## 6. Quick Cheat Sheet

```bash
# 1. Sanity check loss calculation
lmkd try fkl

# 2. Inspect registered losses or transforms
lmkd list criterions
lmkd list transforms -v

# 3. Validate recipe configuration
lmkd check -c recipe.yaml
lmkd check -c recipe.yaml --preflight

# 4. View merged config
lmkd config -c recipe.yaml

# 5. Launch training with overrides
lmkd train -c recipe.yaml --set train.max_steps=500 optim.lr=1e-4

# 6. Evaluate checkpoint
lmkd eval -c recipe.yaml --ckpt ./outputs/checkpoint-500

# 7. Scaffold new experimental method
lmkd init custom_kd --dev
```
