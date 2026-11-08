# Configuration System

LM-KD features a declarative, composable configuration system built on Pydantic and YAML. It allows researchers to manage complex training topologies, distributed backends, and divergence criteria with zero code duplication.

---

## 1. Anatomy of a Recipe

A recipe is a single YAML file describing an entire experiment. A standard recipe consists of top-level sections:

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
  padding: dynamic
  pad_to_multiple_of: 8

teacher:
  provider: hf_frozen
  input: same

producer:
  type: dataset

algorithm: standard
step:
  type: standard

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

optim:
  lr: 2.0e-5
  optimizers:
    default:
      type: adamw
  schedulers:
    - type: cosine
      warmup_ratio: 0.03

grad:
  clip_norm: 1.0

engine:
  type: local
  device: auto
  mixed_precision: bf16

train:
  output_dir: outputs/fkl
  epochs: 3
  batch_size: 16
  micro_batch_size: 2
  log_every: 10
```

---

## 2. Multi-File Inheritance (`_base_`)

Recipes can inherit from one or more base configuration files using the `_base_` key:
- Base configurations are merged recursively from left to right.
- The child file overrides any keys defined in parent configurations.
- Relative paths are resolved relative to the directory of the current recipe.

Common base templates in [`recipes/_base_/`](../../recipes/_base_/):
- `engine_zero1.yaml` / `engine_zero2.yaml` / `engine_zero3.yaml`: DeepSpeed ZeRO presets.
- `optim_muon_talas.yaml`: Modern Muon/AdamW optimizer setups.
- `eval_math.yaml`: Standardized evaluation pipeline on GSM8K and MATH.

---

## 3. Command-Line Dot-Key Overrides

Any value in the configuration hierarchy can be overridden directly from the CLI using `--set` without modifying the YAML file:

```bash
# Override learning rate, batch size, and output directory
lmkd train -c recipes/same_tokenizer/fkl.yaml \
    --set optim.lr=5e-5 \
          train.batch_size=32 \
          train.micro_batch_size=4 \
          train.output_dir=./outputs/exp_high_lr
```

CLI overrides support:
- Dotted paths: `--set section.key=value`
- Native Python primitives (`int`, `float`, `bool`, `str`).
- String literals and numbers:
  ```bash
  --set loss.terms.1.params.divergence=rkl loss.terms.1.params.temperature=2.0
  ```

---

## 4. Validating and Inspecting Configurations

### Inspecting the Resolved Config: `lmkd config`
Print the final, fully merged configuration to verify all inheritance and overrides:

```bash
lmkd config -c recipes/same_tokenizer/fkl.yaml --set optim.lr=1e-4
```

### Static Recipe Validation: `lmkd check`
Verify whether the recipe contains schema violations, invalid types, or missing plugin imports before allocating cluster GPUs:

```bash
lmkd check -c recipes/same_tokenizer/fkl.yaml
```

If any required parameters are missing or an invalid key is present, `lmkd check` immediately halts with a precise traceback pointing to the invalid YAML line.

