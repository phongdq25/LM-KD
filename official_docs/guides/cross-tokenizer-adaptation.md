# Cross-Tokenizer Adaptation Guide

Cross-tokenizer distillation transfers representations and knowledge between models from different architectural families with divergent vocabularies (e.g., [`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B) with 152k tokens $\to$ [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) with 128k tokens).

---

## 1. Required Components

A cross-tokenizer recipe requires three synchronized components:
1. **Dual Tokenizer Producer:** Tokenizes raw text independently with student and teacher tokenizers.
2. **Token Aligner:** Computes an alignment mapping between student subword sequences and teacher subword sequences.
3. **Cross-Tokenizer Criterion & Projector:** Maps differing hidden dimensions ($d_S \neq d_T$) into a shared space.

---

## 2. Configuring the Aligner

LM-KD provides built-in aligners in [`src/lmkd/data/aligners/`](../../src/lmkd/data/aligners/):

- `dtw_edit`: Dynamic Time Warping based on subword string edit distance.
- `parent_groups`: Aggregates subwords that share the same root whitespace/word boundary.
- `exact_token`: Maps exact string character spans.

Example configuration in recipe (`data.transforms`):
```yaml
data:
  transforms:
    - type: align
      aligner:
        type: dtw_edit
        similarity_threshold: 0.75
        distance_metric: levenshtein
```

---

## 3. Learnable Projectors

When student hidden dimension $d_S$ does not equal teacher hidden dimension $d_T$, cross-tokenizer criteria register auxiliary projector modules (such as `LinearProjector`, `MLPProjector`, or `DSLProjector` from `lmkd.models.modules`) in `build_modules()`:

```python
from lmkd.models.modules import MLPProjector

# Inside Criterion.build_modules(self, info: ModelPairInfo, modules: ModuleRegistry):
proj_cfg = MLPProjector.Config(
    in_dim=info.student_hidden,
    out_dim=info.teacher_hidden,
    hidden_dim=4096,
    act="gelu",
)
self.register_aux_module(modules, "proj", MLPProjector(proj_cfg), group="projector")
```

In your recipe, optimizer parameters for auxiliary projectors are configured under `optim.groups`:

```yaml
optim:
  groups:
    - name: projector
      match:
        group: projector
      lr: 0.0005
    - name: default
      match: {}
  optimizers:
    default:
      type: adamw
  lr: 2.0e-5
```

During training, the projector weights are updated alongside the student model.

---

## 4. Full Cross-Tokenizer Recipe Example

```yaml
# recipes/cross_tokenizer/dskdv2.yaml
_base_:
  - ../_base_/engine_zero2.yaml

models:
  student:
    path: meta-llama/Llama-3.2-1B
    dtype: bf16
    task: causal_lm
  teacher:
    path: Qwen/Qwen2.5-7B
    dtype: bf16
    task: causal_lm
    placement: same_device

data:
  train:
    type: jsonl
    path: datasets/same_cross_tokenizer_data/dolly/train.jsonl
  transforms:
    - type: tokenize_sft
      use_chat_template: false
      teacher: true
      teacher_prompt_text: same
    - type: token_offsets
      teacher: true
    - type: align
      aligner:
        type: dtw_edit
        similarity_threshold: 0.75
  max_length: 2048

teacher:
  provider: hf_frozen
  input: pretokenized

loss:
  aggregator:
    type: kd_mix
    kd_ratio: 0.8
    ce_alias: nll
  terms:
    - criterion: nll
      weight: 1.0
    - criterion: dskdv2
      weight: 1.0
      alias: kd
      params:
        temperature: 2.0
        t2s_agreement: 0.0

optim:
  groups:
    - name: projector
      match:
        group: projector
      lr: 0.0005
    - name: default
      match: {}
  optimizers:
    default:
      type: adamw
  lr: 2.0e-5
```

