# Tutorial: Cross-Tokenizer Distillation (Qwen to OPT)

In this tutorial, you will transfer knowledge from [`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B) ($|\mathcal{V}| = 152,064$) into [`facebook/opt-1.3b`](https://huggingface.co/facebook/opt-1.3b) ($|\mathcal{V}| = 50,272$) using **DSKDv2** and **Dynamic Time Warping (DTW)** subword alignment.

---

## 1. The Challenge

Qwen uses a Byte-Pair Encoding (BPE) vocabulary of 152,064 tokens, whereas OPT uses a 50,272-token vocabulary. A single English word like `"disproportionate"` may be split into 2 tokens by Qwen (`["disproportion", "ate"]`) but 4 tokens by OPT (`["dis", "pro", "portion", "ate"]`).

DSKDv2 solves this through:
1. Subword DTW to map sequence token indices.
2. Latent MLP projection mapping OPT's 2048-dim hidden space to Qwen's 3584-dim hidden space.

---

## 2. The Recipe Configuration

Create `tutorial_cross_tok.yaml`:

```yaml
_base_:
  - recipes/_base_/engine_zero2.yaml

models:
  student:
    path: facebook/opt-1.3b
    dtype: bf16
    task: causal_lm
  teacher:
    path: Qwen/Qwen2.5-7B
    dtype: bf16
    task: causal_lm
    placement: same_device

teacher:
  provider: hf_frozen
  input: pretokenized

data:
  train:
    type: hf_dataset
    path: databricks/databricks-dolly-15k
    split: train
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

loss:
  aggregator:
    type: kd_mix
    kd_ratio: 0.5
    ce_alias: nll
  terms:
    - criterion: nll
      weight: 0.5
    - criterion: dskdv2
      weight: 0.5
      alias: kd
      params:
        temperature: 1.0

optim:
  lr: 2.0e-5
  optimizers:
    default:
      type: adamw

train:
  max_steps: 1000
  batch_size: 16
  micro_batch_size: 2
  output_dir: ./outputs/opt_dskdv2
```

---

## 3. Running the Training Job

Validate the configuration:
```bash
lmkd check -c tutorial_cross_tok.yaml
```

Launch distillation:
```bash
lmkd train -c tutorial_cross_tok.yaml
```

During training, observe `loss_rep` (hidden representation alignment) and `loss_soft` (projected probability matching) decreasing in your logs:
```text
[Step 100/1000] Loss: 1.821 | Rep Loss: 0.245 | Soft Loss: 1.120 | SFT: 0.456
[Step 500/1000] Loss: 1.214 | Rep Loss: 0.112 | Soft Loss: 0.784 | SFT: 0.318
[Step 1000/1000] Loss: 0.942 | Rep Loss: 0.065 | Soft Loss: 0.612 | SFT: 0.265
```

---

## 4. Evaluating the Distilled Student

Evaluate on the Dolly test split:
```bash
lmkd eval -c tutorial_cross_tok.yaml \
    --ckpt ./outputs/opt_dskdv2/checkpoint-1000
```

Notice the significant jump in downstream generation quality: standard SFT on OPT-1.3B reaches **20.4% average score**, while DSKDv2 distillation boosts the student to **24.58% average score**!

