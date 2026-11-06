# Quickstart: 10-Minute End-to-End Distillation

This guide walks you through running your first knowledge distillation experiment in **under 10 minutes** using a lightweight student model on a single GPU (or CPU).

By the end of this tutorial, you will:
1. Verify a distillation loss in seconds with synthetic tensors using `lmkd try`.
2. Validate recipe configurations using `lmkd check`.
3. Train a lightweight student model ([`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) distilled from [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B)) on [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) using Forward KL divergence.
4. Run evaluation with `lmkd eval` to measure token agreement and task accuracy.

---

## 1. Instant Sanity Check: `lmkd try`

Before downloading models or datasets, test your distillation criterion directly in memory with random synthetic tensors:

```bash
# Test token distillation loss in seconds
lmkd try token_kd

# Test standard cross-entropy task loss
lmkd try nll
```

Output:
```text
Testing loss 'token_kd' (status: supported)...
Success! Loss terms produced:
  - token_kd: value=2.1482, reduction=token_sum
```

---

## 2. Inspecting and Validating the Recipe

LM-KD uses declarative YAML recipes that specify datasets, teacher/student architectures, divergence losses, optimizers, and runtimes.

Inspect the minimal Forward KL recipe located at [`recipes/same_tokenizer/fkl.yaml`](../../recipes/same_tokenizer/fkl.yaml):

```bash
# View resolved configuration with all inheritance merged
lmkd config -c recipes/same_tokenizer/fkl.yaml

# Perform dry-run validation (checks imports, vocab sizes, parameter names, and schemas)
lmkd check -c recipes/same_tokenizer/fkl.yaml
```

If validation succeeds, you will see:
```text
plugins: 7
  - lmkd.losses.token_kd
  - lmkd.losses.sft
  - lmkd.data.transforms.tokenization
  - lmkd.teachers.providers
  - ...
OK
```

---

## 3. Running 10-Minute Training

Run a quick training run using our optimized CPU/single-GPU profile. We override the maximum steps to 50 for quick execution:

```bash
# Single GPU training with CLI overrides
lmkd train -c recipes/same_tokenizer/fkl.yaml \
    --set train.max_steps=50 \
          train.log_every=10 \
          train.output_dir=./outputs/quickstart_fkl
```

### What Happens Under the Hood:
1. **Model Instantiation:** LM-KD instantiates [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) as student and loads [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) in eval mode (FP16/BF16) as teacher.
2. **Batch Production:** The instruction dataset is streamed, padded, and tokenized into unified [`Batch`](../../src/lmkd/core/batch.py) objects.
3. **Teacher Pass:** Teacher logits are computed on the student's context. When running on low memory, chunked logit streaming is activated automatically to prevent OOM.
4. **Criterion Computation:** Forward KL:
   $$\mathcal{L}_{\text{FKL}} = D_{\text{KL}}(P_T \parallel P_S) = \sum_{v \in V} P_T(v) \log \frac{P_T(v)}{P_S(v)}$$
5. **Optimization:** Gradients are scaled by `num_items_in_batch` (total valid non-padding tokens) and accumulated.

Sample progress log:
```text
[Step 10/50] Loss: 2.312 | Teacher KL: 1.140 | SFT Loss: 1.172 | Tokens/s: 14,210 | VRAM: 4.2GB
[Step 20/50] Loss: 2.105 | Teacher KL: 0.982 | SFT Loss: 1.123 | Tokens/s: 14,480 | VRAM: 4.2GB
[Step 30/50] Loss: 1.954 | Teacher KL: 0.891 | SFT Loss: 1.063 | Tokens/s: 14,390 | VRAM: 4.2GB
[Step 40/50] Loss: 1.840 | Teacher KL: 0.812 | SFT Loss: 1.028 | Tokens/s: 14,410 | VRAM: 4.2GB
[Step 50/50] Loss: 1.765 | Teacher KL: 0.764 | SFT Loss: 1.001 | Tokens/s: 14,450 | VRAM: 4.2GB
✓ Training completed in 3m 42s. Checkpoint saved to ./outputs/quickstart_fkl/checkpoint-50
```

---

## 4. Evaluating the Distilled Student

Once training finishes, evaluate the distilled student checkpoint against downstream benchmarks:

```bash
# Evaluate checkpoint using recipe evaluation suite
lmkd eval -c recipes/same_tokenizer/fkl.yaml \
    --ckpt ./outputs/quickstart_fkl/checkpoint-50
```

Sample evaluation summary:
```text
================================================================================
Evaluation Results: Qwen2.5-0.5B (FKL Checkpoint 50)
================================================================================
Teacher Model:            Qwen/Qwen2.5-1.5B
Evaluation Split:         Dolly-15k (Test Subset)
--------------------------------------------------------------------------------
Metric                    Value        Baseline (Raw Student)   Teacher Reference
--------------------------------------------------------------------------------
Top-1 Token Agreement     68.4%        54.2%                    100.0%
Top-5 Token Agreement     89.1%        78.3%                    100.0%
Mean Forward KL           0.764        1.428                    0.000
ROUGE-L                   34.21        30.15                    38.92
Exact Match               28.5%        24.0%                    33.4%
================================================================================
```

---

## 5. Next Steps

- Explore the theoretical foundation: [Taxonomy of Knowledge Distillation](../concepts/taxonomy.md).
- Dive into core design principles: [Core Abstractions](../architecture/core-abstractions.md).
- Explore advanced baselines: [DistiLLM](../methods/same-tokenizer.md), [CSD](../methods/same-tokenizer.md), [DSKDv2](../methods/cross-tokenizer.md).
- Learn how to implement your own loss: [Developing a Custom Loss](../guides/developing-custom-loss.md).

