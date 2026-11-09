# Parameter-Efficient Distillation (LoRA)

When GPU VRAM is limited, training all parameters of the student model can be cost-prohibitive. LM-KD natively integrates Low-Rank Adaptation (LoRA) through Hugging Face PEFT.

---

## 1. Configuring Student LoRA in Recipes

To distill into a LoRA adapter rather than full model weights, configure the `lora` block inside `models.student`:

```yaml
models:
  student:
    path: meta-llama/Llama-3.2-1B-Instruct
    dtype: bf16
    gradient_checkpointing: true
    lora:
      r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj
      bias: none
```

> [!TIP]
> If a model has tied word embeddings (`tie_word_embeddings: true`), avoid adding `lm_head` or `embed_tokens` to `target_modules`. Stick to attention and MLP projection layers for stability and optimal memory usage.

### Benefits
- **VRAM Savings:** Optimizer states and gradients are maintained only for the adapter parameters ($< 1\%$ of model weights).
- **Faster Checkpointing:** Checkpoints contain only adapter weights (~50MB instead of multiple gigabytes).

---

## 2. Pairing Student LoRA with a Quantized Frozen Teacher

For maximum memory efficiency, combine a `bf16` LoRA student with a 4-bit (`nf4`) or 8-bit (`int8`) frozen teacher. This allows distilling from large teachers (e.g., 70B) on consumer GPUs:

```yaml
models:
  student:
    path: meta-llama/Llama-3.2-1B-Instruct
    dtype: bf16
    lora:
      r: 16
      lora_alpha: 32
  teacher:
    path: meta-llama/Meta-Llama-3-70B-Instruct
    dtype: nf4  # 4-bit NF4 via bitsandbytes
    placement: same_device

teacher:
  provider: hf_frozen
  input: same
```

*(Note: In LM-KD, `dtype: nf4` / `int8` is designed for frozen teacher models, keeping student gradients and optimizer states numerical stability in `bf16` or `fp16`.)*

---

## 3. Merging LoRA Weights After Distillation

After training finishes, merge the learned adapter weights back into the base model for standalone deployment using standard PEFT APIs:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype="auto")
model = PeftModel.from_pretrained(base_model, "./outputs/lora_distill/checkpoint-2000/student")
merged_model = model.merge_and_unload()

output_path = "./outputs/merged_distilled_student"
merged_model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained("./outputs/lora_distill/checkpoint-2000/student")
tokenizer.save_pretrained(output_path)
```

Alternatively, run as a one-line command:

```bash
python -c "from peft import PeftModel; from transformers import AutoModelForCausalLM, AutoTokenizer; base = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct', torch_dtype='auto'); model = PeftModel.from_pretrained(base, './outputs/lora_distill/checkpoint-2000/student'); model.merge_and_unload().save_pretrained('./outputs/merged_distilled_student'); AutoTokenizer.from_pretrained('./outputs/lora_distill/checkpoint-2000/student').save_pretrained('./outputs/merged_distilled_student')"
```

---

## 4. Zero-Latency LoRA Rollout & Evaluation with vLLM

When executing on-policy distillation (GKD, OPSD, GRPO) or periodic generation evaluation (`math_passk`, `generation_rouge`), pairing LoRA with `vllm_colocate` provides massive VRAM savings and near-zero weight synchronization overhead.

### Why LoRA Eliminates OOM on a Single GPU
In Full Parameter training, PyTorch holds ~14 GB model weights + ~56 GB AdamW FP32 states ($m, v$), leaving virtually no VRAM for vLLM on an 80 GB GPU.

With LoRA:
- **Base Model (BF16):** Completely frozen in memory (~14 GB for a 7B model).
- **LoRA Adapter ($A, B$):** Only ~20–50 MB.
- **AdamW States:** Maintained only for adapter parameters (< 0.2 GB).
- **Static VRAM Total:** **~28.3 GB** (Trainer + vLLM base weights combined). Over **50 GB VRAM** remains available exclusively for vLLM KV-Cache!

| Component | Full Parameters (7B BF16) | LoRA Adapter (`hot_load`) |
| :--- | :---: | :---: |
| **Model Weights (Trainer)** | 14 GB | 14 GB |
| **Adapter Weights (Trainer)** | — | **~0.05 GB** |
| **Optimizer States (AdamW)** | **~56 GB** | **~0.2 GB** |
| **Base Model (vLLM)** | 14 GB | 14 GB |
| **Total Static VRAM** | **~84 GB** *(OOM risk on 80 GB)* | **~28.3 GB** *(Safe on 40/80 GB)* |

### Synchronization Modes in `vllm_colocate`

1. **`lora_mode: hot_load` (Recommended):**
   vLLM natively maintains base weights while dynamically applying adapters via `LoRARequest`. LM-KD saves the updated adapter (~30 MB, <0.05s) and pushes the request to vLLM. No base model copying or merging occurs:
   ```yaml
   rollout:
     type: vllm_colocate
     gpu_memory_utilization: 0.65
     enable_sleep_mode: true
     lora_mode: hot_load
   ```

2. **`lora_mode: merge` (In-Process GPU-to-GPU Pointer Transfer):**
   Trainer temporarily merges the adapter into base parameters (`merge_adapter()`), streams the merged tensors directly to vLLM in-process (`model.load_weights()`) without crossing the PCIe bus, and unmerges before the next training step:
   ```yaml
   rollout:
     type: vllm_colocate
     gpu_memory_utilization: 0.60
     enable_sleep_mode: true
     lora_mode: merge
   ```

