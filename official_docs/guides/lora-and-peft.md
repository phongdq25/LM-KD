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
