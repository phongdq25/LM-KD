# Parameter-Efficient Distillation (LoRA & QLoRA)

When GPU VRAM is severely limited, training all parameters of the student model can be cost-prohibitive. LM-KD natively integrates Low-Rank Adaptation (LoRA) and 4-bit Quantized LoRA (QLoRA) through Hugging Face PEFT.

---

## 1. Configuring LoRA in Recipes

To distill into a LoRA adapter rather than full model weights, configure the `peft` block in your recipe:

```yaml
models:
  student:
    path: meta-llama/Llama-3.2-1B-Instruct
    dtype: bf16
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

### Benefits
- **VRAM Savings:** Optimizer states and gradients are maintained only for the adapter parameters (typically $< 1\%$ of model weights).
- **Faster Checkpointing:** Checkpoints contain only adapter weights (~50MB instead of multiple gigabytes).

---

## 2. QLoRA Distillation (4-Bit Base Student)

For maximum memory efficiency, freeze the base student model in 4-bit NormalFloat (NF4) and train 16-bit LoRA adapters:

```yaml
models:
  student:
    path: meta-llama/Llama-3.2-3B-Instruct
    quantization: bitsandbytes_4bit
    lora:
      r: 32
      lora_alpha: 64
```

---

## 3. Merging LoRA Weights After Distillation

After training finishes, merge the learned adapter weights back into the base model for low-latency standalone deployment using standard PEFT APIs:

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

