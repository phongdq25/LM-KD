# Prompt & Chat Templates

Accurate formatting of role delimiters (system, user, assistant) is vital for knowledge distillation. A mismatch between the student's expected chat template and the teacher's formatting degrades generation quality and corrupts distillation logits.

---

## 1. Supported Model Families & Formats

### 1. ChatML (Qwen Family)
Used by `Qwen/Qwen2.5`, `Qwen/Qwen3`:
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Solve x^2 - 4 = 0.<|im_end|>
<|im_start|>assistant
x = 2 or x = -2.<|im_end|>
```

### 2. Llama-3 Header Format
Used by `meta-llama/Llama-3.1`, `meta-llama/Llama-3.2`:
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024<|eot_id|><|start_header_id|>user<|end_header_id|>
What is photosynthesis?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### 3. Mistral / Zephyr Format
Used by `mistralai/Mistral-7B-Instruct`:
```text
<s>[INST] What is photosynthesis? [/INST] Photosynthesis is...</s>
```

### 4. DeepSeek-R1 (Thinking Tag Delimiters)
Used by `deepseek-ai/DeepSeek-R1` and distilled variants:
```text
<｜begin of sentence｜><｜User｜>Find all prime numbers between 1 and 10.<｜Assistant｜><think>
Let's list numbers from 1 to 10: 2, 3, 5, 7.
</think>
The prime numbers are 2, 3, 5, and 7.<｜end of sentence｜>
```

---

## 2. Dynamic Hugging Face Chat Templates

LM-KD prioritizes native Jinja2 chat templates defined in each model's `tokenizer_config.json`:

```yaml
data:
  transforms:
    - type: tokenize_sft
      use_chat_template: true  # Automatically invokes tokenizer.apply_chat_template()
      chat_template_kwargs:
        add_generation_prompt: true
```

When distilling cross-family models (e.g., Qwen teacher to Llama student), the raw text prompt is passed to the student and teacher independent tokenizers, each applying its respective native chat template.

