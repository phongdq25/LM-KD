# Adding Models & API Teachers

LM-KD supports any causal decoder or encoder model available on the Hugging Face Hub, local checkpoints, and remote black-box API teachers.

---

## 1. Adding Open-Weights Hugging Face Models

Any model compatible with `AutoModelForCausalLM` or `AutoModel` can be specified directly in the recipe:

```yaml
models:
  student:
    path: meta-llama/Llama-3.2-1B-Instruct
    dtype: bf16
    attn_implementation: flash_attention_2  # flash_attention_2, sdpa, or eager
    trust_remote_code: false
```

### Quantized Teachers
To fit large teachers (e.g., 70B models) onto limited GPU memory during training, configure 4-bit or 8-bit quantization via `bitsandbytes`:

```yaml
models:
  teacher:
    path: meta-llama/Meta-Llama-3-70B-Instruct
    quantization: bitsandbytes_4bit  # or bitsandbytes_8bit
    dtype: bf16
```

---

## 2. Adding a Black-Box API Teacher

When the teacher model cannot be run locally (e.g., proprietary models such as Google Gemini or OpenAI GPT-4o), supervision takes the form of **synthetic data rollouts** or **logprob queries**.

### Setting Up the API Teacher Provider
LM-KD includes an asynchronous API provider in [`src/lmkd/teachers/providers.py`](../../src/lmkd/teachers/providers.py):

```yaml
teacher:
  provider: remote_api
  api_backend: gemini  # or openai, anthropic
  model_name: gemini-1.5-pro
  api_key_env: GEMINI_API_KEY
  rate_limit_qps: 10
  cache_responses: true
  cache_dir: ./caches/gemini_teacher
```

### Custom API Provider Implementation
You can implement custom endpoints by inheriting from `TeacherProvider`:

```python
from lmkd.core.registry import TEACHERS
from lmkd.teachers.providers import TeacherProvider
from lmkd.teachers.signals import TeacherOutput

@TEACHERS.register("custom_api")
class CustomAPITeacher(TeacherProvider):
    def get_signals(self, batch):
        # Query remote API endpoint asynchronously
        responses = self.client.batch_generate(batch.prompts)
        return TeacherOutput(text=responses)
```

