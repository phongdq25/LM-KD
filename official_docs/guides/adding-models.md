# Adding Models & Teachers

LM-KD supports any causal decoder or encoder model available on the Hugging Face Hub, local checkpoints, quantized models, and high-throughput external scorers like vLLM.

---

## 1. Adding Open-Weights Hugging Face Models

Any model compatible with `AutoModelForCausalLM` or `AutoModel` can be specified directly in the recipe under `models`:

```yaml
models:
  student:
    path: meta-llama/Llama-3.2-1B-Instruct
    dtype: bf16
    attn_implementation: flash_attention_2  # flash_attention_2, sdpa, or eager
    trust_remote_code: false
```

### Quantized Teachers (4-bit, 8-bit, FP8)
To fit large teachers (e.g., 70B models) onto limited GPU memory during training, set `dtype` in the teacher `ModelSpec`:
- `dtype: nf4`: 4-bit NormalFloat quantization via `bitsandbytes` (frozen teacher only).
- `dtype: int8`: 8-bit quantization via `bitsandbytes`.
- `dtype: fp8`: Float8 weight-only quantization via `torchao` (requires CUDA compute capability sm89+ / Ada or Hopper).

```yaml
models:
  teacher:
    path: meta-llama/Meta-Llama-3-70B-Instruct
    dtype: nf4  # or int8, fp8
    placement: same_device  # or device:1, cpu
```

---

## 2. Fast Teacher Scoring with vLLM

When serving large frozen teachers with maximum throughput, LM-KD integrates with vLLM via the `vllm_scorer` provider:

```yaml
models:
  teacher:
    path: meta-llama/Meta-Llama-3-70B-Instruct
    dtype: bf16

teacher:
  provider: vllm_scorer
  k: 20
  gpu_memory_utilization: 0.4
  tensor_parallel_size: 2
```

---

## 3. Custom Teacher Provider Implementation

To build custom teacher providers (e.g. streaming logprobs from a remote API or an external inference cluster), subclass `TeacherProvider` from [`src/lmkd/teachers/providers.py`](../../src/lmkd/teachers/providers.py):

```python
# plugins/custom_teacher.py
from __future__ import annotations
from typing import ClassVar
from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TEACHERS
from lmkd.models.forward import SideOutput, SideRequest
from lmkd.teachers.providers import TeacherProvider
from lmkd.teachers.signals import TeacherOutput, TeacherSignalSpec

@TEACHERS.register("custom_api_teacher")
class CustomAPITeacher(TeacherProvider):
    """Custom teacher provider querying an external service for teacher signals."""

    capabilities: ClassVar[frozenset[str]] = frozenset({"topk", "seq_logp"})

    class Config(BaseConfig):
        endpoint_url: str = "http://localhost:8000/v1/score"
        api_key_env: str = "TEACHER_API_KEY"

    def signals(
        self,
        batch: Batch,
        spec: TeacherSignalSpec | None,
        req: SideRequest | None,
        student: SideOutput | None,
    ) -> TeacherOutput | None:
        # Query remote service for token-level log probabilities
        out = TeacherOutput()
        # Populate out.topk_logp, out.topk_indices, or out.seq_logp
        return out
```

Reference the custom teacher provider in your recipe:
```yaml
imports:
  - ./plugins/custom_teacher.py

teacher:
  provider: custom_api_teacher
  endpoint_url: http://my-cluster:8000/score
```
