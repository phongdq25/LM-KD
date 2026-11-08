# Core Abstractions

LM-KD is built on a small, cohesive set of object-oriented contracts. Understanding these abstractions is essential for both applying existing methods and implementing new research algorithms.

---

## 1. The Universal Batch Container: `Batch` & `FieldSpec`

In traditional implementations, collators return arbitrary tuples or ad-hoc dictionaries with conflicting naming conventions. In LM-KD, all tensor and metadata communication uses [`Batch`](../../src/lmkd/core/batch.py).

### Batch Structure
A `Batch` is a dictionary-like container with namespaced keys (`"<namespace>.<field_name>"`) and a `meta` dictionary:
- `student.input_ids`: Integer tensor of student token IDs.
- `student.attention_mask`: Attention mask for student tokens.
- `seq.loss_mask`: 1 for supervised target tokens, 0 for prompts/padding.
- `teacher.input_ids`: (Optional) Tokens tokenized under the teacher's vocabulary.
- `aligner.mapping`: (Optional) Cross-tokenizer alignment index mapping.

### The Mandatory Target-Aligned Convention
> [!IMPORTANT]
> **Every per-token tensor has length $T \equiv \text{len}(\text{input-ids})$ of its side, where position $t$ describes the target token $\text{input-ids}[t]$.**
> Position 0 is never supervised. Only [`LogitsView`](../../src/lmkd/models/logits_view.py) shifts logits; no loss function or collator performs manual `[:, :-1]` slicing.

### Field Specification: `FieldSpec`
Collators pad, slice, and pin tensors according to declarative [`FieldSpec`](../../src/lmkd/core/batch.py) registrations:

```python
from lmkd.core.batch import FieldSpec, FIELDS
import torch

FIELDS.register(FieldSpec(
    key="student.input_ids",
    kind="token",
    dtype=torch.long,
    pad_value=0,
    side="student",
    doc="Autoregressive token IDs for student"
))
```

Supported field kinds:
- `"token"`: Padded along sequence length $T$.
- `"token_k"`: Top-$K$ values/indices along sequence length.
- `"scalar"`: Per-sequence scalar values (e.g., reward, sequence length).
- `"vector"`: Per-sequence feature vectors.
- `"ragged"`: Variable-length packed structures.
- `"meta"`: Arbitrary non-tensor Python objects.

---

## 2. The Distillation Loss Contract: `Criterion`

All loss functions inherit from [`Criterion`](../../src/lmkd/losses/base.py) (`torch.nn.Module`).

### Core Responsibilities
1. **Feature Declaration (`features()`):** Tells the engine exactly what tensors are needed (student/teacher logits, hidden layers, attention maps) so unused activations are never computed or retained in memory:
   ```python
   def features(self) -> FeatureRequest:
       return FeatureRequest(
           student=SideRequest(logits="view"),
           teacher=SideRequest(logits="view"),
       )
   ```
2. **Forward Computation (`forward()`):** Receives [`LossContext`](../../src/lmkd/losses/base.py) (which provides `ctx.batch`, `ctx.student`, `ctx.teacher`, `ctx.modules`, `ctx.state`, and target-aligned `ctx.student_logits()`), returning [`LossOutput`](../../src/lmkd/losses/base.py):
   ```python
   def forward(self, ctx: LossContext) -> LossOutput:
       # Target-aligned logits via LogitsView (full or chunked)
       s_logits = ctx.student_logits().full()
       t_logits = ctx.teacher.logits.full() if ctx.teacher else None
       diff = (s_logits - t_logits).pow(2).sum()
       mask = ctx.supervised_mask("student")
       return LossOutput(
           terms={"mse": LossTerm(value=(diff * mask.unsqueeze(-1)).sum(), reduction="token_sum")},
           metrics={"mean_diff": diff.detach().mean()}
       )
   ```
3. **Global Reduction Contract:** A `Criterion` returns raw **numerators** tagged with a `reduction` mode (`"token_sum"`, `"seq_sum"`, `"mean"`). The centralized [`LossAggregator`](../../src/lmkd/losses/aggregator.py) divides by the global denominator across all micro-batches and GPU ranks ($Z_{\text{tok}}$ or $Z_{\text{seq}}$).

---

## 3. Teacher Provider: `TeacherProvider`

[`TeacherProvider`](../../src/lmkd/teachers/providers.py) decouples *how* teacher supervision is produced from how the loss is computed:

- **Local Hugging Face Teacher:** Runs teacher forward passes on-device in `eval` mode with mixed precision and `torch.no_grad()`.
- **Precomputed Logit Cache:** Streams pre-extracted teacher logits or top-$K$ indices from disk (HDF5 / Arrow / Safetensors), skipping teacher inference during training.
- **Remote API Teacher:** Queries black-box proprietary teachers (e.g., Gemini, OpenAI) asynchronously for synthetic responses, reasoning chains, or logprobs.
- **Self-Teacher (EMA / Lagged):** Maintains an Exponential Moving Average of student weights or samples from lagged checkpoints for self-distillation algorithms.

---

## 4. Rollout Engine: `RolloutEngine`

For on-policy distillation methods (GKD, OPSD, SPIN), the [`RolloutEngine`](../../src/lmkd/rollout/base.py) manages autoregressive generation:

- **Colocated vLLM Engine:** Hosts a vLLM engine within the same GPU process, synchronizing updated student weights with minimal latency and high KV-cache throughput.
- **Server vLLM Engine:** Dedicated inference workers streaming generations across high-speed IPC or gRPC sockets for multi-node deployments.
- **Hugging Face Generation:** Lightweight fallback for single-GPU development and debugging without requiring CUDA-specific vLLM builds.

---

## 5. Execution Engine: `Engine` & `Trainer`

[`Engine`](../../src/lmkd/train/engine.py) manages hardware orchestration, precision scaling, and distributed communications:
- Configures **DeepSpeed ZeRO** (Stage 1, 2, or 3) or **FSDP2**.
- Wraps optimizers ([AdamW](../../src/lmkd/train/optim.py), [Muon](../../src/lmkd/train/optim.py), [TALAS](../../src/lmkd/train/optim.py)).
- Centralizes cross-GPU token normalization to guarantee mathematically identical gradient updates regardless of batch size partitioning.

