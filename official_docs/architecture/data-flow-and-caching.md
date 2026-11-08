# Data Flow & Teacher Caching

This document explains the runtime lifecycle of training data in LM-KD, from raw text ingestion to gradient accumulation, as well as the offline teacher caching subsystem.

---

## 1. End-to-End Training Data Flow

```mermaid
sequenceDiagram
    autonumber
    participant D as Data Producer
    participant B as Batch Collator
    participant T as Teacher Provider
    participant S as Student Model
    participant C as Distillation Criterion
    participant A as Loss Aggregator
    participant E as Distributed Engine

    D->>B: Stream text samples (Prompt, Response)
    B->>B: Tokenize, Pad/Pack, Apply loss_mask
    B->>T: Forward teacher input_ids
    alt Online Teacher
        T->>T: Compute teacher logits (no_grad, eval)
    else Precomputed Cache
        T->>T: Stream Top-K logits from batch cache columns
    end
    B->>S: Forward student input_ids
    S->>S: Student forward pass (autograd)
    S->>C: Emit student logits / hidden states
    T->>C: Emit teacher logits / hidden states
    C->>C: Compute chunked divergence numerators
    C->>A: Return LossOutput(terms, metrics)
    A->>A: Divide by global Z_tok / Z_seq
    A->>E: Backward pass & grad accumulation
```

---

## 2. Sequence Packing & Zero-Padding Efficiency

In standard padding, short sequences in a batch are padded with zeros to match the longest sequence. For multi-turn chats or instruction datasets with high sequence length variance, GPU compute is wasted on padding tokens.

LM-KD implements **attention-isolated sequence packing**:
- Multiple short sequences are concatenated end-to-end into a single contiguous context of length $L_{\max}$ (e.g., 4096 or 8192).
- FlashAttention-2 cumulative sequence length markers (`cu_seqlens`) prevent attention leakage across sequence boundaries.
- Packing is enabled with a single configuration flag:
  ```yaml
  data:
    packing: true
    max_length: 4096
  ```

---

## 3. Lazy Logit Chunking: Preventing Large-Vocab OOMs

Modern LLM vocabularies ($V > 150,000$) make materializing full $[B, S, V]$ tensors during forward/backward passes prohibitively expensive.

LM-KD solves this through **target-aligned lazy chunked logit views** ([`src/lmkd/models/logits_view.py`](../../src/lmkd/models/logits_view.py) and [`src/lmkd/models/chunked_head.py`](../../src/lmkd/models/chunked_head.py)):
1. Student hidden representations are projected to logits in chunks of sequence length $C$ (e.g., $C = 1024$) via `ChunkedHeadFunction`.
2. Divergence terms (such as $\sum P_T \log \frac{P_T}{P_S}$) are accumulated incrementally.
3. Intermediate logit chunks are discarded immediately, preserving GPU VRAM.

```python
from lmkd.models.logits_view import chunked_token_divergence
from lmkd.losses.functional.divergences import get_divergence

# Evaluates KL divergence chunk-by-chunk via LogitsView without full [B, S, V] VRAM allocation
fkl_fn = get_divergence("fkl", temperature=1.0)
loss_mask = ctx.supervised_mask("student")
kl_per_token = chunked_token_divergence(
    student=ctx.student_logits(),
    teacher=ctx.teacher.logits,
    fn=fkl_fn,
    mask=loss_mask,
    chunk_size=1024,
)
```

---

## 4. Offline Teacher Caching Pipeline

For static instruction corpora, running teacher inference repeatedly at every epoch or step wastes immense compute. LM-KD provides an offline pipeline stage to precompute and cache teacher representations to disk.

### Generating the Teacher Cache: `lmkd pipeline`
Create a pipeline configuration (`cache_pipeline.yaml`):
```yaml
pipeline:
  - stage: prepare_dataset
    data:
      train:
        type: hf_dataset
        path: databricks/databricks-dolly-15k
        split: train
    transforms:
      - type: tokenize_sft
        use_chat_template: true
  - stage: teacher_cache
    model: Qwen/Qwen2.5-14B-Instruct
    kind: topk
    k: 50
    dtype: bf16
    batch_size: 8
```

Execute the pipeline:
```bash
lmkd pipeline -c cache_pipeline.yaml
```

### Consuming the Cache in Distillation
Update your recipe to read from the cached teacher columns:
```yaml
teacher:
  provider: cache
  kind: topk
```

When running with cached teacher signals, training speed increases significantly, and GPU memory required for the teacher model is reduced to zero.
