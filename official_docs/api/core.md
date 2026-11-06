# API Reference: Core Classes

Technical specifications and docstrings for core LM-KD data structures and runtimes.

---

## 1. `lmkd.core.batch.Batch`

The universal container for tensor and non-tensor batch communication.

### Class Definition
```python
class Batch(MutableMapping[str, Any]):
    """Universal batch contract.
    
    A flat dictionary-like mapping with namespaced keys (e.g. 'student.input_ids',
    'seq.loss_mask') and a 'meta' dict for non-tensor objects.
    """
```

### Key Methods & Properties
- `batch[key]`: Access tensor by namespaced string key.
- `batch.to(device, non_blocking=False)`: Migrates all contained tensors to target device.
- `batch.pin_memory()`: Pins contained CPU tensors for non-blocking asynchronous CUDA copy.
- `batch.num_tokens`: Total valid target tokens where `seq.loss_mask == 1`.
- `batch.batch_size`: Number of sequences in the batch ($B$).
- `batch.slice(start, end)`: Returns a sliced sub-batch without copying memory.

---

## 2. `lmkd.core.batch.FieldSpec`

Declares collation rules for individual batch tensor keys.

### Class Definition
```python
@dataclass(frozen=True)
class FieldSpec:
    key: str
    kind: Literal["token", "token_k", "scalar", "vector", "ragged", "meta"]
    dtype: torch.dtype | None = None
    pad_value: float | int = 0
    side: Literal["student", "teacher"] | None = None
    doc: str = ""
```

---

## 3. `lmkd.train.runtime.Runtime`

Manages initialized models, optimizers, distributed engines, and execution loops.

### Methods
- `fit() -> Runtime`: Runs the complete training lifecycle (`begin()` $\to$ steps $\to$ `end()`).
- `train_step(loss_fn=None) -> StepOutput`: Executes a single training step, accumulating gradients across micro-batches and taking an optimizer step.
- `evaluate() -> dict[str, float]`: Runs registered evaluation suites and returns a dictionary of metric scores.
- `save(tag: str)`: Saves current student weights and tokenizer to the output directory.
- `forward(batch: Batch) -> SideOutput`: Safe model forward pass through distributed wrappers.

