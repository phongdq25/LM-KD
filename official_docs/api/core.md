# API Reference: Core Classes

Technical specifications and docstrings for core LM-KD data structures and runtimes.

---

## 1. `lmkd.core.batch.Batch`

The universal container for tensor and non-tensor batch communication across pipeline stages, engines, and criteria.

### Class Definition
```python
class Batch(MutableMapping[str, Any]):
    """Universal batch contract.

    A flat dictionary-like mapping with namespaced keys (e.g. 'student.input_ids',
    'seq.loss_mask') and a 'meta' dict for non-tensor objects.
    """
```

### Key Methods & Properties
- `batch[key]`: Access tensor or object by namespaced string key (`"<ns>.<name>"`).
- `batch.to(device, non_blocking=True)`: Migrates all contained tensors to the target device.
- `batch.batch_size`: Number of sequences in the batch ($B$).
- `batch.ns(prefix)`: Returns a dict of items matching `<prefix>.<name>` with the prefix stripped (e.g., `batch.ns("student")`).
- `batch.select(idx)`: Returns a new `Batch` selecting rows along dimension 0 by an index tensor or list of ints.
- `batch.split(n_micro)`: Splits along dimension 0 into `n_micro` micro-batches.
- `batch.require(*keys)`: Asserts required batch keys are present, raising `KeyError` if any are missing.
- `batch.copy()`: Shallow copy of internal tensor mapping and meta dict.

*(Note: Target token counting is performed via `count_supervised(batch)` from `lmkd.train.step_strategies`.)*

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

Manages initialized models, optimizers, distributed engines, and training execution loops.

### Methods
- `fit() -> Runtime`: Runs the complete training lifecycle (`begin()` $\to$ steps $\to$ `end()`).
- `begin()`: Sets up distributed state, schedules, and invokes `on_train_begin` across callbacks.
- `next_batch() -> Batch`: Fetches the next batch from the data producer.
- `train_step(batch=None, loss_fn=None) -> StepOutput`: Executes a single training step, accumulating gradients across micro-batches and stepping the optimizer.
- `end()`: Finalizes training, runs final evaluations/callbacks, and shuts down engines.
- `evaluate(phase="always") -> dict[str, float]`: Runs registered evaluation suites and returns a dictionary of metric scores.
- `save(tag: str) -> Path`: Saves current student weights and tokenizer to the output directory.
- `unwrap_model() -> nn.Module`: Accesses the raw student module unwrapped from distributed/FSDP/ZeRO wrappers.
- `forward(batch, hidden=False, logits=True, grad=True) -> ModelView`: Safe model forward pass through distributed wrappers.
