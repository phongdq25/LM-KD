# Code Style & Engineering Standards

To maintain high code quality and architectural integrity across the repository, all contributions must adhere to the following standards.

---

## 1. Automated Formatting & Linting

LM-KD uses **Ruff** for linting and code formatting:

```bash
# Check for linting violations
ruff check .

# Automatically fix lint issues
ruff check --fix .

# Auto-format codebase
ruff format .
```

### Formatting Rules
- **Line Length:** 120 characters maximum.
- **Python Version:** Python 3.10+ semantics. Always include `from __future__ import annotations` at the top of every module.
- **Modern Type Hints:** Use native union syntax (`str | Path | None` instead of `Optional[Union[str, Path]]`).

---

## 2. Inviolable Architectural Rules

When writing plugins or editing core modules, the following rules are strictly enforced:

### Rule 1: The Target-Aligned Convention
> [!IMPORTANT]
> Every per-token tensor has length $T \equiv \text{len}(\text{input-ids})$ of its side. Position $t$ describes the target token $\text{input-ids}[t]$. Position 0 is never supervised. Only [`LogitsView`](../../src/lmkd/models/logits_view.py) shifts logits; never slice `[:, :-1]` manually inside a Criterion or Collator.

### Rule 2: Criterions Never Call Models Directly
A [`Criterion`](../../src/lmkd/losses/base.py) only accepts pre-extracted student and teacher tensors (`SideOutput`, `TeacherOutput`). It never invokes `student(input_ids)`. This guarantees safety when models are wrapped in DeepSpeed ZeRO-3 or FSDP sharding.

### Rule 3: Centralized Global Normalization
Never divide loss tensors by `batch_size` or sequence length inside a Criterion. Return raw unnormalized numerators in [`LossTerm`](../../src/lmkd/losses/base.py) with `reduction="token_sum"`, `"seq_sum"`, or `"mean"`. The framework's centralized aggregator normalizes by global cross-rank denominators.

