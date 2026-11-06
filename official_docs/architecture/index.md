# LM-KD Architecture & Design Principles

LM-KD addresses the fragmentation, rigidity, and inefficiency of traditional knowledge distillation codebases. Rather than forcing users into separate research forks for each paper, LM-KD unifies distillation through an extensible, modular architecture.

---

## 1. Core Design Principles

<details>
<summary><b>More Detail</b></summary>

1. **Orthogonal Decomposition:** Every distillation run is decomposed along 6 independent axes (Context Origin, Batch Producer, Teacher Provider, Criterion, Step Strategy, and Optimizer). Altering one component requires zero modifications to the others.
2. **Unified Data Structures:** All data exchanged between producers, models, teachers, and loss criteria flows through a standardized tensor container: [`Batch`](../../src/lmkd/core/batch.py).
3. **Lazy Execution & Zero-Memory Overhead:** Heavy objects (such as full teacher logits) are streamed in chunks or queried lazily to prevent out-of-memory bottlenecks.
4. **Strict Registry & Plugin Extensibility:** Any custom loss function, teacher provider, or dataset can be implemented in a standalone Python file and registered with a single `@CRITERIONS.register` decorator without modifying core code.
5. **Declarative Composable Recipes:** Experiments are specified in YAML files supporting hierarchical inheritance (`_base_`), schema validation, and CLI overrides.
6. **Distributed First:** Transparent support for DeepSpeed ZeRO-1/2/3, PyTorch FSDP2, and decoupled vLLM rollouts, with centralized global token normalization (`num_items_in_batch`).

</details>

---

## 2. Architectural System Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                       LM-KD SYSTEM ARCHITECTURE                        │
├────────────────────────────────────────────────────────────────────────┤
│  [ Declarative YAML Recipe ] ──> [ Pydantic RecipeConfig & Validator ] │
│                                                │                       │
│                                                ▼                       │
│                                  [ Unified Plugin Registries ]         │
│                           (Criterions, Teachers, Producers, Engines)   │
│                                                │                       │
│                                                ▼                       │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │                    TRAINING & ROLLOUT PIPELINE                     │ │
│ │                                                                    │ │
│ │  [ Raw Data / Corpus ]                                             │ │
│ │    │                                                               │ │
│ │    ▼                                                               │ │
│ │  [ Batch Producer ] ──> (Tokenize / Pack / Align / Mask)           │ │
│ │    │                                                               │ │
│ │    ▼                                                               │ │
│ │  [ Unified Batch ]                                                 │ │
│ │    │ ├── student_input_ids, attention_mask, loss_mask              │ │
│ │    │ └── teacher_input_ids / aligner mapping                       │ │
│ │    ├───────────────────────────────────────────┐                   │ │
│ │    │                                           │                   │ │
│ │    ▼                                           ▼                   │ │
│ │  [ Student Forward Pass ]            [ Teacher Provider ]          │ │
│ │  (Causal LM / LoRA / FSDP)           (Local / vLLM / API / Cache)  │ │
│ │    │                                           │                   │ │
│ │    ▼                                           ▼                   │ │
│ │  Student Logits / Hidden             Teacher Logits / Hidden       │ │
│ │    └─────────────────────┬─────────────────────┘                   │ │
│ │                          ▼                                         │ │
│ │                 [ Loss Aggregator ]                                │ │
│ │                          │                                         │ │
│ │                          ▼                                         │ │
│ │             [ Step Strategy & Engine ]                             │ │
│ └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Section Navigation

- **[Core Abstractions](core-abstractions.md):** Understanding `Batch`, `FieldSpec`, `Criterion`, `TeacherProvider`, and `Engine`.
- **[Registry & Plugin System](registry-plugin-system.md):** How plugins work, lazy loading, and metadata tracking.
- **[Configuration System](configuration-system.md):** YAML syntax, inheritance via `_base_`, and schema validation.
- **[Data Flow & Caching](data-flow-and-caching.md):** End-to-end data pipeline, sequence packing, and offline teacher caching.
- **[Distributed Backends](distributed-backends.md):** DeepSpeed, FSDP, vLLM rollout co-location, and token normalization.

