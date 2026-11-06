# LM-KD Python API Reference

LM-KD provides both a command-line interface and a first-class programmatic Python API for driving experiments directly from Python scripts, interactive Jupyter notebooks, or custom training loops.

---

## 1. Quick Example: Training in 3 Lines of Python

Using `lmkd.quick`:

```python
import lmkd

# Launch distillation directly in Python without writing YAML
rt = lmkd.quick(
    student="Qwen/Qwen2.5-0.5B",
    teacher="Qwen/Qwen2.5-1.5B",
    data="./datasets/dolly_sample.jsonl",
    loss="fkl",
    max_steps=100,
    lr=2e-5,
    output_dir="./outputs/quick_fkl"
)

# Run the training loop
rt.fit()
```

---

## 2. Low-Level Control: Custom Training Loop

You can build a `Runtime` and drive every micro-step manually:

```python
import lmkd

# Build runtime with custom in-memory model and data
rt = lmkd.build(
    "recipes/same_tokenizer/fkl.yaml",
    overrides=["train.max_steps=500"],
    train_records=[{"prompt": "Hello", "response": "Hi there!"}]
)

rt.begin()
while not rt.done:
    # Executes one optimizer step (including micro-batches & all-reduce)
    step_output = rt.train_step()

    if rt.state.step % 50 == 0:
        eval_metrics = rt.evaluate()
        loss_val = float(step_output.metrics.get("loss", 0.0))
        print(f"Step {rt.state.step} | Loss: {loss_val:.4f} | Eval: {eval_metrics}")

rt.end()
rt.save("final_checkpoint")
```

---

## Section Navigation

- **[Core API Reference](core.md):** `Batch`, `FieldSpec`, `Registry`, `TrainState`, `Runtime`.
- **[Criterions & Losses API](criterions.md):** `Criterion`, `LossOutput`, `LossTerm`, `FeatureRequest`.
- **[CLI Reference](cli-reference.md):** Complete guide to all 10 `lmkd` CLI commands (`train`, `eval`, `check`, `try`, `pipeline`, `init`, etc.).

