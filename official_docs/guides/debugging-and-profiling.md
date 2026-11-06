# Debugging & Performance Profiling

This guide covers diagnostic tools, profiling workflows, and best practices for identifying performance bottlenecks in LM-KD.

---

## 1. CLI Diagnostic Tools

LM-KD includes built-in diagnostic commands to verify implementations before launching distributed jobs:

```bash
# 1. Inspect architectural summary of a recipe
lmkd explain -c recipes/cross_tokenizer/dskdv2.yaml

# 2. Check schema and parameter compatibility
lmkd check -c recipes/same_tokenizer/fkl.yaml

# 3. Dry-run a loss function in seconds with synthetic tensors
lmkd try fkl --params '{"temperature": 2.0}'
```

---

## 2. Diagnosing `NaN` and Gradient Explosions

If training suddenly crashes with `Loss: NaN` or `Grad Norm: Inf`:

1. **Enable Anomaly Detection:**
   Run with `--train.detect_anomaly true`. PyTorch will halt immediately at the exact forward or backward operation that produced `NaN`:
   ```bash
   lmkd train -c recipes/same_tokenizer/fkl.yaml --set train.detect_anomaly=true
   ```
2. **Inspect Logit Temperatures:**
   Temperatures that are too low ($T < 0.2$) cause numerical underflow in `torch.log_softmax`. Ensure $T \ge 1.0$ for standard distillation.
3. **Verify Gradient Clipping:**
   Always ensure `grad.clip_norm: 1.0` is active in the gradient configuration.

---

## 3. Profiling GPU Throughput & VRAM

### Measuring Tokens Per Second
LM-KD automatically logs effective throughput to the terminal and TensorBoard:
- **`tokens_per_sec`**: Total non-padded tokens processed across all GPUs per second.
- **`mfu` (Model FLOPs Utilization)**: Percentage of theoretical peak GPU FLOPs achieved.

### Enabling PyTorch Profiler
To capture detailed CUDA kernel execution traces and memory snapshots:

```yaml
train:
  profiler:
    enabled: true
    schedule:
      wait: 5
      warmup: 2
      active: 3
      repeat: 1
    output_dir: ./traces/
```

View the generated traces by opening Google Chrome and navigating to `chrome://tracing` or using Perfetto (`ui.perfetto.dev`).

