# Debugging & Performance Profiling

This guide covers diagnostic tools, debugging workflows, and performance monitoring in LM-KD.

---

## 1. CLI Diagnostic Tools

LM-KD includes built-in diagnostic commands to verify implementations before launching distributed jobs:

```bash
# 1. Inspect architectural summary of a recipe (models, losses, data transforms)
lmkd explain -c recipes/cross_tokenizer/dskdv2.yaml

# 2. Check schema and parameter compatibility against Pydantic validation
lmkd check -c recipes/same_tokenizer/fkl.yaml

# 3. Dry-run a loss function in seconds with synthetic tensors and backward pass
lmkd try token_kd
```

---

## 2. Diagnosing `NaN` and Gradient Explosions

If training suddenly encounters `Loss: NaN` or non-finite gradient steps:

1. **Activate the Built-In `nan_guard` Callback:**
   Configure `nan_guard` to halt execution after repeated non-finite steps:
   ```yaml
   callbacks:
     - type: nan_guard
       patience: 3
   ```
2. **Enable PyTorch Anomaly Detection via Environment Variable:**
   Run with `TORCH_DETECT_ANOMALY=1` to pinpoint the exact forward/backward operation producing `NaN`:
   ```bash
   TORCH_DETECT_ANOMALY=1 lmkd train -c recipes/same_tokenizer/fkl.yaml
   ```
3. **Inspect Logit Temperatures:**
   Temperatures that are too low ($T < 0.2$) cause numerical underflow in `torch.log_softmax`. Ensure $T \ge 1.0$ for standard distillation.
4. **Verify Gradient Clipping:**
   Always ensure `grad.clip_norm: 1.0` is active under `grad:` in your recipe.

---

## 3. Profiling GPU Throughput & VRAM

### Measuring Tokens Per Second
LM-KD automatically tracks and logs effective throughput to the console and TensorBoard/W&B:
- **`tokens_per_sec`**: Total non-padded tokens processed across all GPUs per second.
- **`step_time`**: Latency per optimization step in seconds.

### Memory Optimization Tips
- **Zero Redundancy (ZeRO-2):** Inherit `recipes/_base_/engine_zero2.yaml` for multi-GPU training to partition optimizer states.
- **Micro-Batch Sizing:** Tune `train.micro_batch_size` so that each forward-backward fits comfortably in VRAM while keeping `train.batch_size` at the desired global effective batch size.
- **Activation Checkpointing:** Set `gradient_checkpointing: true` on `models.student` for large sequence lengths.
