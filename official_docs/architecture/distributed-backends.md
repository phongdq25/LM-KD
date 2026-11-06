# Distributed Backends & Parallelism

Scaling LLM knowledge distillation across high-performance multi-GPU clusters requires careful orchestration of distributed strategies, memory partitioning, and exact token normalization.

LM-KD natively supports **DeepSpeed ZeRO**, **PyTorch FSDP2**, and **Ray-decoupled rollouts**.

---

## 1. Supported Distributed Backends

```mermaid
flowchart TD
    Engine["Engine / Distributed Runtime"]
    Engine --> DS["DeepSpeed ZeRO"]
    Engine --> FSDP["PyTorch FSDP2"]
    Engine --> Ray["Ray + verl Engine"]

    DS --> DS1["ZeRO-1: Optimizer State Sharded"]
    DS --> DS2["ZeRO-2: Optimizer + Grad Sharded (+ CPU Offload)"]
    DS --> DS3["ZeRO-3: Full Sharding (Weights + Grads + Optim)"]

    FSDP --> FSDP_A["Zero-Redundancy Sharding"]
    FSDP --> FSDP_B["Direct PyTorch Native Support"]

    Ray --> Ray1["Decoupled Rollout Workers (vLLM)"]
    Ray --> Ray2["Actor-Based Training Pipeline"]
```

### DeepSpeed ZeRO-1, 2, and 3
LM-KD integrates DeepSpeed through standardized engine templates in [`recipes/_base_/deepspeed/`](../../recipes/_base_/deepspeed/):
- **ZeRO-1:** Partitions optimizer states. Ideal when student and teacher comfortably fit in GPU memory.
- **ZeRO-2 + Offload:** Partitions optimizer states and gradients, optionally offloading optimizer states to host RAM. Recommended for distilling 7B–14B models on consumer GPUs.
- **ZeRO-3:** Fully shards student parameters, gradients, and optimizer states across all ranks. Enables training models up to 70B+ parameters.

Example configuration in recipe (layering ZeRO-2 presets):
```yaml
_base_:
  - ../_base_/engine_zero2.yaml

train:
  batch_size: 64
  micro_batch_size: 4
  output_dir: outputs/my_zero2_run
```

Launch with the multi-GPU launcher script:
```bash
NUM_GPUS=4 bash scripts/train_multigpu_deepspeed.sh recipes/same_tokenizer/fkl.yaml \
    --set train.batch_size=64 train.micro_batch_size=4
```

### PyTorch FSDP2 (Fully Sharded Data Parallel)
Native PyTorch distributed backend providing efficient forward/backward parameter prefetching and integration with PyTorch 2.x compilation (`torch.compile`).

---

## 2. Global Token Normalization (`num_items_in_batch`)

A common flaw in distributed LLM training implementations is **naive micro-batch loss averaging**.

### The Problem with Naive Averaging
In causal language modeling, different micro-batches contain different numbers of valid (non-padded, supervised) response tokens. If micro-batch 1 has 1,200 tokens and micro-batch 2 has 300 tokens:
$$\text{Naive Mean} = \frac{1}{2} \left( \frac{\sum_{t \in B_1} \ell_t}{1200} + \frac{\sum_{t \in B_2} \ell_t}{300} \right)$$
In this scenario, tokens in $B_2$ are artificially weighted **4 times more** than tokens in $B_1$, distorting the true gradient of the dataset distribution.

### The LM-KD Solution: Centralized Global Normalization
LM-KD strictly computes:
1. Each micro-batch computes the **sum** of loss numerators: $S_m = \sum_{i \in B_m} \ell_i$.
2. The number of valid supervised tokens $N_m$ in the micro-batch is counted.
3. Before the optimizer step, an `all_reduce` gathers the total valid tokens across all micro-batches and GPU ranks:
   $$Z_{\text{tok}} = \sum_{\text{rank}=1}^R \sum_{m=1}^M N_{m, \text{rank}}$$
4. The global gradient update is normalized exactly:
   $$\mathcal{L}_{\text{global}} = \frac{1}{Z_{\text{tok}}} \sum_{\text{rank}=1}^R \sum_{m=1}^M S_{m, \text{rank}}$$

This ensures that gradient updates are **numerically identical** regardless of batch size per GPU, gradient accumulation steps, or rank count.

---

## 3. Decoupled Rollout with Ray & vLLM

For on-policy algorithms (such as OPSD, GKD, and SPIN), generating autoregressive sequences on training GPUs causes severe pipeline bubbles.

LM-KD supports a **decoupled rollout architecture**:
1. **Training Cluster:** Runs DeepSpeed ZeRO-2 or FSDP across training GPUs, computing backward passes.
2. **Rollout Cluster:** Runs high-throughput vLLM instances dedicated entirely to generation with optimized PagedAttention KV-caches.
3. **Weight Sync:** After $K$ optimizer steps, student weights are broadcast asynchronously to the vLLM engine via NCCL or shared memory (`lmkd.rollout.weight_sync`).

