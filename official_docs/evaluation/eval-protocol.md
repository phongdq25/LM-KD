# Evaluation Protocol & Best Practices

To guarantee that reported empirical gains reflect genuine algorithmic advances rather than decoding artifacts or cherry-picked seeds, LM-KD enforces a standardized evaluation protocol.

---

## 1. Decoding Hyperparameters

| Evaluation Target | Decoding Strategy | Temperature | Top-$p$ | Max New Tokens |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Benchmarks** (MMLU, GSM8K, IFEval) | Greedy Decoding | $0.0$ | $1.0$ | $2,048$ |
| **Long-CoT Reasoning** (MATH-500, AIME) | Greedy Decoding | $0.0$ | $1.0$ | $8,192$ |
| **Pass@$k$ Coding / Math Exploration** | Nucleus Sampling | $0.7$ | $0.95$ | $4,096$ |
| **Open-Ended Instruction** (AlpacaEval, MT-Bench)| Sampling | $0.7$ | $0.9$ | $2,048$ |

> [!IMPORTANT]
> Never evaluate deterministic benchmarks (e.g. GSM8K, MMLU) with stochastic sampling ($T > 0$) unless reporting Pass@$k$ over multiple sampled trajectories ($k \ge 8$).

---

## 2. Multi-Seed Replication Protocol

All benchmark tables published in the official docs are evaluated across **3 distinct random seeds**:
- Seeds: `42`, `123`, `456`.
- Report values as: $\text{Mean} \pm \sigma_{\bar{x}}$ (standard error of the mean).
- A baseline improvement is considered statistically significant only if $p < 0.05$ under a paired bootstrap re-sampling test ($B = 10,000$ iterations).

---

## 3. High-Throughput In-Training Evaluation with vLLM & Optimizer Offload

Multi-sample generation benchmarks (e.g. `math_passk` on MATH-500 with $n=4$ or $8$) require thousands of autoregressive forward steps per evaluation cycle. Evaluating with native Hugging Face (`engine: hf`) can take several hours per epoch.

### Accelerated Evaluation with `vllm_colocate`
LM-KD allows evaluating directly with vLLM during training epochs. By enabling `offload_optimizer: true`, AdamW optimizer states are moved to pinned host RAM during evaluation, allocating **75% of GPU memory** to vLLM's KV-Cache:

```yaml
callbacks:
  - type: evaluate
    every: epoch
    offload_optimizer: true          # Temporarily moves AdamW states to CPU RAM
    load_best_at_end: true
  - type: checkpoint
    every: epoch
    keep_best: 1
    metric: math_passk/math500/pass@1
    mode: max

eval:
  - type: math_passk
    when: epoch
    benchmarks: ["math500"]
    n: 4                             # Samples per problem
    ks: [1]
    grader: boxed_builtin            # Fast builtin math grading
    engine:
      type: vllm_colocate            # 10x-20x faster than native HF generate
      gpu_memory_utilization: 0.75
      enable_sleep_mode: true
```

*(Note: If the training recipe already defines a top-level `rollout: {type: vllm_colocate, ...}`, the evaluator automatically reuses that instance without needing an `engine:` block).*


