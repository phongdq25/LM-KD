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

