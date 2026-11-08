# Scenario 7: Self-Play Fine-Tuning (Adversarial Alignment)

This scenario establishes the common benchmarking protocol for iterative adversarial self-play, where a language model bootstraps its reasoning and instruction-following capabilities by playing against earlier iterations of itself without external reward models or human annotations.

---

## 1. Experimental Models

Evaluated across 3 foundation language models initialized from an identical SFT warm-up checkpoint:
- `Qwen/Qwen2.5-1.5B` (Initialized from SFT checkpoint `Qwen2.5-1.5B-ultrachat200k`)
- `google/gemma-2-2b` (Initialized from SFT checkpoint `gemma-2-2b-ultrachat200k`)
- `Qwen/Qwen3-4B` (Initialized from SFT checkpoint `Qwen3-4B-ultrachat200k`)

---

## 2. Datasets & Resources

### Warmup Fine-Tuning Corpus
- [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k): Standardized 1-epoch supervised warm-up establishing the base policy $M_0$.

### Self-Play Training Prompts
- [UCLA-AGI/SPIN_iter0](https://huggingface.co/datasets/UCLA-AGI/SPIN_iter0): 50,000 conversational prompts sampled from UltraChat50k.

### Evaluation Benchmarks (Test Sets)
Evaluated across 7 benchmarks spanning 4 core domains:
- **Math & Logic:**
  - `GSM8K`: Grade-school arithmetic — HF: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
  - `MATH`: Competition mathematics — HF: [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math)
- **Multi-Domain Knowledge:**
  - `MMLU`: General subject knowledge — HF: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)
  - `MMLU-Pro`: Challenging multi-choice reasoning — HF: [TIGER-AI-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-AI-Lab/MMLU-Pro)
- **Commonsense & Linguistic Reasoning:**
  - `HellaSwag`: Commonsense sentence completion — HF: [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)
  - `Winogrande`: Coreference resolution — HF: [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande)
- **Instruction Following (IF):**
  - `IFEval`: Verifiable instruction following — HF: [google/ifeval](https://huggingface.co/datasets/google/ifeval)

---

## 3. Evaluation Metrics

- **Primary Metrics:** Accuracy or normalized score in $[0, 1]$ across each of the 7 evaluation benchmarks.
- **Aggregated Metric:** Unweighted arithmetic mean (Average) across all 7 benchmark datasets.

---

## 4. Participating Baselines & Paper Links

| Baseline | Official Paper Reference | Venue / Record | Key Alignment Objective |
| :--- | :--- | :--- | :--- |
| **SPIN** | *Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models* (Chen et al.) | ICML 2024 · [PMLR](https://proceedings.mlr.press/v235/chen24j.html) · [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) | Minimax two-player game discriminating human data from student rollouts |
| **T-SPIN** | *Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs* (Wang et al.) | NeurIPS 2025 · [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a797b10ff20562b1ecee0d4e914c1c7-Abstract-Conference.html) · [arXiv:2601.08198](https://arxiv.org/abs/2601.08198) | Triplet self-play anchoring to proto-synthetic rollouts with historical advantages |
| **SPACE** | *SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for LLMs* (Wang et al.) | NeurIPS 2025 · [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5692c7dbc4abcaa50f9ce609819212e5-Abstract-Conference.html) · [arXiv:2512.07175](https://arxiv.org/abs/2512.07175) | Noise Contrastive Estimation binary classification stabilizing gap-based games |
| **SPPO** | *Self-Play Preference Optimization for Language Model Alignment* (Wu et al.) | ICLR 2025 · [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e48fa1c4f08fd1ae35d5df8352c3106d-Abstract-Conference.html) · [arXiv:2405.00675](https://arxiv.org/abs/2405.00675) | Constant-sum symmetric zero-sum game with soft win-rate regression |
| **S-SPPO** | *S-SPPO: Semantic-Calibrated Self-Play Preference Optimization* (Chen et al.) | ICML 2026 · [PMLR](https://proceedings.mlr.press/v306/chen26hc.html) · [arXiv:2606.01561](https://arxiv.org/abs/2606.01561) | Semantic gating & latent repulsion preventing overconfidence on semantic twins |
| **RSPO** | *RSPO: Regularized Self-Play Alignment of Large Language Models* (Tang et al.) | ICML 2026 · [PMLR](https://proceedings.mlr.press/v306/tang26ab.html) · [arXiv:2503.00030](https://arxiv.org/abs/2503.00030) | Regularized self-play optimization bounding policy drift with reference KL |

---

## 5. Common Generalized Train & Test Scenario

- **Self-Play Generation:** At round $t$, the opponent policy $M_{t-1}$ generates responses $y' \sim P_{M_{t-1}}(\cdot \mid x)$ for prompts $x \in \mathcal{D}_{\text{prompts}}$.
- **Minimax Game Optimization:** The main policy $M_t$ is trained to assign higher likelihood to ground-truth demonstrations $y$ than to opponent rollouts $y'$, converging toward a minimax Nash equilibrium.
- **Testing Phase:** Zero-shot and few-shot evaluation across all 7 benchmark tasks using standard evaluation harnesses.

---

## 6. Official Empirical Results

| Method | Math & Logic<br>GSM8K | Math & Logic<br>MATH | Multi-Domain<br>MMLU | Multi-Domain<br>MMLU-Pro | Reasoning<br>HellaSwag | Reasoning<br>Winogrande | IF<br>IFEval | Total<br>Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | | | | | | | | |
| SPIN | 0.616 | 0.161 | 0.577 | 0.260 | 0.646 | 0.655 | 0.170 | 0.441 |
| T-SPIN | 0.629 | 0.277 | 0.602 | 0.292 | 0.679 | 0.645 | 0.179 | 0.472 |
| SPACE | 0.636 | 0.284 | 0.603 | 0.297 | 0.680 | 0.646 | 0.192 | 0.477 |
| SPPO | 0.594 | 0.257 | 0.597 | 0.308 | 0.679 | **0.658** | 0.216 | 0.473 |
| S-SPPO | 0.607 | 0.270 | 0.601 | **0.315** | **0.694** | 0.653 | 0.202 | 0.477 |
| **RSPO** | 0.618 | 0.277 | 0.599 | 0.312 | 0.680 | 0.656 | **0.227** | **0.481** |
| **Gemma-2-2B** | | | | | | | | |
| SPIN | 0.386 | 0.195 | 0.555 | 0.279 | 0.727 | 0.692 | 0.360 | 0.456 |
| T-SPIN | 0.362 | 0.147 | 0.546 | 0.271 | 0.709 | 0.679 | **0.399** | 0.445 |
| SPACE | 0.376 | 0.148 | 0.554 | 0.274 | 0.723 | 0.691 | 0.320 | 0.441 |
| SPPO | 0.387 | 0.180 | 0.559 | **0.284** | 0.708 | 0.688 | 0.346 | 0.450 |
| S-SPPO | 0.380 | 0.176 | 0.554 | 0.276 | 0.712 | 0.691 | 0.351 | 0.449 |
| **RSPO** | 0.383 | 0.188 | 0.555 | 0.278 | 0.717 | **0.695** | 0.349 | **0.452** |
| **Qwen3-4B** | | | | | | | | |
| SPIN | 0.738 | 0.416 | 0.707 | 0.505 | 0.724 | 0.689 | 0.464 | 0.606 |
| T-SPIN | 0.829 | 0.440 | 0.706 | 0.503 | 0.722 | **0.696** | **0.527** | 0.632 |
| SPACE | 0.825 | 0.436 | 0.707 | 0.505 | 0.725 | 0.689 | 0.433 | 0.617 |
| **SPPO** | **0.839** | **0.443** | **0.708** | 0.507 | 0.726 | 0.688 | 0.431 | 0.620 |
| S-SPPO | 0.836 | 0.440 | 0.707 | **0.510** | **0.727** | 0.687 | 0.431 | 0.620 |
| RSPO | 0.833 | 0.435 | 0.705 | 0.504 | 0.725 | 0.692 | 0.454 | 0.621 |

