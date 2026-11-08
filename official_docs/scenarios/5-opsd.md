# Scenario 5: On-Policy Self-Distillation for Mathematical Reasoning

This scenario establishes the common benchmarking protocol for on-policy self-distillation, where a reasoning language model explores and refines its own problem-solving strategies without relying on a larger external teacher model.

---

## 1. Experimental Models

Evaluated across 3 advanced reasoning model architectures:
- `Qwen/Qwen3-8B` (8B parameters)
- `allenai/Olmo3-7B-Think` (7B parameters)
- `Qwen/Qwen3-4B` (4B parameters)

### Pretraining & Initialization Status
All models are initialized from standard instruction-tuned checkpoints (`Base (Instruct)`). During self-distillation, an immutable copy is preserved as reference policy $\theta_0$ to prevent policy drift.

---

## 2. Datasets & Resources

### Training Prompts (Exploration Corpus)
- Mathematical prompts drawn from [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math), and [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT).

### Evaluation Benchmarks (Test Sets)
Evaluated across 3 high-difficulty Olympiad mathematics competitions:
1. **AIME 2025 (AIME25):** Official 2025 American Invitational Mathematics Examination — HF: [HuggingFaceH4/aime_2025](https://huggingface.co/datasets/HuggingFaceH4/aime_2025)
2. **AIME 2026 (AIME26):** Predictive benchmark of 2026 competition problems.
3. **HMMT 2025 (HMMT25):** Harvard-MIT Mathematics Tournament 2025 — HF: [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math)

---

## 3. Evaluation Metrics

- **Primary Metric:** Strict Accuracy (%) on extracted final numerical/boxed answers.
- **Aggregated Metric:** Unweighted arithmetic mean (Average) across AIME25, AIME26, and HMMT25.

---

## 4. Participating Baselines & Paper Links

| Baseline | Official Paper Reference | Venue / Record | Key Distillation Objective |
| :--- | :--- | :--- | :--- |
| **Base (Instruct)** | Pretrained instruction checkpoint | - | Direct evaluation without mathematical reinforcement |
| **+ SFT** | Supervised Fine-Tuning | - | Token-level cross-entropy on ground-truth solutions |
| **+ GRPO** | *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (Shao et al.) | arXiv 2024 · [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | Group Relative Policy Optimization with mathematical verifier |
| **+ OPSD** | *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models* (Zhao et al.) | ICML 2026 · [PMLR](https://proceedings.mlr.press/v306/zhao26be.html) · [arXiv:2601.18734](https://arxiv.org/abs/2601.18734) | Privileged-context teacher rationalizing on-policy student rollouts via Gen-JSD |
| **+ RLSD** | *Self-Distilled RLVR* (Yang et al.) | arXiv 2026 · [arXiv:2604.03128](https://arxiv.org/abs/2604.03128) · [code](https://github.com/iie-ycx/RLSD) | RLVR outcome reward + self-distillation token update magnitude |
| **+ SDPO** | *Reinforcement Learning via Self-Distillation* (Hübotter et al.) | ICML 2026 · [PMLR](https://proceedings.mlr.press/v306/hubotter26a.html) · [arXiv:2601.20802](https://arxiv.org/abs/2601.20802) | Self-Distillation Policy Optimization leveraging rich feedback |

---

## 5. Common Generalized Train & Test Scenario

- **On-Policy Trajectory Sampling:** At each training step, the student policy $\theta_t$ samples a group of candidate reasoning paths (group size $G = 8$) for each prompt via vLLM.
- **Outcome Verification:** A symbolic verifier (`math_verify`) evaluates each candidate against ground-truth mathematical answers, assigning reward $r \in \{0, 1\}$.
- **Self-Distillation Update:** Successful reasoning rollouts are distilled into the student weights via policy gradient or filtered self-imitation with KL divergence regularization against $\theta_0$.
- **Testing Phase:** Zero-shot evaluation with greedy decoding (`temperature=0.0`, `max_new_tokens=4096`).

---

## 6. Official Empirical Results

| Method | AIME25 (Acc. %) | AIME26 (Acc. %) | HMMT25 (Acc. %) | Average |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen3-8B** | | | | |
| Base (Instruct) | 66.7 | 69.2 | 44.2 | 60.0 |
| \quad + SFT | 64.2 | 66.7 | 42.9 | 57.9 |
| \quad + GRPO | 68.9 | 70.3 | 46.7 | 62.0 |
| \quad + **OPSD** | **70.0** | **72.5** | **47.5** | **63.3** |
| \quad + RLSD | 69.4 | 68.6 | 46.9 | 61.6 |
| \quad + SDPO | 68.9 | 70.0 | 45.6 | 61.5 |
| **Olmo3-7B-Think** | | | | |
| Base (Instruct) | 65.3 | 68.3 | **46.7** | 60.1 |
| \quad + SFT | 66.7 | 66.1 | 40.6 | 57.8 |
| \quad + GRPO | 69.2 | 72.8 | 44.7 | 62.2 |
| \quad + OPSD | 68.1 | 68.9 | 45.6 | 60.8 |
| \quad + RLSD | 69.2 | **73.1** | 44.4 | 62.2 |
| \quad + **SDPO** | **69.4** | 71.9 | 45.8 | **62.4** |
| **Qwen3-4B** | | | | |
| Base (Instruct) | 66.7 | 66.7 | 42.2 | 58.5 |
| \quad + SFT | 62.3 | 63.9 | 43.4 | 56.5 |
| \quad + GRPO | 68.1 | 65.8 | 44.4 | 59.4 |
| \quad + **OPSD** | **68.9** | **70.6** | **45.5** | **61.7** |
| \quad + RLSD | 67.8 | 65.8 | 44.4 | 59.3 |
| \quad + SDPO | 64.4 | 67.8 | 44.4 | 58.9 |

