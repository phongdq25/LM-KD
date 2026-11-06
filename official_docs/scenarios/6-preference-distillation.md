# Scenario 6: Reward-Model-Free Preference Distillation

This scenario establishes the common benchmarking protocol for aligning student models to human preference distributions without training, deploying, or maintaining an external Reward Model (RM) or executing multi-stage RLHF loops.

---

## 1. Experimental Models

Evaluated across 3 distinct Teacher $\rightarrow$ Student configurations:

- **Configuration 1 (Llama Family):**
  - **Teacher:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
  - **Student:** `meta-llama/Llama-3.2-1B-Instruct`
- **Configuration 2 (Qwen 8B to 1.7B):**
  - **Teacher:** `Qwen/Qwen3-8B`
  - **Student:** `Qwen/Qwen3-1.7B`
- **Configuration 3 (Qwen 1.7B to 0.6B):**
  - **Teacher:** `Qwen/Qwen3-1.7B`
  - **Student:** `Qwen/Qwen3-0.6B`

### Pretraining Status
Student models start from standard SFT checkpoints. Teachers are evaluated under both DPO-aligned and SFT checkpoints to establish clear performance ceilings.

---

## 2. Datasets & Resources

### Training Corpus
- [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized): 60,000 binarized prompt-chosen-rejected triples $(x, y^+, y^-)$.
- *Source Config:* Streamed/cached directly from HuggingFace Hub via `{type: hf_dataset, path: HuggingFaceH4/ultrafeedback_binarized, split: train_prefs}`.

### Evaluation Benchmarks (Test Sets)
Evaluated across 5 standardized reasoning and alignment benchmarks:
1. **HellaSwag:** 0-shot commonsense reasoning — HF: [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)
2. **ARC (AI2 Reasoning Challenge):** 0-shot complex science QA — HF: [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)
3. **TruthfulQA (MC2):** 0-shot resistance to human misconceptions — HF: [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa)
4. **Winogrande:** 0-shot coreference resolution — HF: [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande)
5. **GSM8K:** 8-shot grade-school mathematical word problems — HF: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)

---

## 3. Evaluation Metrics

- **Accuracy (%):** Standardized log-likelihood evaluation accuracy on HellaSwag, ARC, TruthfulQA, Winogrande, and GSM8K.
- **Aggregated Metric:** Unweighted arithmetic mean (Average) across all 5 benchmark datasets.

---

## 4. Participating Baselines & Paper Links

| Baseline | Official Paper Reference | Venue / Record | Key Distillation Objective |
| :--- | :--- | :--- | :--- |
| **DPO teacher Reference** | Empirical Upper Bound | - | Teacher aligned via Direct Preference Optimization (Rafailov et al., NeurIPS 2023) |
| **SFT teacher Reference** | Supervised Teacher Baseline | - | Teacher trained with standard SFT on preferred responses |
| **SFT Baseline** | Supervised Student Baseline | - | Student trained with standard SFT on preferred responses $y^+$ |
| **VPD (PAD)** | *Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models* (Gu et al.) | Findings ACL 2025 · [ACL Anthology](https://aclanthology.org/2025.findings-acl.822/) | Vanilla Preference Distillation matching teacher margin on $(y^+, y^-)$ |
| **DCKD** | *Advantage-Guided Distillation for Preference Alignment in Small Language Models* (Gao et al.) | ICLR 2025 · [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html) · [arXiv:2502.17927](https://arxiv.org/abs/2502.17927) | Dual-Constrained KD combining forward KL on $y^+$ and contrastive margin |
| **TVKD** | *Preference Distillation via Value-based Reinforcement Learning* (Kwon et al.) | NeurIPS 2025 · [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/98e967164ae2f6811b975d686dece3eb-Abstract-Conference.html) | Teacher Value-based KD via potential-based reward shaping |
| **ADPA** | *Advantage-Guided Distillation for Preference Alignment in Small Language Models* (Gao et al.) | ICLR 2025 · [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html) · [arXiv:2502.17927](https://arxiv.org/abs/2502.17927) | Advantage-Guided Distillation weighting pairs by teacher advantage certainty |

---

## 5. Common Generalized Train & Test Scenario

- **Data Formatting:** Samples provide prompt $x$, preferred response $y^+$, and dispreferred response $y^-$.
- **Direct Distillation Phase:** Methods update student parameters to widen the probability margin $P_\theta(y^+ \mid x) - P_\theta(y^- \mid x)$ while matching the teacher's distribution on $y^+$ and penalizing agreement on $y^-$.
- **Testing Phase:** Zero-shot and few-shot log-likelihood evaluation across all 5 evaluation benchmarks using identical LM-Evaluation-Harness tasks.

---

## 6. Official Empirical Results

| Method | HellaSwag | ARC | TruthfulQA | Winogrande | GSM8K | Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Llama-3.1-8B $\rightarrow$ Llama-3.2-1B** | | | | | | |
| DPO teacher Reference | 83.05 | 52.30 | 57.37 | 78.22 | 52.77 | 64.74 |
| SFT teacher Reference | 81.85 | 51.20 | 53.59 | 78.45 | 47.84 | 62.59 |
| SFT Baseline | 65.84 | 39.24 | 41.13 | 62.90 | 7.28 | 43.28 |
| VPD (PAD) | 65.87 | 40.31 | 46.18 | 62.05 | 6.49 | 44.18 |
| DCKD | 65.52 | 38.29 | 48.60 | 62.27 | 6.90 | 44.32 |
| **TVKD** | **67.64** | 39.12 | 50.53 | 62.59 | 6.91 | **45.36** |
| ADPA | 65.87 | 38.57 | **51.46** | 62.27 | 6.88 | 45.01 |
| **Qwen3-8B $\rightarrow$ Qwen3-1.7B** | | | | | | |
| DPO teacher Reference | 80.60 | 66.55 | 61.52 | 75.93 | 80.89 | 73.10 |
| SFT teacher Reference | 79.30 | 66.04 | 58.96 | 75.69 | 79.53 | 71.90 |
| SFT Baseline | 67.47 | 52.90 | 49.52 | 64.40 | 64.37 | 59.73 |
| VPD (PAD) | 67.56 | 54.82 | 54.94 | 64.17 | 64.29 | 61.16 |
| DCKD | 66.75 | 54.92 | 54.26 | 64.64 | **66.49** | 61.41 |
| TVKD | 68.47 | 54.90 | **56.52** | 64.40 | 64.37 | 61.73 |
| **ADPA** | 68.30 | **56.23** | 55.50 | 64.56 | 64.25 | **61.77** |
| **Qwen3-1.7B $\rightarrow$ Qwen3-0.6B** | | | | | | |
| DPO teacher Reference | 68.44 | 53.84 | 52.54 | 64.01 | 65.25 | 61.46 |
| SFT teacher Reference | 67.47 | 52.90 | 49.52 | 64.40 | 64.37 | 60.59 |
| SFT Baseline | 53.08 | 37.63 | 45.92 | 59.51 | 47.84 | 49.09 |
| VPD (PAD) | **54.83** | 38.12 | 46.15 | **60.21** | 51.74 | 50.21 |
| DCKD | 53.12 | 37.84 | 46.92 | 59.27 | 53.18 | 50.01 |
| **TVKD** | 53.42 | 38.48 | **49.00** | 59.98 | **54.44** | **50.93** |
| ADPA | 54.10 | **39.33** | 46.11 | 59.27 | 53.22 | 50.50 |

