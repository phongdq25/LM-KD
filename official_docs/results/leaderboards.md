# Master Benchmark Leaderboards

This document compiles the complete benchmark results across all 7 experimental scenarios. All baseline methods are strictly normalized to official paper titles and publication records in the [Baseline Methods & Paper Registry](../scenarios/index.md#baseline-methods--paper-registry).

For full descriptions of the experimental setups, models, dataset links, and shared workflows, refer to the corresponding [**Unified Scenarios**](../scenarios/index.md) and [**Methods Catalog**](../methods/index.md).

---

## 1. Table 1: Sentence Embedding Representation Distillation
**Companion:** [Scenario 1: Sentence Embedding](../scenarios/1-sentence-embedding.md)

<details>
<summary><b>More Detail</b></summary>


- **Datasets:** 9 MTEB benchmark datasets (In-domain marked with *, Out-of-domain without).
- **Metrics:** Classification Macro-F1 (%), Pair Classification Average Precision (AP, %), STS Spearman Correlation ($\rho \times 100$).

| Method | Classification (F1)<br>Banking77 | Classification (F1)<br>Tweet | Classification (F1)<br>Emotion* | Pair Classify (AP)<br>MRPC | Pair Classify (AP)<br>SciTail | Pair Classify (AP)<br>WiC* | STS (Spearman)<br>SICK | STS (Spearman)<br>STS12 | STS (Spearman)<br>STSB* | Domain Avg<br>Avg-In | Domain Avg<br>Avg-Out | Total<br>Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-Embedding 0.6B $\rightarrow$ MiniLMv2 H384 22M** | | | | | | | | | | | | |
| Teacher | 93.21 | 71.33 | 66.53 | 83.37 | 90.00 | 66.85 | 80.64 | 77.11 | 84.60 | 72.66 | 82.61 | 79.29 |
| Student base | 68.29 | 69.90 | 41.07 | 78.39 | 64.56 | 59.58 | 47.60 | 21.95 | 22.08 | 40.91 | 58.45 | 52.60 |
| SimCSE-unsup | 86.67 | 69.56 | 54.00 | 82.29 | 73.76 | 65.71 | 66.04 | 59.11 | 62.08 | 60.60 | 72.91 | 68.80 |
| Jasper and Stella | 85.94 | 71.25 | 57.99 | 83.36 | 76.17 | 68.25 | 70.57 | 63.81 | 70.11 | 65.45 | 75.18 | 71.94 |
| DistillCSE | 88.03 | 69.87 | 54.24 | 83.20 | 77.66 | 67.88 | 70.09 | 68.19 | 71.53 | 64.55 | 76.17 | 72.30 |
| EMO | 87.03 | 71.67 | 59.63 | 83.45 | 78.17 | **68.89** | 70.81 | 65.78 | 71.42 | 66.65 | 76.15 | 72.98 |
| MoL | 86.64 | 71.44 | 59.02 | 82.97 | 78.60 | 68.23 | 69.18 | 67.77 | 71.09 | 66.11 | 76.10 | 72.77 |
| **TALAS** | 86.69 | **72.77** | **60.94** | **85.10** | **81.42** | 66.65 | **72.51** | **70.77** | **76.22** | **67.94** | **78.21** | **74.79** |
| **BGE-M3 $\rightarrow$ MiniLMv2 H768 66M** | | | | | | | | | | | | |
| Teacher | 93.52 | 73.85 | 68.56 | 85.81 | 91.87 | 61.47 | 79.18 | 78.73 | 84.87 | 71.63 | 83.83 | 79.76 |
| Student base | 81.84 | 71.67 | 47.70 | 79.67 | 65.06 | 60.89 | 49.46 | 26.76 | 24.12 | 44.24 | 62.41 | 56.35 |
| SimCSE-unsup | 87.89 | 71.45 | 55.65 | 83.87 | 76.32 | 67.96 | 66.93 | 59.66 | 63.97 | 62.53 | 74.35 | 70.41 |
| DistillCSE | 90.50 | 72.70 | 59.88 | 84.39 | 78.48 | **69.49** | 73.97 | **70.00** | 73.73 | 67.70 | 78.34 | 74.79 |
| EMO | 90.19 | 74.26 | 61.48 | 85.31 | 80.82 | 68.30 | 75.41 | 67.06 | 76.35 | 68.71 | 78.84 | 75.46 |
| MoL | 90.12 | 73.65 | **67.34** | 84.91 | 80.50 | 67.86 | 76.08 | 68.30 | 74.44 | **69.88** | 78.93 | 75.91 |
| **TALAS-ASAM** | **90.62** | **74.81** | 63.30 | **87.05** | **81.94** | 66.28 | **76.86** | 69.04 | **79.01** | 69.53 | **80.05** | **76.55** |
| **Qwen3-Embedding 4B $\rightarrow$ BERT-base 109M** | | | | | | | | | | | | |
| Teacher | 93.63 | 72.22 | 68.94 | 84.79 | 91.77 | 66.80 | 83.43 | 82.55 | 86.87 | 74.20 | 84.73 | 81.22 |
| Student base | 84.86 | 68.02 | 47.79 | 77.36 | 67.74 | 59.22 | 42.43 | 21.54 | 20.30 | 42.44 | 60.33 | 54.36 |
| SimCSE-unsup | 89.21 | 69.13 | 53.55 | 82.86 | 76.59 | **71.64** | 68.19 | 61.68 | 70.36 | 65.18 | 74.61 | 71.47 |
| Jasper and Stella | 89.71 | 73.83 | 65.98 | 85.33 | 80.99 | 70.50 | 73.96 | 67.90 | 75.65 | 70.71 | 78.62 | 75.98 |
| DistillCSE | 90.94 | 70.90 | 60.80 | 82.86 | 78.98 | 68.63 | 75.12 | 72.61 | 77.08 | 68.84 | 78.57 | 75.32 |
| EMO | 90.78 | 73.54 | 67.48 | 84.46 | 81.57 | 69.76 | 75.66 | 68.83 | 77.17 | 71.47 | 79.14 | 76.58 |
| MoL | 90.68 | 72.48 | 70.23 | 85.67 | 81.31 | 68.17 | 75.13 | 67.87 | 75.99 | 71.46 | 78.86 | 76.39 |
| **TALAS-ASAM** | **91.43** | **74.30** | **70.71** | **86.47** | **82.66** | 69.24 | **78.38** | **75.40** | **80.88** | **73.61** | **81.44** | **78.83** |


</details>

---

## 2. Table 2: Same-Tokenizer Instruction Distillation
**Companion:** [Scenario 2: Same-Tokenizer Instruction Distillation](../scenarios/2-same-tokenizer.md)

<details>
<summary><b>More Detail</b></summary>


- **Metric:** ROUGE-L ($\uparrow$) across Dolly, SelfInst, Vicuna, and Super-NaturalInstructions (S-NI).

| Methods | Dolly | SelfInst | Vicuna | S-NI | Average |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GPT-2 1.5B $\rightarrow$ GPT-2 120M** | | | | | |
| Teacher Reference | 28.71 | 15.68 | 17.00 | 28.80 | 22.55 |
| SFT Baseline | 24.61 | 11.14 | 14.36 | 16.85 | 16.74 |
| FDD | 25.47 | 12.45 | 16.44 | 23.54 | 19.48 |
| DistiLLM | 25.65 | 13.39 | 16.50 | 25.28 | 20.21 |
| DistiLLM-2 | 22.44 | 12.52 | 12.30 | 27.10 | 18.59 |
| CSD | 25.75 | 13.20 | 16.60 | 24.00 | 19.89 |
| AMiD | 24.40 | 13.00 | 16.56 | 24.03 | 19.50 |
| **DistiLLM + MTA** | **25.77** | **14.19** | **16.67** | **29.18** | **21.45** |
| **Qwen1.5 1.8B $\rightarrow$ Qwen1.5 0.5B** | | | | | |
| Teacher Reference | 29.84 | 21.03 | 20.36 | 34.99 | 26.55 |
| SFT Baseline | 24.83 | 13.31 | 16.97 | 22.07 | 19.30 |
| FDD | 25.08 | 12.24 | 16.08 | 23.69 | 19.27 |
| DistiLLM | 25.16 | 12.90 | 15.86 | 25.28 | 19.80 |
| DistiLLM-2 | 27.48 | 17.95 | 17.14 | 30.99 | 23.39 |
| CSD | 25.42 | 13.78 | 15.78 | 30.60 | 21.39 |
| AMiD | 25.89 | 14.46 | 16.88 | 23.38 | 20.15 |
| **DistiLLM-2 + MTA** | **27.93** | **18.87** | **18.63** | **33.49** | **24.73** |
| **OPT 6.7B $\rightarrow$ OPT 1.3B** | | | | | |
| Teacher Reference | 28.92 | 17.08 | 18.09 | 31.25 | 23.84 |
| SFT Baseline | 26.00 | 11.40 | 15.60 | 23.10 | 19.03 |
| FDD | 26.07 | 14.82 | 17.09 | 28.98 | 21.74 |
| DistiLLM | 27.87 | 15.99 | 18.02 | 30.02 | 22.98 |
| DistiLLM-2 | 26.31 | 17.62 | 17.60 | 30.85 | 23.10 |
| CSD | 24.39 | 13.78 | 16.44 | 28.33 | 20.74 |
| AMiD | 25.87 | 16.87 | 17.45 | 29.86 | 22.51 |
| **AMiD + MTA** | **28.49** | **17.78** | **18.45** | **31.44** | **24.04** |


</details>

---

## 3. Table 3: Cross-Tokenizer Representation Distillation
**Companion:** [Scenario 3: Cross-Tokenizer Representation Distillation](../scenarios/3-cross-tokenizer.md)

<details>
<summary><b>More Detail</b></summary>


- **Metric:** ROUGE-L ($\uparrow$) across Dolly, SelfInst, Vicuna, and S-NI across 5 disparate-vocabulary configurations.

| Methods | Dolly | SelfInst | Vicuna | S-NI | Average |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen1.5 1.8B $\rightarrow$ GPT-2 120M** | | | | | |
| Teacher Reference | 29.84 | 21.03 | 20.36 | 34.99 | 26.55 |
| SFT Baseline | 24.61 | 11.14 | 14.36 | 16.85 | 16.74 |
| ULD | 23.77 | 9.30 | 14.33 | 14.04 | 15.36 |
| MinED | 24.21 | 10.02 | 14.96 | 16.40 | 16.40 |
| MultiLevelOT | 23.92 | 13.04 | 15.28 | 22.43 | 18.67 |
| DSKDv2 | 24.12 | 12.63 | 16.06 | 18.96 | 17.94 |
| DWA-KD | 24.29 | 11.15 | 15.64 | 19.63 | 17.68 |
| ResidualKD | 23.75 | 12.47 | 15.84 | 22.19 | 18.56 |
| MCW-KD | 25.03 | 11.41 | 15.91 | 18.71 | 17.77 |
| ResidualKD + MTA | 24.01 | 15.79 | 13.30 | 22.78 | 18.97 |
| **SRA** | **23.36** | **13.16** | **16.08** | **23.70** | **19.08** |
| **Qwen1.5 1.8B $\rightarrow$ GPT-2 0.4B** | | | | | |
| Teacher Reference | 29.84 | 21.03 | 20.36 | 34.99 | 26.55 |
| SFT Baseline | 23.90 | 9.20 | 14.60 | 12.93 | 15.16 |
| ULD | 23.90 | 9.96 | 15.04 | 16.26 | 16.29 |
| MinED | 24.48 | 11.21 | 15.56 | 15.69 | 16.74 |
| MultiLevelOT | 25.53 | 13.23 | 15.72 | 22.56 | 19.26 |
| DSKDv2 | 25.58 | 12.04 | 15.46 | 20.03 | 18.28 |
| DWA-KD | 26.63 | 12.73 | 15.92 | 20.84 | 19.03 |
| ResidualKD | 24.87 | 14.62 | 16.31 | 23.62 | 19.86 |
| MCW-KD | 26.12 | 11.97 | 16.15 | 19.98 | 18.56 |
| **ResidualKD + MTA** | **25.75** | **14.63** | **16.74** | **24.49** | **20.40** |
| SRA | 23.97 | 13.64 | 16.31 | 24.49 | 19.60 |
| **Qwen2.5 7B $\rightarrow$ GPT-2 1.5B** | | | | | |
| Teacher Reference | 29.10 | 26.12 | 20.95 | 38.16 | 28.58 |
| SFT Baseline | 25.15 | 14.80 | 16.32 | 22.80 | 19.77 |
| ULD | 24.52 | 15.11 | 15.94 | 26.18 | 20.44 |
| MinED | 25.52 | 15.39 | 16.15 | 26.25 | 20.83 |
| MultiLevelOT | 26.24 | 17.31 | 17.28 | 26.14 | 21.74 |
| DSKDv2 | 25.58 | 16.75 | 17.19 | 28.22 | 21.94 |
| DWA-KD | 26.29 | 17.13 | 16.81 | 29.10 | 22.33 |
| ResidualKD | 25.73 | 17.75 | 17.11 | 28.20 | 22.20 |
| **MCW-KD** | **27.94** | **18.35** | **19.19** | **28.70** | **23.55** |
| DWA-KD + MTA | 26.88 | 17.72 | 17.45 | 28.90 | 22.74 |
| SRA | 26.45 | 17.22 | 18.25 | 29.50 | 22.86 |
| **Qwen2.5 7B $\rightarrow$ OPT 2.7B** | | | | | |
| Teacher Reference | 29.10 | 26.12 | 20.95 | 38.16 | 28.58 |
| SFT Baseline | 29.55 | 15.06 | 17.63 | 26.83 | 22.27 |
| ULD | 26.65 | 15.37 | 16.97 | 25.44 | 21.11 |
| MinED | 26.89 | 14.98 | 17.04 | 25.94 | 21.21 |
| MultiLevelOT | 28.30 | 17.81 | 17.28 | 29.36 | 23.19 |
| DSKDv2 | 27.15 | 16.80 | 18.16 | 29.18 | 22.82 |
| DWA-KD | 28.45 | 15.65 | 18.16 | 30.54 | 23.20 |
| ResidualKD | 28.10 | 18.12 | 18.22 | 28.05 | 23.12 |
| MCW-KD | 28.57 | 17.29 | 18.68 | 29.75 | 23.57 |
| **DWA-KD + MTA** | **28.83** | **17.48** | **19.00** | **31.01** | **24.08** |
| SRA | 28.52 | 17.14 | 18.48 | 27.35 | 22.87 |
| **Mistral 7B $\rightarrow$ TinyLLaMA 1.1B** | | | | | |
| Teacher Reference | 32.12 | 26.31 | 20.94 | 35.58 | 28.74 |
| SFT Baseline | 23.92 | 16.98 | 16.64 | 28.52 | 21.52 |
| ULD | 25.48 | 17.72 | 17.31 | 32.54 | 23.26 |
| MinED | 25.54 | 18.23 | 17.02 | 31.42 | 23.05 |
| MultiLevelOT | 26.41 | 15.16 | 16.94 | 26.66 | 21.29 |
| DSKDv2 | 26.26 | 20.04 | 19.09 | 29.27 | 23.67 |
| DWA-KD | 27.99 | 19.75 | 17.10 | 31.70 | 24.13 |
| ResidualKD | 25.76 | 18.31 | 19.34 | 30.33 | 23.43 |
| **MCW-KD** | **26.51** | **18.71** | **18.40** | **37.00** | **25.16** |
| DSKDv2 + MTA | 27.09 | 20.34 | 19.81 | 33.10 | 25.08 |
| SRA | 25.02 | 20.05 | 19.69 | 32.98 | 24.44 |


</details>

---

## 4. Table 4: Long-CoT Reasoning Chain Distillation
**Companion:** [Scenario 4: Long-CoT Reasoning Chain Distillation](../scenarios/4-long-cot.md)

<details>
<summary><b>More Detail</b></summary>


- **Teacher:** `deepseek-ai/DeepSeek-R1`
- **Metrics:** Pass@1 (p@1, %) and Pass@3 (p@3, %) across Olympiad and advanced mathematics competitions.

| Method | AIME25<br>p@1 | AIME25<br>p@3 | AIME24<br>p@1 | AIME24<br>p@3 | AMC12<br>p@1 | AMC12<br>p@3 | MATH500<br>p@1 | MATH500<br>p@3 | Average<br>p@1 | Average<br>p@3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSeek-R1-Distill-Qwen-1.5B** *(Full Fine-Tuning)* | | | | | | | | | | |
| Zero-shot (Base) | 10.00 | 20.00 | 15.56 | 33.33 | 49.80 | 63.86 | 72.33 | 81.80 | 36.92 | 49.75 |
| SFT (long CoT)$^\dagger$ | 10.00 | 20.00 | 15.56 | **30.00** | 44.58 | 61.45 | 68.40 | 83.00 | 34.64 | 48.61 |
| SGL$^\dagger$ | 16.67 | 23.33 | **17.78** | 23.33 | 48.59 | 59.04 | 74.13 | 86.00 | 39.29 | 47.93 |
| Segment-Selective SFT$^\dagger$ | 21.11 | 30.00 | 16.67 | 23.33 | 50.60 | 65.06 | 74.87 | 84.60 | **40.81** | 50.75 |
| **P-ALIGN** | 16.67 | 26.67 | **17.78** | **30.00** | **50.20** | **63.86** | **78.40** | **87.60** | 40.76 | **52.03** |
| **Qwen2.5-7B-Instruct** *(LoRA)* | | | | | | | | | | |
| Zero-shot (Base) | 4.44 | 10.00 | 8.89 | 13.33 | 44.98 | 63.86 | 74.67 | 85.20 | 33.24 | 43.10 |
| SFT (long CoT)$^\dagger$ | 11.11 | 16.67 | 8.89 | 13.33 | 34.94 | 48.19 | 65.27 | 80.40 | 30.05 | 39.65 |
| SGL$^\dagger$ | 8.89 | 20.00 | 8.89 | 16.67 | **44.58** | 60.24 | **75.13** | **85.40** | **34.37** | 45.58 |
| Segment-Selective SFT$^\dagger$ | 6.67 | 13.33 | 12.22 | **23.33** | 42.57 | **65.06** | 66.20 | 81.40 | 31.92 | **45.78** |
| **P-ALIGN** | **12.22** | **23.33** | **12.22** | 20.00 | 38.55 | 51.81 | 71.87 | 83.40 | 33.72 | 44.64 |
| **Qwen3-8B** *(LoRA)* | | | | | | | | | | |
| Zero-shot (Base) | 20.00 | 23.33 | 22.22 | 33.33 | 52.61 | 68.67 | 77.00 | 84.80 | 42.96 | 52.53 |
| SFT (long CoT)$^\dagger$ | 20.00 | 23.33 | 22.22 | 33.33 | 57.43 | 71.08 | 81.47 | 89.80 | 45.28 | 54.39 |
| SGL$^\dagger$ | 17.78 | 26.67 | 23.33 | 30.00 | 53.82 | 68.67 | 80.33 | 90.00 | 43.81 | 53.84 |
| Segment-Selective SFT$^\dagger$ | **22.22** | **33.33** | 22.22 | 33.33 | 55.42 | 68.67 | 80.33 | 89.40 | 45.05 | 56.18 |
| **P-ALIGN** | 21.11 | 30.00 | **30.00** | **40.00** | **60.64** | **73.49** | **85.20** | **91.80** | **49.24** | **58.82** |

*(Note: $^\dagger$ indicates trained on the s1K-1.1 corpus. Zero-shot results are reported for baseline context and excluded from ranking.)*


</details>

---

## 5. Table 5: On-Policy Self-Distillation for Mathematical Reasoning
**Companion:** [Scenario 5: On-Policy Self-Distillation for Reasoning](../scenarios/5-opsd.md)

<details>
<summary><b>More Detail</b></summary>


- **Metric:** Strict Accuracy (%) on AIME25, AIME26, and HMMT25.

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


</details>

---

## 6. Table 6: Preference Distillation
**Companion:** [Scenario 6: Reward-Model-Free Preference Distillation](../scenarios/6-preference-distillation.md)

<details>
<summary><b>More Detail</b></summary>


- **Metric:** Accuracy (%) across HellaSwag, ARC, TruthfulQA, Winogrande, and GSM8K.

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


</details>

---

## 7. Table 7: Self-Play Fine-Tuning
**Companion:** [Scenario 7: Adversarial Self-Play Fine-Tuning](../scenarios/7-self-play-fine-tuning.md)

<details>
<summary><b>More Detail</b></summary>


- **Metric:** Accuracy / Normalized score in $[0, 1]$ across 7 benchmarks.

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


</details>