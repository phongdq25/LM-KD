# Scenario 3: Cross-Tokenizer Representation & Vocabulary Distillation

This scenario establishes the common benchmarking protocol for cross-architecture and cross-vocabulary knowledge distillation where the student and teacher models possess entirely different tokenizers, vocabulary sizes ($V_S \neq V_T$), and subword segmentation boundaries.

---

## 1. Experimental Models

Evaluated across 5 distinct Teacher $\rightarrow$ Student configurations representing various vocabulary sizes and embedding dimensions:

- **Pair 1 (Qwen1.5 $\rightarrow$ Small GPT-2):**
  - **Teacher:** `Qwen/Qwen1.5-1.8B` ($V = 151{,}936$, $D = 2048$)
  - **Student:** `openai-community/gpt2` (120M parameters, $V = 50{,}257$, $D = 768$)
- **Pair 2 (Qwen1.5 $\rightarrow$ Medium GPT-2):**
  - **Teacher:** `Qwen/Qwen1.5-1.8B` ($V = 151{,}936$, $D = 2048$)
  - **Student:** `openai-community/gpt2-medium` (0.4B parameters, $V = 50{,}257$, $D = 1024$)
- **Pair 3 (Qwen2.5 $\rightarrow$ Large GPT-2):**
  - **Teacher:** `Qwen/Qwen2.5-7B` ($V = 152{,}064$, $D = 3584$)
  - **Student:** `openai-community/gpt2-xl` (1.5B parameters, $V = 50{,}257$, $D = 1600$)
- **Pair 4 (Qwen2.5 $\rightarrow$ OPT):**
  - **Teacher:** `Qwen/Qwen2.5-7B` ($V = 152{,}064$, $D = 3584$)
  - **Student:** `facebook/opt-2.7b` ($V = 50{,}272$, $D = 2560$)
- **Pair 5 (Mistral $\rightarrow$ TinyLLaMA):**
  - **Teacher:** `mistralai/Mistral-7B-v0.1` ($V = 32{,}000$, $D = 4096$)
  - **Student:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` ($V = 32{,}000$, disparate LLaMA tokenizer mapping)

### Pretraining & Initialization Status
All models are initialized from official pretrained foundation checkpoints. Teachers are frozen in evaluation mode.

---

## 2. Datasets & Resources

### Training Corpus
- [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (Training split: 15,000 instruction-response pairs).
- *Local Workspace Path:* [`datasets/same_cross_tokenizer_data/dolly/train.jsonl`](../../datasets/same_cross_tokenizer_data/dolly/train.jsonl)

### Validation Split
- Dolly 15k validation set: [`datasets/same_cross_tokenizer_data/dolly/dev.jsonl`](../../datasets/same_cross_tokenizer_data/dolly/dev.jsonl)

### Evaluation Benchmarks (Test Splits)
1. **Dolly (In-domain Test):** [`datasets/same_cross_tokenizer_data/dolly/valid.jsonl`](../../datasets/same_cross_tokenizer_data/dolly/valid.jsonl) — HF: [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
2. **Self-Instruct:** [`datasets/same_cross_tokenizer_data/self-inst/valid.jsonl`](../../datasets/same_cross_tokenizer_data/self-inst/valid.jsonl) — HF: [yizhongw/self_instruct](https://huggingface.co/datasets/yizhongw/self_instruct)
3. **Vicuna Evaluation:** [`datasets/same_cross_tokenizer_data/vicuna/valid.jsonl`](../../datasets/same_cross_tokenizer_data/vicuna/valid.jsonl) — HF: [lmsys/vicuna-eval](https://huggingface.co/datasets/lmsys/vicuna-eval)
4. **Super-NaturalInstructions (S-NI):** [`datasets/same_cross_tokenizer_data/sinst/11_/valid.jsonl`](../../datasets/same_cross_tokenizer_data/sinst/11_/valid.jsonl) — HF: [allenai/super-naturalinstructions](https://huggingface.co/datasets/allenai/super-naturalinstructions)

### Cross-Tokenizer Vocab Mappings
Pre-extracted vocabulary alignment matrices located in [`datasets/vocab_alignment/`](../../datasets/vocab_alignment/):
- `qwen_to_gpt2`
- `qwen_to_gpt2_medium`
- `qwen_to_opt`
- `mistral_to_tinyllama`

---

## 3. Evaluation Metrics

- **Primary Metric:** ROUGE-L F1-score ($\uparrow$) measuring generation fidelity across Dolly, SelfInst, Vicuna, and S-NI.
- **Aggregated Metric:** Unweighted arithmetic mean (Avg.) across all 4 evaluation test sets.

---

## 4. Participating Baselines & Paper Links

| Baseline | Official Paper Reference | Venue / Record | Key Distillation Objective |
| :--- | :--- | :--- | :--- |
| **Teacher Reference** | Empirical Upper Bound | - | Uncompressed frozen teacher model |
| **SFT Baseline** | Standard Supervised Fine-Tuning | - | Token-level cross-entropy in student vocabulary |
| **ULD** | *Towards Cross-Tokenizer Distillation: The Universal Logit Distillation Loss for LLMs* (Boizard et al.) | TMLR 2025 · [OpenReview](https://openreview.net/forum?id=buBxXiG09A) · [arXiv:2402.12030](https://arxiv.org/abs/2402.12030) | Sorted probability rank-profile matching without vocab alignment |
| **MinED** | *Knowledge Fusion of Large Language Models* (Wan et al., FuseLLM) | ICLR 2024 · [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4fd5cfd2e31bebbccfa5ffa354c04bdc-Abstract-Conference.html) · [arXiv:2401.10491](https://arxiv.org/abs/2401.10491) | Dynamic programming minimum-edit-distance token mapping |
| **MultiLevelOT** | *Multi-Level Optimal Transport for Universal Cross-Tokenizer KD on LMs* (Cui et al.) | AAAI 2025 · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34543) · [arXiv:2412.14528](https://arxiv.org/abs/2412.14528) | Multi-level entropic Sinkhorn optimal transport |
| **DSKDv2** | *A Dual-Space Framework for General Knowledge Distillation of LLMs* (Zhang et al.) | arXiv 2025 · [arXiv:2504.11426](https://arxiv.org/abs/2504.11426) | Pseudoinverse LM-head initialization + agreement masks |
| **DWA-KD** | *DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer KD* (Vu et al.) | Findings EACL 2026 · [ACL Anthology](https://aclanthology.org/2026.findings-eacl.181/) · [arXiv:2602.21669](https://arxiv.org/abs/2602.21669) | Dual-space uncertainty weighting + Soft-DTW sequence alignment |
| **ResidualKD** | *Knowledge Distillation for Large Language Models through Residual Learning* (On et al.) | ICLR 2026 · [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/608fe7e32f7b773545cc1d656a0fdc98-Abstract-Conference.html) | Two-stage decoupled projector pretraining and residual learning |
| **MCW-KD** | *MCW-KD: Multi-Cost Wasserstein Knowledge Distillation for Large Language Models* (Vuong et al.) | AAAI 2026 · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40619) | Multi-cost Wasserstein optimal transport aligning representations & logits |
| **MTA Integrations** | *MTA: Multi-Granular Trajectory Alignment for LLM Distillation* (Chi et al.) | ACL 2026 · [ACL Anthology](https://aclanthology.org/2026.acl-long.1507/) | Multi-Granular Trajectory Alignment combined with DSKDv2 / DWA / ResidualKD |
| **SRA** | *SRA: Span Representation Alignment for Large Language Model Distillation* (Dao et al.) | ACL 2026 · [ACL Anthology](https://aclanthology.org/2026.acl-long.1522/) · [PDF](https://aclanthology.org/2026.acl-long.1522.pdf) | Center-of-Mass span representation alignment via LCS |

---

## 5. Common Generalized Train & Test Scenario

- **Dual-Tokenizer Ingestion:** Raw texts from `Dolly-15k train` are tokenized concurrently with both student and teacher tokenizers, generating synchronized `student.input_ids`, `teacher.input_ids`, and character span offsets.
- **Projector Initialization:** Projector layers (such as Moore-Penrose pseudo-inverse or learned linear heads) bridge the hidden dimension gap ($d_S \to d_T$).
- **Training Phase:** Baselines optimize alignment across token boundaries using character-span matching, soft vocabulary projections, or optimal transport cost matrices, combined with standard next-token cross-entropy on the student tokens.
- **Testing Phase:** Evaluated with student vocabulary decoding (`temperature=0.0`, `max_new_tokens=1024`) across the 4 test sets.

---

## 6. Official Empirical Results

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

