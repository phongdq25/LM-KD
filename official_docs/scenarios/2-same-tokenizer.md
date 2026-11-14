# Scenario 2: Same-Tokenizer Instruction Distillation

This scenario defines the standardized benchmarking protocol for instruction following and conversational capabilities where student and teacher models share the identical subword vocabulary and tokenizer ($\mathcal{V}_S \equiv \mathcal{V}_T$).

---

## 1. Experimental Models

Evaluated across 3 distinct Teacher $\rightarrow$ Student pairs sharing the same tokenizer:

- **Pair 1 (GPT-2 Family, BPE $|\mathcal{V}| = 50,257$):**
  - **Teacher:** `openai-community/gpt2-xl` (1.5B parameters)
  - **Student:** `openai-community/gpt2` (120M parameters)
- **Pair 2 (Qwen1.5 Family, BPE $|\mathcal{V}| = 151,936$):**
  - **Teacher:** `Qwen/Qwen1.5-1.8B` (1.8B parameters)
  - **Student:** `Qwen/Qwen1.5-0.5B` (0.5B parameters)
- **Pair 3 (OPT Family, BPE $|\mathcal{V}| = 50,272$):**
  - **Teacher:** `facebook/opt-6.7b` (6.7B parameters)
  - **Student:** `facebook/opt-1.3b` (1.3B parameters)

### Pretraining & Initialization Status
- Models are initialized from official foundation or instruction-tuned checkpoints.
- Teachers remain completely frozen in evaluation mode.

---

## 2. Datasets & Resources

### Training Corpus
- [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (Training split: 15,000 instructions across brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization).
- *Local Workspace Path:* [`datasets/same_cross_tokenizer_data/dolly/train.jsonl`](../../datasets/same_cross_tokenizer_data/dolly/train.jsonl)

### Validation Split
- Dolly 15k validation set: [`datasets/same_cross_tokenizer_data/dolly/dev.jsonl`](../../datasets/same_cross_tokenizer_data/dolly/dev.jsonl)

### Evaluation Benchmarks (Test Splits)
Zero-shot generation evaluated across 4 diverse benchmark sets:
1. **Dolly (In-domain Test):** [`datasets/same_cross_tokenizer_data/dolly/valid.jsonl`](../../datasets/same_cross_tokenizer_data/dolly/valid.jsonl) — HF: [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
2. **Self-Instruct (User-Oriented Prompts):** [`datasets/same_cross_tokenizer_data/self-inst/valid.jsonl`](../../datasets/same_cross_tokenizer_data/self-inst/valid.jsonl) — HF: [yizhongw/self_instruct](https://huggingface.co/datasets/yizhongw/self_instruct)
3. **Vicuna Evaluation (Challenging Open-ended Prompts):** [`datasets/same_cross_tokenizer_data/vicuna/valid.jsonl`](../../datasets/same_cross_tokenizer_data/vicuna/valid.jsonl) — HF: [lmsys/vicuna-eval](https://huggingface.co/datasets/lmsys/vicuna-eval)
4. **Super-NaturalInstructions (S-NI Task Generalization):** [`datasets/same_cross_tokenizer_data/sinst/11_/valid.jsonl`](../../datasets/same_cross_tokenizer_data/sinst/11_/valid.jsonl) — HF: [allenai/super-naturalinstructions](https://huggingface.co/datasets/allenai/super-naturalinstructions)

---

## 3. Evaluation Metrics

- **Primary Metric:** ROUGE-L F1-score ($\uparrow$) measuring longest common subsequence match on generated responses across each benchmark.
- **Aggregated Metric:** Unweighted arithmetic mean (Avg.) across all 4 evaluation test sets.

---

## 4. Participating Baselines & Paper Links

| Baseline | Official Paper Reference | Venue / Record | Key Distillation Objective |
| :--- | :--- | :--- | :--- |
| **Teacher Reference** | Empirical Upper Bound | - | Uncompressed frozen teacher model |
| **SFT Baseline** | Standard Supervised Fine-Tuning | - | Token-level cross-entropy on instruction responses |
| **FDD** | *Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation* (Gong et al.) | ACL 2025 · [ACL Anthology](https://aclanthology.org/2025.acl-long.1125/) | Feature Dynamics Distillation aligning layer hidden velocity $\Delta h$ |
| **DistiLLM / DistiLLM-2** | *DistiLLM: Towards Streamlined Distillation for Large Language Models* (Ko et al.) | ICML 2024 · [PMLR](https://proceedings.mlr.press/v235/ko24c.html) · [arXiv:2402.03898](https://arxiv.org/abs/2402.03898) | Skewed Reverse KL (SRKL) + Adaptive Selective Generation Optimization |
| **CSD** | *Distillation of Large Language Models via Concrete Score Matching* (Kim et al.) | ICLR 2026 · [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1bfc9f74afa91b9b8add5a97a97001a1-Abstract-Conference.html) · [arXiv:2509.25837](https://arxiv.org/abs/2509.25837) | Concrete Score Distillation via logit shift-invariant pairwise residuals |
| **AMiD** | *AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution* (Shin et al.) | ICLR 2026 · [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/54a468d19bc5727800096d9c7c51a423-Abstract-Conference.html) · [arXiv:2510.15982](https://arxiv.org/abs/2510.15982) | Continuous α-mixture assistant distribution interpolation |
| **DistiLLM / AMiD + MTA** | *MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation* (Chi et al.) | ACL 2026 · [ACL Anthology](https://aclanthology.org/2026.acl-long.1507/) | Multi-Granular Trajectory Alignment across word and syntactic phrase tiers |

---

## 5. Common Generalized Train & Test Scenario

- **Data Ingestion & Formatting:** All instruction samples are formatted with role delimiters (`User:` and `Assistant:`). Gradients are strictly backpropagated through assistant response tokens using unified `seq.loss_mask`.
- **Training Phase:** All baseline algorithms ingest the identical token stream. At each training step, the student policy is updated according to the specific divergence objective while normalising gradients by `num_items_in_batch`.
- **Testing Phase:** Checkpoints are evaluated via greedy autoregressive decoding (`temperature=0.0`, `max_new_tokens=1024`) across the 4 benchmark datasets without task-specific fine-tuning.

---

## 6. Official Empirical Results

| Method | Dolly | SelfInst | Vicuna | S-NI | Average |
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

