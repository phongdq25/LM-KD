# Scenario 1: Sentence Embedding Representation Distillation

This scenario establishes the common benchmarking protocol for distilling massive dense embedding models into ultra-compact, high-speed bi-encoders while preserving semantic search, classification, and retrieval capabilities across the Massive Text Embedding Benchmark (MTEB).

---

## 1. Experimental Models

Evaluated across 3 distinct Teacher $\rightarrow$ Student pairs representing different model capacities and compression tiers:

- **Pair 1 (Ultra-compact Embedding):**
  - **Teacher:** `Qwen/Qwen3-Embedding-0.6B` (600M parameters, hidden dimension $D = 1024$)
  - **Student:** `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base` (22M parameters, 6 layers, hidden dimension $D = 384$)
- **Pair 2 (Mid-size Multilingual Embedding):**
  - **Teacher:** `BAAI/bge-m3` (567M parameters, hidden dimension $D = 1024$)
  - **Student:** `nreimers/MiniLMv2-L12-H768-distilled-from-BERT-Large` (66M parameters, 12 layers, hidden dimension $D = 768$)
- **Pair 3 (BERT Foundation Bi-Encoder):**
  - **Teacher:** `Qwen/Qwen3-Embedding-4B` (4B parameters, hidden dimension $D = 2560$)
  - **Student:** `google-bert/bert-base-uncased` (109M parameters, 12 layers, hidden dimension $D = 768$)

### Pretraining & Initialization Status
- Students start from pretrained encoder base checkpoints (`AutoModel.from_pretrained`).
- Teachers are completely frozen in evaluation mode.

---

## 2. Datasets & Resources

### Training Corpus (Unlabeled)
- Unlabeled corpus `train_full.csv`.
- *Local Workspace Path:* [`datasets/embedding_data/train_full.csv`](../../datasets/embedding_data/train_full.csv)
- *Hugging Face Hub:* [sentence-transformers/all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) & [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)

### Evaluation Benchmarks (Test Splits)
Multi-task evaluation across 9 MTEB benchmark datasets, categorized into In-Domain (*) and Out-of-Domain:
- **Sentence Classification (Macro-F1 %):**
  - `Banking77` (Out-of-Domain): [`datasets/embedding_data/multi-data/banking77_test.csv`](../../datasets/embedding_data/multi-data/banking77_test.csv) — HF: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
  - `Tweet` (Out-of-Domain): [`datasets/embedding_data/multi-data/tweet_test.csv`](../../datasets/embedding_data/multi-data/tweet_test.csv) — HF: [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (sentiment task)
  - `Emotion*` (In-Domain): [`datasets/embedding_data/multi-data/emotion_validation.csv`](../../datasets/embedding_data/multi-data/emotion_validation.csv) — HF: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- **Pair Classification (Average Precision - AP %):**
  - `MRPC` (Out-of-Domain): [`datasets/embedding_data/multi-data/mrpc_test.csv`](../../datasets/embedding_data/multi-data/mrpc_test.csv) — HF: [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue)
  - `SciTail` (Out-of-Domain): [`datasets/embedding_data/multi-data/scitail_test.csv`](../../datasets/embedding_data/multi-data/scitail_test.csv) — HF: [allenai/scitail](https://huggingface.co/datasets/allenai/scitail)
  - `WiC*` (In-Domain): [`datasets/embedding_data/multi-data/wic_validation.csv`](../../datasets/embedding_data/multi-data/wic_validation.csv) — HF: [super_glue](https://huggingface.co/datasets/super_glue)
- **Semantic Textual Similarity (Spearman Rank Correlation $\rho \times 100$):**
  - `SICK` (Out-of-Domain): [`datasets/embedding_data/multi-data/sick_validation.csv`](../../datasets/embedding_data/multi-data/sick_validation.csv) — HF: [sick](https://huggingface.co/datasets/sick)
  - `STS12` (Out-of-Domain): [`datasets/embedding_data/multi-data/sts12_validation.csv`](../../datasets/embedding_data/multi-data/sts12_validation.csv) — HF: [mteb/sts12-sts](https://huggingface.co/datasets/mteb/sts12-sts)
  - `STSB*` (In-Domain): [`datasets/embedding_data/multi-data/stsb_validation.csv`](../../datasets/embedding_data/multi-data/stsb_validation.csv) — HF: [mteb/stsbenchmark-sts](https://huggingface.co/datasets/mteb/stsbenchmark-sts)

---

## 3. Evaluation Metrics

- **Classification:** Macro-F1 (%) evaluated via a standard Linear-Probe classifier.
- **Pair Classification:** Average Precision (AP, %).
- **Semantic Textual Similarity (STS):** Spearman rank correlation coefficient ($\rho \times 100$).
- **Aggregated Scores:**
  - `Avg-In`: Arithmetic mean across 3 in-domain tasks (Emotion, WiC, STSB).
  - `Avg-Out`: Arithmetic mean across 6 out-of-domain tasks (Banking77, Tweet, MRPC, SciTail, SICK, STS12).
  - `Avg.`: Overall arithmetic mean across all 9 MTEB benchmark datasets.

---

## 4. Participating Baselines & Paper Links

| Baseline | Official Paper Reference | Venue / Record | Key Distillation Objective |
| :--- | :--- | :--- | :--- |
| **Teacher** | Empirical Upper Bound | - | Uncompressed frozen teacher embeddings |
| **Student base** | Unmodified pretrained student | - | Direct zero-shot evaluation without distillation |
| **SimCSE-unsup** | *SimCSE: Simple Contrastive Learning of Sentence Embeddings* (Gao et al.) | EMNLP 2021 · [ACL Anthology](https://aclanthology.org/2021.emnlp-main.552/) | Unsupervised dropout contrastive baseline |
| **Jasper and Stella** | *Jasper and Stella: Distillation of SOTA Embedding Models* (Zhang et al.) | arXiv 2024 · [arXiv:2412.19048](https://arxiv.org/abs/2412.19048) | Decoupled 2-stage MLP projector + joint InfoNCE fine-tuning |
| **DistillCSE** | *DistillCSE: Distilled Contrastive Learning for Sentence Embeddings* (Xu et al.) | Findings EMNLP 2023 · [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.547/) | Pairwise batch relational cosine similarity topology matching |
| **EMO** | *EMO: Embedding Model Distillation via Intra-Model Relation and Optimal Transport Alignments* (Truong et al.) | EMNLP 2025 · [ACL Anthology](https://aclanthology.org/2025.emnlp-main.385/) | Intra-model CKA relational structure + optimal transport |
| **MoL** | *MoL: Mixture of Layers in Cross-Tokenizer Embedding Model Distillation* (Vu et al.) | Knowledge-Based Systems 2026 · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705126007276) | Learnable dynamic mixture across intermediate teacher layers |
| **TALAS / TALAS-ASAM** | *TALAS: Teacher-Anchored Layer Alignment with Adaptive Sharpness-Aware Minimization for Embedding Distillation* (Dao et al.) | ACL 2026 · [ACL Anthology](https://aclanthology.org/2026.acl-long.1509/) | Teacher-anchored multi-layer distillation + ASAM flat minima |

---

## 5. Common Generalized Train & Test Scenario

- **Offline Precomputation Phase:** The teacher model precomputes sentence embeddings for all 100,000 lines of `train_full.csv`. These representations are cached to disk so that every baseline optimizes against identical target vectors, removing stochastic variance.
- **Training Phase:** All baselines train exclusively on the unlabeled `train_full.csv` corpus. Each student model receives the raw text, extracts pooled sentence embeddings, and optimizes its parameters according to its respective distillation objective.
- **Testing Phase:** After distillation, student checkpoints are completely frozen. Models undergo Zero-Shot transfer evaluation on all 9 MTEB test sets without any task-specific fine-tuning or adaptation.

---

## 6. Official Empirical Results

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

