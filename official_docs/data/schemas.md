# Internal Data Schemas

This document details the standardized input data schemas supported by **LM-KD** across all data transforms, collators, and baseline tasks.

Each schema explicitly delineates:
- **Required Fields:** Mandatory fields required for tokenization, collation, and model training.
- **Optional Fields:** Metadata, verifier helpers, or auxiliary fields that **can be safely omitted**.
- **Supported Aliases:** Alternative field names automatically recognized and resolved by the codebase.

---

## 1. Instruction & SFT Distillation Schema

Used in Supervised Fine-Tuning and Knowledge Distillation with shared or cross-family tokenizers (e.g., Dolly-15k, Self-Instruct, Vicuna).

### Full Example:
```json
{
  "id": "dolly_sample_001",
  "system": "You are a helpful, respectful AI assistant.",
  "prompt": "What is the difference between nuclear fission and nuclear fusion?",
  "response": "Nuclear fission is the splitting of a heavy atomic nucleus into lighter nuclei, whereas nuclear fusion is the combining of light atomic nuclei to form a heavier nucleus.",
  "metadata": {
    "category": "science",
    "source": "databricks/dolly-15k"
  }
}
```

### Minimal Example:
```json
{
  "prompt": "What is the difference between nuclear fission and nuclear fusion?",
  "response": "Nuclear fission is the splitting of a heavy atomic nucleus into lighter nuclei..."
}
```

### Field Breakdown:
| Field | Status | Description & Notes |
| :--- | :---: | :--- |
| `prompt` | **Required** | The input instruction or query. Automatically falls back to aliases: `instruction`, `problem`, `question`, `Question`, or user turn in `conversations` / `messages`. |
| `response` | **Required** | The target output or teacher completion. Automatically falls back to aliases: `output`, `solution`, `target`, `answer`, `Answer`, or assistant turn in `conversations` / `messages`. |
| `id` | **Optional** (Can omit) | Unique sample identifier for tracking or caching. |
| `system` | **Optional** (Can omit) | System prompt. Tokenizers handle this automatically via Jinja2 `chat_template`; not required as a standalone root field. |
| `metadata` | **Optional** (Can omit) | Arbitrary dictionary containing supplemental details (e.g., category, source). |

---

## 2. Long-CoT & Mathematical Reasoning Schema

Used in step-by-step reasoning distillation (Long-CoT), On-Policy self-distillation (OPSD, SDPO, RLSD), and math benchmarks (MATH500, AIME, AMC).

### Full Example:
```json
{
  "id": "math_aime_2024_03",
  "problem": "Find the number of positive integers n less than 1000 such that...",
  "solution": "Let's analyze the divisibility condition. First, notice that n must be congruent to 3 mod 7...\n\nTherefore, the answer is \\boxed{142}.",
  "answer": "142"
}
```

### Minimal Example (On-Policy RL Loop):
```json
{
  "problem": "Find the number of positive integers n less than 1000 such that...",
  "answer": "142"
}
```

### Field Breakdown:
| Field | Status | Description & Notes |
| :--- | :---: | :--- |
| `problem` | **Required** | Problem description or prompt. Automatically falls back to aliases: `question`, `input`. |
| `solution` | **Required for SFT** | Full reasoning chain ending with the final boxed answer. Falls back to `output`, `response`. In pure on-policy RL rollout (prompt-only generation), this can be omitted if `answer` is provided. |
| `answer` | **Recommended / Optional** | Ground-truth short answer used by `math_verify`. **Can be omitted** if `solution` already contains `\boxed{...}`, as the verifier automatically extracts it via `last_boxed()`. |
| `id` | **Optional** (Can omit) | Problem identifier. |
| `thought_chain` | **Redundant** (Do not need) | Reasoning traces belong directly inside `solution` (or wrapped in `<think>...</think>`). |
| `steps` | **Redundant** (Do not need) | Disjoint step lists are not consumed by the raw data collators. |
| `boxed_answer` | **Redundant** (Do not need) | Automatically parsed by `last_boxed()` or derived from `answer`. |
| `final_answer` | **Redundant** (Do not need) | Use `answer` instead. |

---

## 3. Preference Distillation Schema

Used in reference-free preference distillation and direct alignment algorithms (DPO, IPO, SimPO, PBSD, CTPD).

### Full Example:
```json
{
  "id": "pref_pair_042",
  "prompt": "How can I improve my focus while studying?",
  "chosen": "Here are four science-backed strategies to improve study focus:\n1. Use the Pomodoro technique...",
  "rejected": "Just sit down and force yourself to read without checking your phone.",
  "teacher_scores": [0.88, 0.21]
}
```

### Minimal Example:
```json
{
  "prompt": "How can I improve my focus while studying?",
  "chosen": "Here are four science-backed strategies to improve study focus...",
  "rejected": "Just sit down and force yourself to read without checking your phone."
}
```

### Field Breakdown:
| Field | Status | Description & Notes |
| :--- | :---: | :--- |
| `prompt` | **Required** | The user prompt. Can also be inferred if `chosen` and `rejected` share an identical conversational prefix. |
| `chosen` | **Required** | The preferred response (winning candidate). Alternatively formatted as `responses: [chosen, rejected]`. |
| `rejected` | **Required** | The dispreferred response (losing candidate). |
| `id` | **Optional** (Can omit) | Pair identifier. |
| `teacher_scores` | **Optional** (Can omit) | Array of teacher ratings `[score_chosen, score_rejected]`. Only required by hierarchical/offline teacher-guided criteria (e.g., PBSD, CTPD, DP3O). Standard DPO/SimPO does not require this. |
| `margin` | **Optional** (Can omit) | Precomputed reward margin. Dynamically computed when needed. |

---

## 4. Sentence Embedding Schema

Used in representation distillation and encoder fine-tuning (e.g., TALAS, SimCSE, EMO).

### A. Unsupervised Representation Training Format (JSONL or CSV):
Real-world example from `datasets/embedding_data/train_full.csv`:
```csv
text,source
"i am feeling a little nervous and anxious but never second guessing my decision",EMOTION
```
Or JSONL:
```json
{
  "text": "A dog chasing a frisbee across a grassy park."
}
```

### B. Pair Evaluation / Probing Format (STS & Classification):
Real-world example from `datasets/embedding_data/multi-data/sts12_test.csv`:
```csv
sentence1,sentence2,score
"the problem likely will mean corrective changes","he said the problem needs to be corrected",4.4
```

### Field Breakdown:
| Field | Status | Description & Notes |
| :--- | :---: | :--- |
| `text` | **Required (Training)** | Single sentence or document to encode (fallback: `query`). Read by `TokenizeText`. |
| `sentence1`, `sentence2` | **Required (Pair Eval)** | Sentence pair to evaluate semantic similarity or entailment. |
| `score` / `label` | **Required (Pair Eval)** | Continuous similarity score (STS) or discrete classification label (0/1). |
| `positive`, `negative` | **Optional** (Can omit) | Only needed for explicit triplet contrastive pipelines; standard unsupervised embedding distillation (TALAS, SimCSE) requires **only `text`**. |
| `similarity_label`, `task` | **Optional** (Can omit) | Supplementary task metadata; not used in standard training loops. |
| `source` / `idx` | **Optional** (Can omit) | Dataset provenance tracking. |
