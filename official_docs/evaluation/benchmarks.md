# Benchmark Suites

LM-KD integrates evaluation benchmark suites directly into its execution runtime and recipe configurations. Benchmarks can be evaluated via `lmkd eval` or automatically during training checkpoints.

---

## 1. Mathematical & Reasoning Benchmarks

Handled by [`MathPassK`](../../src/lmkd/eval/evaluators.py#L223) and [`BenchmarkSource`](../../src/lmkd/eval/benchmarks.py#L168-L192):

| Benchmark Key | Loader / Dataset | Problem Count & Domain | Default Grader |
| :--- | :--- | :--- | :--- |
| **`math500`** | `HuggingFaceH4/MATH-500` | 500 challenging high-school competition math problems | `math_verify` |
| **`aime24`** | `Maxwell-Jia/AIME_2024` | 30 American Invitational Mathematics Exam 2024 problems | `math_verify` |
| **`aime25`** | `yentinglin/aime_2025` | 30 American Invitational Mathematics Exam 2025 problems | `math_verify` |
| **`amc23`** | `math-ai/amc23` | 40 problems from AMC 12A / 12B (2023) | `math_verify` |
| **`amc12`** | `AI-MO/aimo-validation-amc`| 83 AMC 12 competition problems (2022–2023) | `math_verify` |
| **`minerva`** | `math-ai/minervamath` | 272 undergraduate quantitative STEM problems | `math_verify` |
| **`olympiadbench`**| `Hothan/OlympiadBench` | English open-ended text competition subset | `math_verify` |
| **`gpqa`** | `Idavidrein/gpqa` | High-difficulty diamond subset multiple choice | `mcq` |

### Running Reasoning Evaluation
```bash
lmkd eval -c recipes/long_cot/sgl_spectral.yaml \
    --ckpt ./outputs/student_checkpoint
```

---

## 2. Instruction & Generation Benchmarks

Handled by [`GenerationRouge`](../../src/lmkd/eval/evaluators.py#L136):

| Benchmark | Dataset Path | Primary Metric |
| :--- | :--- | :--- |
| **`dolly`** | [`datasets/same_cross_tokenizer_data/dolly/dev.jsonl`](../../datasets/same_cross_tokenizer_data/dolly/dev.jsonl) | `rougeL`, `exact_match` |
| **`self-inst`** | [`datasets/same_cross_tokenizer_data/self-inst/valid.jsonl`](../../datasets/same_cross_tokenizer_data/self-inst/valid.jsonl) | `rougeL`, `exact_match` |
| **`vicuna`** | [`datasets/same_cross_tokenizer_data/vicuna/valid.jsonl`](../../datasets/same_cross_tokenizer_data/vicuna/valid.jsonl) | `rougeL`, `exact_match` |
| **`sinst`** (S-NI) | [`datasets/same_cross_tokenizer_data/sinst/11_/valid.jsonl`](../../datasets/same_cross_tokenizer_data/sinst/11_/valid.jsonl) | `rougeL`, `exact_match` |

---

## 3. Sentence Embedding Suite (9 Tasks)

Handled by [`EmbeddingSTS`](../../src/lmkd/eval/embedding.py#L200), [`EmbeddingLinearProbe`](../../src/lmkd/eval/embedding.py#L234), and [`EmbeddingPair`](../../src/lmkd/eval/embedding.py#L270):

```bash
lmkd eval -c recipes/embedding/talas_m.yaml \
    --ckpt ./outputs/student_checkpoint
```

- **In-Domain STS:** `stsb`, `sick`, `sts12` (Spearman $\rho \times 100$).
- **Pair Classification & Entailment:** `mrpc`, `scitail`, `wic` (Accuracy / AP $\times 100$).
- **Transfer Classification Probing:** `tweet`, `banking77`, `emotion` (Macro-F1 $\times 100$).
