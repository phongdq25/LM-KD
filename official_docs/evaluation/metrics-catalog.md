# Metrics Catalog

LM-KD tracks a rigorous, codebase-verified catalog of evaluation metrics and answer graders implemented directly in [`src/lmkd/eval/`](../../src/lmkd/eval/). All metric definitions and grading logic strictly match the toolkit runtime.

---

## 1. Evaluator Plugins (`EVALUATORS`)

The framework registers 6 standardized evaluators under the [`EVALUATORS`](../../src/lmkd/eval/evaluators.py) registry:

| Evaluator Plugin | Implemented Class | Target Domain | Metrics Computed |
| :--- | :--- | :--- | :--- |
| **`dev_loss`** | [`DevLoss`](../../src/lmkd/eval/evaluators.py#L31-L117) | Training validation | `dev_loss/loss`, term components (teacher-forced cross-entropy) |
| **`generation_rouge`** | [`GenerationRouge`](../../src/lmkd/eval/evaluators.py#L136-L221) | Open-ended instruction | `exact_match`, `rougeL` (stemmed ROUGE-L F1 $\times 100$) |
| **`math_passk`** | [`MathPassK`](../../src/lmkd/eval/evaluators.py#L223-L347) | Multi-step mathematical reasoning | `pass@1`, `pass@k`, `any_correct`, `truncation_rate`, `no_boxed_rate`, `mean_len` |
| **`embedding_sts`** | [`EmbeddingSTS`](../../src/lmkd/eval/embedding.py#L200-L232) | Sentence embeddings | `sts/<name>/spearman` (Spearman rank correlation $\rho \times 100$) |
| **`embedding_linear_probe`** | [`EmbeddingLinearProbe`](../../src/lmkd/eval/embedding.py#L234-L268) | Classification probing | `cls/<name>/macro_f1`, `cls/<name>/accuracy` ($\times 100$) |
| **`embedding_pair`** | [`EmbeddingPair`](../../src/lmkd/eval/embedding.py#L270-L303) | Pairwise classification | `pair/<name>/ap` (Average Precision $\times 100$) |

---

## 2. Mathematical Reasoning Metrics (`math_passk`)

Implemented in [`src/lmkd/eval/metrics.py`](../../src/lmkd/eval/metrics.py):

### Unbiased Pass@k Estimator
Following Chen et al. (2021), when sampling $n$ completions per problem with $c$ correct solutions:
$$\text{pass}@k = \mathbb{E}\left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right] = 1 - \prod_{i=n-c+1}^n \left( 1 - \frac{k}{i} \right)$$
- **`pass@1`:** The mean accuracy over all generations across all problems (avoiding single-column evaluation bias).
- **`pass@k`:** The unbiased estimator averaged over all benchmark problems where $n_i \ge k$.
- **`any_correct`:** Fraction of problems with at least one correct generated solution ($\ge 1$).

### Trajectory Diagnostics
- **`truncation_rate`:** Fraction of generations truncated by context window limits (`finish_reason == "length"`).
- **`no_boxed_rate`:** Fraction of non-empty generations that fail to produce a `\boxed{...}` answer.
- **`mean_len`:** Average number of generated tokens per problem trajectory.

---

## 3. Answer Graders (`GRADERS`)

Graders in [`src/lmkd/eval/graders.py`](../../src/lmkd/eval/graders.py) score candidate responses against gold reference answers:

| Grader Name | Class | Implementation Mechanism |
| :--- | :--- | :--- |
| **`exact_match_norm`** | `ExactMatchNormGrader` | Normalizes text (lowercase, punctuation, whitespace removal) and checks string equality. |
| **`rouge_l`** | `RougeLGrader` | Stemmed ROUGE-L F1 score in $[0, 1]$ using Google's `rouge_score` library. |
| **`boxed_builtin`** | `BoxedBuiltinGrader` | Robust nested brace matching extracting `\boxed{...}`, followed by `normalize_math` LaTeX cleaning. |
| **`math_verify`** | `MathVerifyGrader` | Symbolic mathematical parsing and verification using the `math_verify` package with POSIX timeout safety. |
| **`oat`** | `OatGrader` | Ported from Garena Online OAT math reward function (`boxed_reward_fn`). |
| **`palign`** | `PAlignGrader` | Composite grader: accepts a prediction if **either** `math_verify` OR `oat` accepts it. |
| **`mcq`** | `MCQGrader` | Regular expression scanning from the response tail to extract multiple-choice option letters (A–D). |

---

## 4. Dense Embedding Metrics (`src/lmkd/eval/embedding.py`)

- **Spearman Correlation ($\rho$):**
  Given cosine similarity between student sentence representations $e_1, e_2$ mapped to scores $s = (\cos(e_1, e_2) + 1.0) \times 2.5$, computes:
  $$\rho = \frac{\sum (R(s_i) - \bar{R}_s)(R(y_i) - \bar{R}_y)}{\sqrt{\sum (R(s_i) - \bar{R}_s)^2 \sum (R(y_i) - \bar{R}_y)^2}}$$
- **Linear Probe Macro-F1:** Trains a logistic regression classifier on frozen student representations (`x_train = encode_texts(...)`) and evaluates Macro-F1 on held-out test splits.
- **Pairwise Average Precision (AP):** Scans 200 threshold cutoffs over cosine similarities to compute the exact Area Under the Precision-Recall Curve.
