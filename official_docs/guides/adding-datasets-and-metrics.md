# Adding Datasets & Custom Metrics

This guide covers registering custom datasets, configuring prompt formatters, and adding domain-specific evaluation metrics.

---

## 1. Adding a Custom Dataset

### Using Local Files (No Code Required)
LM-KD directly loads local JSONL, CSV, and Parquet files:

```yaml
data:
  train:
    type: jsonl
    path: /path/to/my_data/train.jsonl
  eval:
    type: jsonl
    path: /path/to/my_data/val.jsonl
  transforms:
    - type: tokenize_sft
      prompt_field: query
      response_field: solution
  max_length: 2048
```

### Implementing a Custom Data Source
For custom data extraction pipelines, register a source in [`src/lmkd/data/dataset.py`](../../src/lmkd/data/dataset.py):

```python
from lmkd.core.registry import SOURCES

@SOURCES.register("my_domain_data")
def load_domain_data(split: str, **kwargs):
    # Load and clean domain records
    raw_samples = load_from_database(split=split)
    for sample in raw_samples:
        yield {
            "prompt": sample["instruction"],
            "response": sample["demonstration"]
        }
```

---

## 2. Adding a Custom Evaluation Metric

Metrics are registered under `EVALUATORS` and execute during the `lmkd eval` pipeline:

```python
from lmkd.core.registry import EVALUATORS
import re

@EVALUATORS.register("math_boxed_match")
class MathBoxedMatchEvaluator:
    """Evaluates mathematical accuracy by extracting \\boxed{...} answers."""

    def extract_boxed(self, text: str) -> str:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        return match.group(1).strip() if match else ""

    def evaluate(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        correct = 0
        for pred, ref in zip(predictions, references):
            if self.extract_boxed(pred) == self.extract_boxed(ref):
                correct += 1
        return {"boxed_accuracy": correct / len(predictions)}
```

Use your metric in a recipe:
```yaml
eval:
  - type: math_boxed_match
    when: end
```

