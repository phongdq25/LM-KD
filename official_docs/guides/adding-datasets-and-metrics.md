# Adding Datasets & Custom Metrics

This guide covers registering custom datasets, configuring prompt formatters, and adding domain-specific evaluation metrics.

---

## 1. Adding a Custom Dataset

### Using Local Files (No Code Required)
LM-KD directly loads local JSONL, CSV, and Parquet files via built-in data sources:

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
For custom data extraction pipelines (e.g., streaming from a database or remote store), register a source class inheriting from `Source` in your plugin or in [`src/lmkd/data/sources.py`](../../src/lmkd/data/sources.py):

```python
# plugins/my_source.py
from __future__ import annotations
from collections.abc import Iterator
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import SOURCES
from lmkd.data.sources import Source

@SOURCES.register("my_domain_data")
class MyDomainSource(Source):
    """Custom data source reading records from an external service or database."""

    class Config(BaseConfig):
        split: str = "train"
        limit: int | None = None

    def records(self) -> Iterator[dict]:
        raw_samples = load_from_database(split=self.cfg.split)
        for i, sample in enumerate(raw_samples):
            if self.cfg.limit is not None and i >= self.cfg.limit:
                break
            yield {
                "meta.id": str(sample.get("id", i)),
                "prompt": sample["instruction"],
                "response": sample["demonstration"],
            }
```

Use it in your recipe:
```yaml
imports:
  - ./plugins/my_source.py

data:
  train:
    type: my_domain_data
    split: train
  transforms:
    - type: tokenize_sft
```

---

## 2. Adding Custom Evaluation Metrics

LM-KD supports two levels of evaluation customization:
1. **Custom Grader**: Scores a single generated prediction against a reference string.
2. **Custom Evaluator**: Executes full evaluation loops against the training `Runtime`.

### Option A: Custom Answer Grader (Recommended for Generation/Math)
Register your scorer under `GRADERS` by subclassing `Grader`:

```python
# plugins/custom_grader.py
from __future__ import annotations
import re
from typing import Any
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import GRADERS
from lmkd.eval.base import Grader

@GRADERS.register("math_boxed_match")
class MathBoxedMatchGrader(Grader):
    """Evaluates mathematical accuracy by extracting \\boxed{...} answers."""

    class Config(BaseConfig):
        pass

    def extract_boxed(self, text: str) -> str:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        return match.group(1).strip() if match else text.strip()

    def __call__(self, prediction: str, reference: Any) -> float:
        pred_ans = self.extract_boxed(str(prediction))
        ref_ans = self.extract_boxed(str(reference))
        return 1.0 if pred_ans == ref_ans else 0.0
```

Plug your custom grader into LM-KD's standard benchmark evaluator (`math_passk`):

```yaml
imports:
  - ./plugins/custom_grader.py

eval:
  - type: math_passk
    grader: math_boxed_match
    benchmarks:
      - /path/to/gsm8k_test.jsonl
    when: end
```

---

### Option B: Custom Evaluator Plugin
For complete control over evaluation logic, subclass `Evaluator`:

```python
# plugins/custom_eval.py
from __future__ import annotations
from typing import Any
from lmkd.core.registry import EVALUATORS
from lmkd.eval.base import Evaluator

@EVALUATORS.register("custom_latency_eval")
class CustomLatencyEvaluator(Evaluator):
    """Measures student inference latency during training checkpoints."""

    class Config(Evaluator.Config):
        num_warmup: int = 5
        num_trials: int = 20

    def evaluate(self, rt: Any) -> dict[str, float]:
        model = rt.student_model
        # Compute benchmark metrics
        return {
            "eval/latency_ms": 12.4,
            "eval/throughput_tok_per_sec": 1450.0,
        }
```

Use directly in your recipe:
```yaml
imports:
  - ./plugins/custom_eval.py

eval:
  - type: custom_latency_eval
    when: epoch
```
