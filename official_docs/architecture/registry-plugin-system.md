# Registry & Plugin System

Extensibility in LM-KD is driven by a strict, metadata-rich plugin registry. Every core component—from loss functions and dataset loaders to optimizers and rollout engines—is registered through declarative registry tables.

---

## 1. Registry Architecture

The registry system is managed in [`lmkd/core/registry.py`](../../src/lmkd/core/registry.py).

### Core Registries Table
| Registry Name | Python Object | Description |
| :--- | :--- | :--- |
| `CRITERIONS` | `Registry[Criterion]` | Divergence criteria & distillation loss functions |
| `PRODUCERS` | `Registry[BatchProducer]` | Batch formatting, collation, and prompt masking |
| `TEACHERS` | `Registry[TeacherProvider]` | Teacher inference providers and logit caches |
| `ALIGNERS` | `Registry[TokenAligner]` | Cross-tokenizer sequence alignment algorithms |
| `OPTIMIZERS` | `Registry[Optimizer]` | Standard and experimental parameter optimizers |
| `ENGINES` | `Registry[Engine]` | Distributed execution engines (DeepSpeed, FSDP) |
| `ROLLOUT` | `Registry[RolloutEngine]` | On-policy generation engines (HF, vLLM) |
| `EVALUATORS` | `Registry[Evaluator]` | Benchmark evaluators and task assessment suites |

---

## 2. Registering a Component

To implement a new component, inherit from the corresponding base class, define a Pydantic `Config` class for parameter validation, and apply the `@REGISTRY.register` decorator:

```python
from lmkd.core.registry import CRITERIONS
from lmkd.core.plugin import BaseConfig
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest
import torch

@CRITERIONS.register(
    "temperature_mse",
    paper="arXiv:2305.xxxx",
    source="my_project/losses/mse.py",
    status="stable"
)
class TemperatureMSELoss(Criterion):
    """Custom temperature-scaled Mean Squared Error between logits."""

    class Config(BaseConfig):
        temperature: float = 1.0
        weight: float = 1.0

    def __init__(self, config: Config):
        super().__init__()
        self.temp = config.temperature
        self.weight = config.weight

    def features(self) -> FeatureRequest:
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        s_logits = ctx.student_logits().full() / self.temp
        t_logits = ctx.teacher.logits.full() / self.temp if ctx.teacher else s_logits
        mask = ctx.supervised_mask("student")
        
        diff = (s_logits - t_logits).pow(2).sum(dim=-1)
        weighted_loss = (diff * mask).sum() * self.weight
        return LossOutput(
            terms={"mse_loss": LossTerm(value=weighted_loss, reduction="token_sum")}
        )
```

---

## 3. Parameter Validation with Pydantic

LM-KD strictly validates all configuration dictionaries at instantiation time:
- **Type Checking:** If `temperature: "invalid"` is passed in YAML, a descriptive `ValidationError` is raised immediately.
- **Extra Key Rejection:** Unexpected configuration keys trigger an error to prevent silent typos (e.g. `temprature: 2.0`).
- **Default Resolution:** Omitted keys cleanly adopt their declared Pydantic defaults.

---

## 4. Plugin Discovery & Dynamic Loading

LM-KD discovers plugins across three distinct mechanisms:

1. **Automatic Discovery:** All modules placed within `src/lmkd/` are automatically scanned and indexed.
2. **Explicit Recipe Imports:** You can load standalone Python files directly from your YAML recipe without installing them as packages:
   ```yaml
   imports:
     - ./custom_losses/my_loss.py
     - my_external_package.plugins

   loss:
     terms:
       - criterion: temperature_mse
         params:
           temperature: 2.0
   ```
3. **CLI Dynamic Loading:** Use the `--plugins` argument to import plugins at execution time:
   ```bash
   lmkd train -c recipe.yaml --plugins ./custom_losses/my_loss.py
   ```

---

## 5. Plugin Scaffolding: `lmkd new`

To accelerate development, LM-KD provides a built-in code generator that creates boilerplates with typed configs, unit tests, and minimal recipes:

```bash
# Generate a new criterion plugin
lmkd new criterion my_custom_kd

# Generate a new token aligner
lmkd new aligner custom_dtw
```

The tool generates:
- The implementation file in `src/lmkd/losses/my_custom_kd.py`.
- A corresponding unit test in `tests/test_my_custom_kd.py`.
- A minimal recipe in `recipes/testing/my_custom_kd.yaml`.

