# Developing a Custom Distillation Loss

In this tutorial, you will create a custom research loss from scratch without modifying the core `lmkd` codebase: **Adaptive Confidence-Weighted KL (ACW-KL)**.

The idea: When the teacher is uncertain (the margin between its top-1 and top-2 logits is small), the student should rely more on standard ground-truth cross-entropy. When the teacher is highly confident (large margin), the student strongly imitates the teacher's distribution.

---

## 1. Implement the Loss in a Standalone File

Create a file in your project workspace: `plugins/acw_kl.py`:

```python
# plugins/acw_kl.py
from __future__ import annotations
from typing import ClassVar
import torch
import torch.nn.functional as F
from lmkd.core.registry import CRITERIONS
from lmkd.core.plugin import BaseConfig
from lmkd.losses.base import Criterion, LossContext, LossOutput, LossTerm
from lmkd.models.forward import FeatureRequest, SideRequest
from lmkd.teachers.signals import TeacherSignalSpec

@CRITERIONS.register("acw_kl", paper="UserCustomLoss:2025")
class AdaptiveConfidenceWeightedKL(Criterion):
    """Adaptive Confidence-Weighted KL Divergence.

    Weights teacher KL divergence by the top-1 / top-2 margin of teacher logits.
    """

    needs_teacher: ClassVar[bool] = True

    class Config(BaseConfig):
        temperature: float = 1.0
        min_weight: float = 0.1
        margin_scale: float = 2.0

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        self.temp = self.cfg.temperature
        self.min_weight = self.cfg.min_weight
        self.margin_scale = self.cfg.margin_scale

    def features(self) -> FeatureRequest:
        # Request target-aligned logits from both student and teacher
        return FeatureRequest(
            student=SideRequest(logits="view"),
            teacher=SideRequest(logits="view"),
            teacher_signal=TeacherSignalSpec(kind="full"),
        )

    def forward(self, ctx: LossContext) -> LossOutput:
        s_logits = ctx.student_logits().full() / self.temp
        t_logits = (ctx.teacher.logits.full() if ctx.teacher else s_logits) / self.temp

        # 1. Compute teacher confidence margin: top1 - top2
        with torch.no_grad():
            top2_vals, _ = torch.topk(t_logits, k=2, dim=-1)
            margin = top2_vals[..., 0] - top2_vals[..., 1]
            # Dynamic weight between [min_weight, 1.0]
            weight = torch.sigmoid(margin * self.margin_scale).clamp(min=self.min_weight)

        # 2. Compute token-level Forward KL
        p_t = F.softmax(t_logits, dim=-1)
        log_p_s = F.log_softmax(s_logits, dim=-1)
        log_p_t = F.log_softmax(t_logits, dim=-1)

        # KL = sum( P_T * (log P_T - log P_S) )
        kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1) * (self.temp ** 2)

        # 3. Apply confidence weighting and response loss mask
        loss_mask = ctx.supervised_mask("student")
        weighted_kl = (kl * weight * loss_mask).sum()

        return LossOutput(
            terms={
                "acw_kl": LossTerm(value=weighted_kl, reduction="token_sum")
            },
            metrics={
                "mean_teacher_margin": margin.detach().mean(),
                "mean_confidence_weight": weight.detach().mean()
            }
        )
```

---

## 2. Using the Custom Loss in a Recipe

Create your experiment recipe `experiments/my_experiment.yaml` and reference your file in `imports:`:

```yaml
# experiments/my_experiment.yaml
imports:
  - ./plugins/acw_kl.py

_base_:
  - ../recipes/_base_/engine_zero2.yaml

models:
  student:
    path: Qwen/Qwen2.5-0.5B
    dtype: bf16
  teacher:
    path: Qwen/Qwen2.5-1.5B
    dtype: bf16

teacher:
  provider: hf_frozen
  input: same

loss:
  terms:
    - criterion: acw_kl
      params:
        temperature: 1.5
        min_weight: 0.2
        margin_scale: 1.5

data:
  train:
    type: hf_dataset
    path: databricks/databricks-dolly-15k
    split: train
  transforms:
    - type: tokenize_sft
      use_chat_template: true
  max_length: 1024
```

---

## 3. Verify and Train

You can smoke-test your criterion gradient flow on synthetic tensors without loading real models:

```bash
lmkd try acw_kl --plugins ./plugins/acw_kl.py
```

Run the recipe configuration check to ensure your custom plugin loads and validates against the schema:

```bash
lmkd check -c experiments/my_experiment.yaml
```

Launch training:
```bash
lmkd train -c experiments/my_experiment.yaml --set train.max_steps=100
```

Notice how your custom metrics (`mean_teacher_margin` and `mean_confidence_weight`) automatically appear in the TensorBoard/W&B training dashboards!

