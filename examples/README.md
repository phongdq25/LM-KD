# LM-KD Examples Directory

This folder contains clean, self-contained examples demonstrating different usage patterns of the LM-KD framework:
from 1-second in-memory quickstarts, to custom loss definitions, manual training loops, and offline smoke tests.

---

## Directory Index

| File / Folder | Purpose | Command |
|---|---|---|
| [`quickstart_cpu.py`](quickstart_cpu.py) | **Zero-download 5-step distillation on CPU.** In-memory toy student/teacher models and dataset. | `python examples/quickstart_cpu.py` |
| [`custom_loop.py`](custom_loop.py) | **Custom Python training loop.** Assembles runtime from YAML recipe, attaches a custom evaluation metric, and runs a manual `while not rt.done: rt.train_step()` loop. | `python examples/custom_loop.py -c recipes/same_tokenizer/fkl.yaml --set train.max_steps=10` |
| [`my_losses.py`](my_losses.py) | **Defining custom losses in minutes.** Shows Tier-0 (`@lmkd.compute_loss`) and Tier-1 (`@lmkd.register_loss`) APIs with no core changes. | `lmkd train -c examples/recipes/tier1_fkl.yaml --set train.max_steps=10` |
| [`new_method.py`](new_method.py) | **Full new-method development demo.** Creates target-aligned custom loss with confidence masking, packages it into a runnable CPU pipeline. | `python examples/new_method.py --steps 2` |
| [`reproduce_tiny.py`](reproduce_tiny.py) | **Offline smoke test across all 6 baseline categories.** Fast end-to-end execution of FKL, ULD, SGL, GRPO, DPO, and TALAS on synthetic data. | `python examples/reproduce_tiny.py --output-dir /tmp/tiny-repro` |
| [`rl_kd_single_forward.py`](rl_kd_single_forward.py) | **Unified RL + Distillation with single student forward.** Demonstrates high-throughput on-policy generation with joint policy optimization and distillation. | `python examples/rl_kd_single_forward.py --help` |
| [`data/train_sample.jsonl`](data/train_sample.jsonl) | Sample input JSONL dataset for smoke testing recipes. | — |
| [`recipes/fkl_full.yaml`](recipes/fkl_full.yaml) | **Master Configuration Template.** 14 fully annotated configuration sections with all valid options and explanations. | `lmkd check -c examples/recipes/fkl_full.yaml` |

---

## Walkthroughs

### 1. Zero-Download Quickstart (`quickstart_cpu.py`)
Run an immediate 5-step knowledge distillation run on CPU without needing GPU hardware, HuggingFace downloads, or configuration files:
```bash
python examples/quickstart_cpu.py
```

### 2. Rapid Prototyping of a Custom Loss (`my_losses.py`)
Add a new loss function without editing framework internals:
```python
import lmkd
import torch.nn.functional as F

@lmkd.register_loss("my_fkl", teacher=True, reduction="token_sum")
def my_fkl(s, t, batch, temperature: float = 1.0):
    """Forward KL on HuggingFace-style logits."""
    s_logits = s.logits[:, :-1] / temperature
    t_logits = t.logits[:, :-1] / temperature
    mask = (batch["student.labels"][:, 1:] != -100).float()
    per_token = -(F.softmax(t_logits, -1) * F.log_softmax(s_logits, -1)).sum(-1)
    return (per_token * mask).sum()
```
Run this loss directly in a recipe:
```bash
lmkd train -c examples/recipes/tier1_fkl.yaml --set train.max_steps=10
```

### 3. Writing Your Own Training Loop (`custom_loop.py`)
Retain full Python control over the training loop while preserving all framework semantics (gradient accumulation,
mixed precision, DeepSpeed, parameter schedulers, and callbacks):
```python
import lmkd

rt = lmkd.build("recipes/same_tokenizer/fkl.yaml")
rt.begin()
while not rt.done:
    out = rt.train_step(loss_fn=my_custom_loss)
    if rt.state.step % 100 == 0:
        print(f"Step {rt.state.step} | Loss: {out.metrics['loss/total']:.4f}")
rt.end()
```
Run with:
```bash
python examples/custom_loop.py -c recipes/same_tokenizer/fkl.yaml --set train.max_steps=20
```

