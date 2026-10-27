# LM-KD — Unified Large Language Model Knowledge Distillation Framework

<p align="center">
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?style=flat-square&logo=pytorch" alt="PyTorch"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?style=flat-square&logo=python" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green.svg?style=flat-square" alt="License"></a>
</p>

**LM-KD** is an open-source, production-grade knowledge distillation framework for Large Language Models.
It unifies over **90 published distillation baselines** across token-level logit transfer, cross-tokenizer alignment,
Long-Chain-of-Thought (Long-CoT) reasoning compression, on-policy self-distillation, preference alignment, and
dense embedding distillation behind **one canonical, highly-optimized training loop**.

---

## Highlights

* **One Canonical Training Loop (`rt.train_step`)**: Eliminates code duplication across research repositories. Every baseline is expressed as a modular configuration of plugins.
* **Zero $[B, T, V]$ Memory Blowup**: Evaluates divergences on target-aligned chunks via `LogitsView`, slashing peak VRAM by **over 60%** without sacrificing mathematical precision.
* **90+ Published Baselines**: Faithfully replicates canonical papers including DistiLLM, GKD, DSKDv2, ULD, MinED, E-SCOT, SSFT, P-ALIGN, SGL-Spectral, OPSD, SDPO, RLSD, DPO, SimPO, KTO, and TALAS.
* **Production Distributed Backends**: Seamless execution on single GPU/CPU, DeepSpeed ZeRO-1/2/3, native PyTorch FSDP2, and vLLM PagedAttention rollout generation.
* **Extensible & Hackable**: Define custom loss functions, transforms, or optimizers in a single Python file using `@lmkd.loss` or subclassing `Criterion` without modifying framework internals.
* **Dual Execution Modes**: Full CLI workflow with declarative YAML recipes plus an ergonomic Python library API (`lmkd.build`, `rt.fit()`, `rt.train_step()`).

---

## Navigation & Documentation

| Guide | Description |
|---|---|
| 🚀 [**Getting Started Guide**](docs/GETTING_STARTED.md) | Beginner-friendly tutorial: installation, concepts, and your first distillation run. |
| 📚 [**Baseline Reproduction Guide**](docs/BASELINE_REPRODUCTION.md) | Step-by-step reproduction instructions and commands for all 90+ supported papers. |
| 🏗️ [**Architecture in 15 Minutes**](docs/ARCHITECTURE.md) | High-level architectural walkthrough: the 6 axes, execution lifecycle, and design principles. |
| 🧪 [**Developing a New Method**](docs/NEW_METHOD_GUIDE.md) | Quickstart guide for researchers adding new distillation losses or token aligners. |
| 📋 [**Master Recipe Template**](examples/recipes/fkl_full.yaml) | Fully annotated YAML recipe detailing all 14 configuration sections and available parameters. |
| 🧑‍💻 [**Contributor Guide**](docs/CONTRIBUTING.md) | In-depth engineering standards, architectural contracts (C1–C17), and testing criteria. |
| ⚡ [**Performance & Benchmarks**](docs/performance.md) | VRAM scaling, throughput measurements, and fused Liger Kernel configurations. |

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/LM-KD.git
cd LM-KD/lm-kd

# Create a virtual environment
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"
```

Verify your installation with the automated test suite:
```bash
pytest tests/unit/
```

### 2. Instant Zero-Download Distillation (30 Seconds on CPU)

Run an immediate in-memory distillation loop without downloading external weights or datasets:
```bash
python examples/quickstart_cpu.py
```

### 3. Training with a YAML Recipe

Launch distillation using a pre-configured baseline recipe:
```bash
# 1. Validate recipe schema and dependencies statically
lmkd check -c recipes/same_tokenizer/fkl.yaml

# 2. Run a 1-batch GPU preflight smoke test
lmkd check -c recipes/same_tokenizer/fkl.yaml --preflight

# 3. Launch training on single GPU
CUDA_VISIBLE_DEVICES=0 lmkd train -c recipes/same_tokenizer/fkl.yaml \
    --set train.max_steps=500 optim.lr=2e-5
```

---

## Baseline Reproduction Matrix

LM-KD models every distillation method as a coordinate across **six orthogonal axes**:
**A** (Data Transforms) $\times$ **B** (Batch Producers) $\times$ **C** (Teacher Signals) $\times$ **D** (Objectives) $\times$ **E** (Step Strategies) $\times$ **F** (Optimizers).

| Research Family | Key Representative Baselines | Loss / Alignment Mechanism | Recipe Directory |
|---|---|---|---|
| **Same-Tokenizer Logit KD** | DistiLLM, GKD, CSD, TVD, JSD, SFKL, SRKL | Forward/Reverse KL, Adaptive SGO, Mixed rollouts, CSD surrogate | `recipes/same_tokenizer/` |
| **Cross-Tokenizer KD** | DSKD, DSKDv2, ETA, ULD, MinED, MLOT, E-SCOT, IMPACT | Exact token matching, Edit-distance DTW, Optimal Transport, BPIA | `recipes/cross_tokenizer/` |
| **Long-CoT Reasoning** | SSFT, P-ALIGN, SGL-Spectral, IWC, SARW | Integrated Gradients, Student Prefix Search, Spectral SVD Energy | `recipes/long_cot/` |
| **On-Policy / Self-Distillation** | OPSD, SDPO, RLSD, Tinker RKL | Self-teacher EMA, Group rollouts, Math verifiers, PPO clipped policy | `recipes/self_distill/` |
| **Preference Distillation** | DPO, IPO, SimPO, KTO, TDPO, β-DPO, PBSD | Pairwise preference margins, Token-dependent DPO, Kahneman-Tversky | `recipes/preference/` |
| **Embedding Distillation** | TALAS-M, TALAS-ASAM, SimCSE, TAMD | Multi-head CLS attention, Cosine/MSE distances, Muon optimizer | `recipes/embedding/` |

List all 90+ supported baselines via CLI:
```bash
bash scripts/run_baseline.sh --list
```

For complete reproduction commands, data contracts, and paper citations, see the [**Baseline Reproduction Guide**](docs/BASELINE_REPRODUCTION.md).

---

## Python API (Library Mode)

LM-KD can be imported and executed as a standard Python library, granting full programmatic control over training:

```python
import lmkd

# Launch a training run from a recipe with overrides
lmkd.run("recipes/same_tokenizer/fkl.yaml", overrides=["train.max_steps=100"])

# Or construct a custom training loop while retaining all framework semantics:
rt = lmkd.build("recipes/same_tokenizer/fkl.yaml", evaluators=[my_evaluator_fn])
rt.begin()
while not rt.done:
    out = rt.train_step(loss_fn=my_custom_loss)
    if rt.state.step % 50 == 0:
        print(f"Step {rt.state.step} | Loss: {out.metrics['loss/total']:.4f}")
rt.end()
```

`lmkd.build` also accepts live in-memory PyTorch models, tokenizers, and dataset records directly:
```python
rt = lmkd.build(
    recipe_dict,
    student_model=my_student,
    student_tokenizer=my_tok,
    teacher_model=my_teacher,
    train_records=[{"prompt": "Hello", "response": "World!"}],
)
rt.fit()
```

---

## Prototyping a Custom Loss in Minutes

Define a custom distillation objective using the `@lmkd.loss` decorator. No registry lists to edit, no core changes:

```python
import lmkd
from lmkd import Inputs, TokenLoss

@lmkd.loss("confident_fkl", teacher=True, chunk_size=1024)
def confident_fkl(s: Inputs, t: Inputs, temperature: float = 2.0, threshold: float = 0.5) -> TokenLoss:
    """Forward KL evaluated only where teacher confidence exceeds a threshold."""
    s_logp = (s.logits / temperature).log_softmax(-1)
    t_logp = (t.logits / temperature).log_softmax(-1)
    t_prob = t_logp.exp()
    
    kl_divergence = (t_prob * (t_logp - s_logp)).sum(-1) * (temperature ** 2)
    mask = t_prob.max(-1).values >= threshold
    return TokenLoss(values=kl_divergence, mask=mask)
```

Plug your custom loss into any YAML recipe via `imports: [my_custom_loss.py]`:
```yaml
loss:
  terms:
    - criterion: nll
    - criterion: confident_fkl
      weight: 1.0
      params:
        temperature: 2.0
        threshold: 0.6
```

---

## Distributed & Multi-GPU Training

LM-KD provides built-in multi-GPU distributed launchers in `scripts/`:

### DeepSpeed ZeRO-2 / ZeRO-3
```bash
NUM_GPUS=4 bash scripts/train_multigpu_deepspeed.sh recipes/same_tokenizer/fkl_zero2.yaml \
    --set train.batch_size=64 train.micro_batch_size=4
```

### PyTorch Native FSDP2
```bash
NUM_GPUS=4 bash scripts/train_multigpu_fsdp2.sh recipes/same_tokenizer/fkl.yaml \
    --set train.batch_size=64 train.micro_batch_size=4
```

### Resume from Checkpoint
```bash
CUDA_VISIBLE_DEVICES=0 lmkd train -c recipes/same_tokenizer/fkl.yaml \
    --set train.resume_from=outputs/fkl/checkpoint-500
```

---

## Repository Structure

```
LM-KD/
├── src/lmkd/               # Core framework package
│   ├── core/              # Config, plugin registry, batch, state, and validation
│   ├── models/            # LogitsView, forward runners, adapters, and projectors
│   ├── data/              # Sources, transforms, tokenization, and batch collators
│   ├── losses/            # Functional divergence primitives and 90+ criteria
│   ├── teachers/          # Teacher signal providers (hf_frozen, self-teacher, cache)
│   ├── train/             # Runtime, trainer, engines (local, deepspeed, fsdp2), optimizers
│   ├── rollout/           # Generation engines (HF, vLLM colocate, vLLM server)
│   ├── pipeline/          # Offline DAG multi-stage runner with content-addressed caching
│   ├── eval/              # Standard evaluators and metrics (ROUGE, pass@k, STS)
│   └── compat/            # Isolated third-party API adapters (DeepSpeed, vLLM, TRL, HF)
├── recipes/               # Fully self-contained YAML recipes for all 90+ baselines
├── scripts/               # Production bash runners (single-GPU, multi-GPU, reproduction, eval)
├── examples/              # Standalone runnable scripts, tutorials, and master template
├── docs/                  # In-depth architectural, getting started, and contributor guides
└── tests/                 # Comprehensive test suite (unit, parity, xtok, and e2e)
```

---

## Contributing

We welcome contributions from the community! Please consult the [**Contributor Guide**](docs/CONTRIBUTING.md)
for architectural contracts, coding standards, and Definition of Done checklists.

```bash
# Verify code formatting and linting
ruff check src tests examples recipes
ruff format --check src tests examples

# Run static type checks
mypy

# Run all unit tests
pytest tests/unit/
```

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

