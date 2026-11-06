# Installation & Environment Setup

This guide details the system prerequisites, dependencies, and environment setup instructions for installing and verifying **LM-KD**.

---

## 1. System Requirements

| Component | Minimum Specification | Recommended Specification |
|---|---|---|
| **Operating System** | Linux (Ubuntu 20.04/22.04 LTS) | Ubuntu 22.04 LTS / RHEL 9 |
| **Python** | 3.10 or 3.11 | Python 3.11 |
| **CUDA** | 12.1 or 12.4 | CUDA 12.4 with CuDNN 9 |
| **GPU Hardware** | 1× NVIDIA GPU with $\ge 24\text{ GB}$ VRAM (RTX 3090 / A10G / L4) | 1–8× NVIDIA A100 (80GB) / H100 |
| **Storage** | 50 GB free disk space | High-speed NVMe SSD ($\ge 500\text{ GB}$) |

---

## 2. Recommended Setup: UV / Virtualenv

We strongly recommend using [`uv`](https://github.com/astral-sh/uv) or standard Python `venv` for fast, reproducible dependency resolution.

### Option A: Fast Setup with `uv` (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/LM-KD.git
cd LM-KD/lm-kd

# 2. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment and sync dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all]"
```

### Option B: Conda / Pip Setup

```bash
# 1. Create and activate conda environment
conda create -n lmkd python=3.11 -y
conda activate lmkd

# 2. Install PyTorch with CUDA 12.1 or 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install LM-KD with all baseline dependencies
cd lm-kd
pip install -e ".[all]"
```

---

## 3. Acceleration & Distributed Backends

Depending on the distillation family you plan to run, install the optional acceleration engines:

### A. FlashAttention-2 (Required for context length $\ge 2048$)
```bash
pip install flash-attn --no-build-isolation
```

### B. vLLM (Required for On-Policy Rollouts & High-Throughput Eval)
```bash
# Standard vLLM installation
pip install vllm>=0.6.3
```

### C. DeepSpeed (Required for ZeRO-2 / ZeRO-3 Memory Optimization)
```bash
pip install deepspeed
```

---

## 4. Verifying the Installation

Run the unified sanity check suite to verify that CUDA devices, tensor operations, registries, and configuration schemas are properly initialized:

```bash
# 1. Run unit test suite
pytest tests/unit -v

# 2. Run fast offline smoke test suite (all 6 baseline domains, CPU, ~20s)
bash scripts/run_smoke_tests.sh
```

Expected output:
```text
============================= test session starts ==============================
collected 24 items
tests/test_sanity.py::test_registries_loaded PASSED                       [  4%]
tests/test_sanity.py::test_logits_view_streaming PASSED                   [  8%]
tests/test_sanity.py::test_exact_token_alignment PASSED                   [ 12%]
...
============================== 24 passed in 3.42s ==============================
[OK] Checked 48 baseline configuration schemas successfully. 0 errors detected.
```

