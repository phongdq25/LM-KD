# Contributing to LM-KD

Thank you for your interest in contributing to the LM-KD open-source toolkit! Whether you are implementing a newly published distillation paper, improving distributed engine efficiency, or refining documentation, your contributions help advance the open science of efficient language model distillation.

---

## Areas Where We Welcome Contributions

1. **New Baselines:** Implementing published knowledge distillation, alignment, or reasoning distillation papers.
2. **New Distributed Backends:** Adding support for Megatron-LM tensor/pipeline parallelism, Apple MLX, or specialized hardware (TPUs, AMD ROCm).
3. **Optimized Kernels:** Custom Triton or CUDA kernels for fused chunked KL divergences or fast Sinkhorn optimal transport.
4. **Benchmarks & Data Loaders:** Ingesting new reasoning or conversational benchmarks into `lmkd.data`.

---

## Contributor Guides

- **[Code Style & Standards](code-style.md):** Formatting with Ruff, type hints, docstring rules, and the target-aligned convention.
- **[Pull Request Checklist](pr-checklist.md):** Step-by-step instructions for submitting clean, review-ready PRs.

