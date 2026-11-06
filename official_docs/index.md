# LM-KD: Unified Language Model Knowledge Distillation Toolkit

Welcome to the official documentation for **LM-KD** (*Language Model Knowledge Distillation*), a production-grade, modular, and hardware-efficient research framework designed for systematic benchmarking, reproducible evaluation, and rapid algorithmic prototyping of knowledge distillation (KD) and alignment techniques for large language models.

---

## 🎯 Target Audiences

To serve diverse research and engineering needs, this documentation is structured around four primary user personas:

| Persona | Primary Goal | Recommended Starting Points |
|---|---|---|
| **Newcomers** | Understand what LM-KD is and run your first distillation experiment in 10 minutes. | [Getting Started](getting-started/quickstart.md), [Concepts & Taxonomy](concepts/taxonomy.md), [Tutorials](tutorials/distilling-llama-to-opt.md) |
| **Baseline Users** | Reproduce published distillation methods across heterogeneous models and benchmarks. | [Methods Catalog](methods/index.md), [Empirical Results](results/index.md), [Pre-built Scenarios](scenarios/index.md) |
| **Method Developers** | Implement novel divergence objectives, span aligners, or multi-stage distillation pipelines. | [Architecture](architecture/core-abstractions.md), [Developer Guides](guides/index.md), [API Reference](api/index.md) |
| **Contributors** | Extend the framework, write plugins, maintain reproducibility, and submit pull requests. | [Contributing Guide](contributing/index.md), [PR Checklist](contributing/pr-checklist.md), [Code Style Standards](contributing/code-style.md) |

---

## 💡 Why LM-KD?

Knowledge distillation has evolved far beyond classical logit matching. Modern techniques encompass **cross-tokenizer vocabulary transfer**, **long chain-of-thought (CoT) reasoning compression**, **on-policy self-distillation with reinforcement learning**, **offline preference distillation**, and **iterative self-play alignment**.

However, empirical research in this domain has long suffered from severe codebase fragmentation:
- **Disparate Codebases:** Researchers routinely rely on bespoke forks of different libraries (TRL, LLaMA-Factory, verl, Megatron), obscuring whether empirical gains stem from algorithmic breakthroughs or hidden engineering artifacts (such as padding masks, learning rate warmups, or sequence accumulation denominators).
- **GPU Memory Explosions:** Distilling models with massive vocabularies ($V \ge 128{,}000$, e.g., LLaMA-3, Qwen-2.5) across long context windows rapidly triggers Out-Of-Memory (OOM) errors during logit materialization.
- **Incompatible Interfaces:** Comparing a token-level loss with a sequence-level or representation-level loss often requires rewriting the entire training loop.

**LM-KD resolves these challenges by introducing a single, hardware-optimized runtime governed by six orthogonal design axes:**

```
                                 ┌──────────────────────────────────────────────┐
                                 │              LM-KD ARCHITECTURE              │
                                 └──────────────────────┬───────────────────────┘
                                                        │
          ┌─────────────────────┬───────────────────────┼───────────────────────┬─────────────────────┐
          ▼                     ▼                       ▼                       ▼                     ▼
┌──────────────────┐  ┌──────────────────┐   ┌───────────────────┐   ┌──────────────────┐  ┌──────────────────┐
│ Context Origin   │  │ Alignment/Spans  │   │ Teacher Signal    │   │ Divergence Loss  │  │ Step & Normalize │
├──────────────────┤  ├──────────────────┤   ├───────────────────┤   ├──────────────────┤  ├──────────────────┤
│• External Corpus │  │• Character Offset│   │• Live Inference   │   │• 90+ Objectives  │  │• num_items norm  │
│• Current-Policy  │  │• Subword Mapping │   │• Top-K Logit Cache│   │• Forward/Rev KL  │  │• Lazy Chunking   │
│• Lagged-Policy   │  │• Trace Segment   │   │• Privileged Hints │   │• Optimal Transp. │  │• ZeRO / FSDP /   │
│• Mixed Schedules │  │• ETA / BPIA / DTW│   │• Embeddings Store │   │• Preference NLL  │  │  vLLM Colocation │
└──────────────────┘  └──────────────────┘   └───────────────────┘   └──────────────────┘  └──────────────────┘
```

---

## 🚀 Key Features & Capabilities

- **Unified Plugin Registry:** Over **90 verified distillation and alignment algorithms** accessible via declarative YAML configurations without writing a single line of training boilerplate.
- **Target-Aligned Lazy Chunking (`LogitsView`):** Slashes peak activation memory by over **60%** by evaluating divergences in streaming sequence chunks without materializing full `[Batch, Length, Vocab]` tensors.
- **Decoupled High-Throughput Rollouts:** Deep integration with **vLLM PagedAttention**, supporting GPU co-location (sleep mode memory sharing) and distributed asynchronous sampling.
- **Centralized Micro-batch Normalization:** Eliminates gradient scale drift across distributed data-parallel workers by normalizing updates by the exact count of supervised tokens (`num_items_in_batch`).
- **Comprehensive Cross-Tokenizer Tooling:** Native support for cross-vocabulary transfer between incompatible architectures (e.g., Qwen2.5 $\rightarrow$ GPT-2/OPT) via exact token alignment (ETA), boundary-preserving incremental alignment (IMPACT/BPIA), and optimal transport.
- **Scientific Provenance:** Automated emission of `reproduction.json` capturing git hashes, input data checksums, hardware environments, and exact hyperparameters.

---

## 📊 Feature & Method Matrix

LM-KD supports seven distinct distillation and alignment paradigms across state-of-the-art open-weights architectures, aligned with the [Baseline Methods & Paper Registry](scenarios/index.md#baseline-methods--paper-registry):

| Distillation Family | Core Mechanism | Supported Teacher $\rightarrow$ Student Setups | Supported Hardware Backends | Status |
|---|---|---|---|:---:|
| **Same-Tokenizer Logit KD** | FKL, RKL, JSD, TVD, SFKL, SRKL, CSD, DistiLLM, GKD, FDD, AMiD, MTA | Qwen3-4B $\rightarrow$ 0.6B; LLaMA-3-8B $\rightarrow$ 1B; OPT-6.7B $\rightarrow$ 1.3B | DDP, ZeRO-2/3, FSDP2, vLLM Colocate | ✅ Reproduced |
| **Cross-Tokenizer KD** | DSKD, DSKDv2, DSKD-CMA, DSKDv2-ETA, ULD, MinED, MultiLevelOT, E-SCOT, ALM, SRA, MCW-KD, ResidualKD, DWA-KD, MTA | Qwen2.5-1.5B/7B $\rightarrow$ GPT-2 (120M–1.5B), OPT (350M–2.7B), TinyLLaMA | DDP, ZeRO-2, 2-Stage Projector DAG | ✅ Reproduced |
| **Long-CoT Reasoning** | SSFT (Segment Attribution), P-ALIGN (Adaptive Prefix Search), SGL-Spectral (SVD Consensus Subspace) | DeepSeek-R1 (671B/70B) $\rightarrow$ Qwen2.5 (1.5B/7B), Qwen3-8B | ZeRO-3, FSDP2, FlashAttention-2, vLLM | ✅ Reproduced |
| **On-Policy / Self-Distill** | OPSD (Gen-JSD Privileged Rationalization), SDPO (Self-Distillation Policy Optimization), RLSD (Self-Distilled RLVR) | Qwen3 (1.7B, 4B, 8B), Olmo3-7B-Think, Qwen3-VL-8B | Ray + verl, PyTorch Accelerate + vLLM | ✅ Reproduced |
| **Preference Distillation** | VPD (PAD), PPD, DCKD, TVKD, ADPA, CTPD, DP3O, PBSD | LLaMA-3.1-8B $\rightarrow$ LLaMA-3.2-1B; Qwen3-8B $\rightarrow$ 1.7B $\rightarrow$ 0.6B | Offline Top-50 Cache, Single-GPU LoRA | ✅ Reproduced |
| **Sentence Embedding KD** | TALAS (TALAS-M, TALAS-ASAM), DistillCSE, EMO, Jasper, MoL | BGE-M3, Qwen3-Embedding (0.6B/4B) $\rightarrow$ MiniLMv2 (H384/H768), BERT-Base | Hybrid Muon + AdamW, ASAM, CPU/GPU | ✅ Reproduced |
| **Self-Play Fine-Tuning** | SPIN, T-SPIN (Triplet anchor), SPACE (NCE), SPPO (Zero-sum winrate), S-SPPO, RSPO | Qwen2.5-1.5B, Gemma-2-2B, Qwen3-4B (Iterative self-play loops 0 $\rightarrow$ 3) | Multi-iteration driver, vLLM + DDP | ✅ Reproduced |

---

## 🗺️ Documentation Sitemap

```
official_docs/
├── index.md                         # Master Overview & Welcome
├── getting-started/                 # Installation, Quickstart, Project Structure, FAQ
├── concepts/                        # Formal Taxonomy, Theoretical Divergences, 6-Axis Decomposition
├── architecture/                    # Abstractions, Registry System, Config Schema, Data Flow, Distributed
├── methods/                         # Mathematical Formulations & Pseudocode for all 7 Baseline Families
├── guides/                          # Step-by-step How-to guides for adding baselines, losses, models
├── data/                            # Schemas, Preprocessing pipelines, Dataset catalog, Prompts
├── evaluation/                      # Metrics catalog, Benchmarks (MTEB, LM-Eval, Math), Graders
├── scenarios/                       # Unified Scenarios & Empirical Results across 7 Core Experiments
│   ├── 1-sentence-embedding.md      # Scenario 1 & Table 1: Sentence Embedding Distillation (MTEB 9 tasks)
│   ├── 2-same-tokenizer.md          # Same-Tokenizer Instruction KD (3 pairs)
│   ├── 3-cross-tokenizer.md         # Scenario 3 & Table 3: Cross-Tokenizer KD (5 pairs)
│   ├── 4-long-cot.md                # Long-CoT Reasoning Distillation (DeepSeek-R1)
│   ├── 5-opsd.md                    # On-Policy Mathematical Self-Distillation
│   ├── 6-preference-distillation.md # RM-Free Preference Distillation (3 setups)
│   └── 7-self-play-fine-tuning.md   # Adversarial Self-Play Fine-Tuning
├── results/                         # Master Consolidated Leaderboards
│   └── leaderboards.md              # All 7 unedited benchmark tables compiled in one reference
├── api/                             # Python API reference for Core, Models, Losses, Teachers, Trainers
├── tutorials/                       # Interactive end-to-end notebooks & research workflows
└── contributing/                    # Contribution standards, PR workflows, and code hygiene
```

