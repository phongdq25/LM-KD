# LM-KD Execution Scripts

This directory contains standardized, executable shell scripts organized into subdirectories mirroring `recipes/`
for every supported baseline, as well as general-purpose distributed training and utility scripts.

---

## Directory Organization

```
scripts/
├── same_tokenizer/              # Scripts for same-tokenizer logit KD (FKL, RKL, DistiLLM, GKD, CSD, TVD, JSD, ...)
├── cross_tokenizer/             # Scripts for cross-tokenizer KD (DSKD, DSKDv2, ULD, MinED, MLOT, E-SCOT, IMPACT, ...)
│   └── ablation/                # Scripts for DSKDv2 + IMPACT ablation studies
├── long_cot/                    # Scripts for Long-CoT reasoning (SGL-Spectral, IWC, SSFT, P-ALIGN, SARW, ...)
├── self_distill/                # Scripts for on-policy self-distillation & RL (OPSD, SDPO, RLSD, GRPO, ...)
├── preference/                  # Scripts for preference distillation (DPO, SimPO, IPO, KTO, TDPO, β-DPO, ...)
├── embedding/                   # Scripts for sentence embedding KD (TALAS-M, TALAS-ASAM)
│
├── train_single_gpu.sh          # General single-GPU / CPU launcher with any recipe
├── train_multigpu_deepspeed.sh  # Multi-GPU launcher with DeepSpeed ZeRO-2/3
├── train_multigpu_fsdp2.sh      # Multi-GPU launcher with PyTorch Native FSDP2
├── run_baseline.sh              # Universal reproduction launcher
├── evaluate.sh                  # Standalone checkpoint evaluator
├── run_smoke_tests.sh           # Fast offline smoke tests across all 6 categories
├── run_residualkd_full.sh       # 2-Phase pipeline runner for ResidualKD
└── reproduce.sh                 # Shorthand to lmkd.reproduce CLI
```

---

## Running Specific Baselines

Every baseline recipe under `recipes/` has a direct corresponding script under `scripts/`.
All scripts accept extra parameters and `--set` overrides passed through to `lmkd train`.

### 1. Same-Tokenizer Logit Distillation
```bash
# Run Forward KL distillation
bash scripts/same_tokenizer/run_fkl.sh

# Run DistiLLM (Adaptive On-Policy Generation with Skewed Reverse-KL)
bash scripts/same_tokenizer/run_distillm.sh

# Run Conditional Student Divergence (CSD)
bash scripts/same_tokenizer/run_csd.sh

# Run with overrides (e.g., limit steps and adjust learning rate)
bash scripts/same_tokenizer/run_fkl.sh --set train.max_steps=200 optim.lr=1e-5
```

### 2. Cross-Tokenizer Representation Transfer
```bash
# Run Universal Logit Distillation (ULD)
bash scripts/cross_tokenizer/run_uld.sh

# Run DSKDv2 with Exact Token Alignment (ETA)
bash scripts/cross_tokenizer/run_dskdv2_eta.sh

# Run E-SCOT (Elastic Span Constrained Optimal Transport)
bash scripts/cross_tokenizer/run_escot.sh

# Run Two-Phase ResidualKD pipeline (Phase 1 Projector -> Phase 2 Distillation)
bash scripts/cross_tokenizer/run_residualkd_full.sh
```

### 3. Long-CoT Reasoning Compression
```bash
# Run Spectral Guided Learning with Stable IWC
bash scripts/long_cot/run_sgl_spectral_pipeline.sh

# Run Segment-Selective SFT (SSFT) with Integrated Gradients
bash scripts/long_cot/run_ssft.sh

# Run P-ALIGN with Student Prefix Search
bash scripts/long_cot/run_palign.sh
```

### 4. On-Policy & Self-Distillation
```bash
# Run On-Policy Self-Distillation (OPSD)
bash scripts/self_distill/run_opsd.sh

# Run Self-Distillation with Preference Optimization (SDPO)
bash scripts/self_distill/run_sdpo.sh

# Run RL Self-Distillation (RLSD) with Group Rollouts
bash scripts/self_distill/run_rlsd.sh

# Run GRPO
bash scripts/self_distill/run_grpo.sh
```

### 5. Preference Distillation & Alignment
```bash
# Run Direct Preference Optimization (DPO)
bash scripts/preference/run_dpo.sh

# Run Simple Preference Optimization (SimPO)
bash scripts/preference/run_simpo.sh

# Run Token-Dependent DPO (TDPO)
bash scripts/preference/run_tdpo.sh

# Run Kahneman-Tversky Optimization (KTO via TRL backend)
bash scripts/preference/run_kto.sh
```

### 6. Embedding Distillation
```bash
# Run TALAS-M with Muon optimizer and CLS-attention
bash scripts/embedding/run_talas_m.sh

# Run TALAS-ASAM (Adaptive Sharpness-Aware Minimization)
bash scripts/embedding/run_talas_asam.sh
```

---

## General Utility Scripts

### Fast Offline Smoke Test
Run synthetic tiny models across all 6 baseline categories on CPU in ~20 seconds without downloading external weights:
```bash
bash scripts/run_smoke_tests.sh
```

### Distributed Multi-GPU Training
Run on 4 GPUs with DeepSpeed ZeRO-2:
```bash
NUM_GPUS=4 bash scripts/train_multigpu_deepspeed.sh recipes/same_tokenizer/fkl_zero2.yaml
```

Run on 4 GPUs with native PyTorch FSDP2:
```bash
NUM_GPUS=4 bash scripts/train_multigpu_fsdp2.sh recipes/same_tokenizer/fkl.yaml
```

### Checkpoint Evaluation
Evaluate a saved checkpoint against the metrics declared in its recipe:
```bash
bash scripts/evaluate.sh recipes/same_tokenizer/fkl.yaml --ckpt outputs/fkl/checkpoint-final
```

Run on 4 GPUs with native PyTorch FSDP2:
```bash
NUM_GPUS=4 bash scripts/train_multigpu_fsdp2.sh recipes/same_tokenizer/fkl.yaml
```

### Checkpoint Evaluation
Evaluate a saved checkpoint against the metrics declared in its recipe:
```bash
bash scripts/evaluate.sh recipes/same_tokenizer/fkl.yaml --ckpt outputs/fkl/checkpoint-final
```

