# Unified Scenarios & Empirical Results

To eliminate fragmentation between experimental setups and benchmark tables, LM-KD unifies **Deployment Scenarios** and **Empirical Benchmark Results** into a single cohesive reference.

Every scenario in this section defines a **universal, standardized experiment protocol shared across all competing baselines**—including models, dataset splits, evaluation metrics, and general training/test workflows—accompanied directly by its **authoritative results table**.

---

## Master Directory of Scenarios & Benchmark Results

| Scenario & Domain | Primary Focus | Key Models Evaluated | Datasets & Benchmarks | Detailed Document |
| :--- | :--- | :--- | :--- | :--- |
| **1. Sentence Embedding Distillation** | Compressing bi-encoders for dense retrieval and search | Qwen3-Embedding (0.6B, 4B), BGE-M3 $\to$ MiniLMv2, BERT-Base | AllNLI, Wikipedia $\to$ 9 MTEB tasks (Classification, Pair, STS) | [**Scenario 1 Doc & Results**](1-sentence-embedding.md) |
| **2. Same-Tokenizer Instruction Distillation** | General instruction-following with identical vocabulary | GPT-2 (1.5B $\to$ 120M), Qwen1.5 (1.8B $\to$ 0.5B), OPT (6.7B $\to$ 1.3B) | Dolly-15k $\to$ Dolly, SelfInst, Vicuna, Super-NaturalInstructions | [**Scenario 2 Doc & Results**](2-same-tokenizer.md) |
| **3. Cross-Tokenizer Representation Distillation** | Knowledge transfer across disparate vocabularies ($V_S \neq V_T$) | Qwen1.5, Qwen2.5, Mistral $\to$ GPT-2, OPT, TinyLLaMA (5 pairs) | Dolly-15k $\to$ Dolly, SelfInst, Vicuna, Super-NaturalInstructions | [**Scenario 3 Doc & Results**](3-cross-tokenizer.md) |
| **4. Long-CoT Reasoning Chain Distillation** | Transferring deep mathematical reasoning from DeepSeek-R1 | DeepSeek-R1 $\to$ R1-Distill-1.5B, Qwen2.5-Math-7B, Qwen3-8B | s1K-1.1, LIMO $\to$ AIME 2024, AIME 2025, AMC 12, MATH-500 | [**Scenario 4 Doc & Results**](4-long-cot.md) |
| **5. On-Policy Self-Distillation for Reasoning** | Autonomous mathematical bootstrapping without external teacher | Qwen3-8B, Olmo3-7B-Think, Qwen3-4B | GSM8K, MATH, Numina $\to$ AIME 2025, AIME 2026, HMMT 2025 | [**Scenario 5 Doc & Results**](5-opsd.md) |
| **6. Reward-Model-Free Preference Distillation** | Direct preference alignment without training external reward models | Llama-3.1-8B, Qwen3 (8B, 1.7B) $\to$ Llama-3.2-1B, Qwen3 (1.7B, 0.6B) | UltraFeedback Binarized $\to$ HellaSwag, ARC, TruthfulQA, Winogrande, GSM8K | [**Scenario 6 Doc & Results**](6-preference-distillation.md) |
| **7. Self-Play Fine-Tuning** | Adversarial iterative game against earlier student iterations | Qwen2.5-1.5B, Gemma-2-2B, Qwen3-4B | UltraChat-200k, SPIN_iter0 $\to$ GSM8K, MATH, MMLU, MMLU-Pro, HellaSwag, IFEval | [**Scenario 7 Doc & Results**](7-self-play-fine-tuning.md) |

---

## Baseline Methods & Paper Registry

All 49+ baselines implemented in LM-KD are cataloged below by experimental scenario, indexed with their official publication record, venue, and primary source paper links.

### Scenario 1: Sentence Embedding Baselines

| Baseline | Paper Title & Authors | Venue | Primary Source Link | Core Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **TALAS** | *TALAS: Teacher-Anchored Layer Alignment with Adaptive Sharpness-Aware Minimization for Embedding Distillation* — Dao et al. | ACL 2026 | [ACL Anthology](https://aclanthology.org/2026.acl-long.1509/) | Dual-anchor layer alignment with ASAM to flatten loss landscapes. |
| **DistillCSE** | *DistillCSE: Distilled Contrastive Learning for Sentence Embeddings* — Xu et al. | Findings EMNLP 2023 | [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.547/) | Distills contrastive similarity structures using soft teacher ranking distributions. |
| **EMO** | *EMO: Embedding Model Distillation via Intra-Model Relation and Optimal Transport Alignments* — Truong et al. | EMNLP 2025 | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.385/) | Combines MinED mapping, CKA intra-model attention relation alignment, and optimal transport. |
| **Jasper** | *Jasper and Stella: Distillation of SOTA Embedding Models* — Zhang et al. | arXiv 2024 | [arXiv:2412.19048](https://arxiv.org/abs/2412.19048) | Multi-stage contrastive distillation framework with hard negative mining. |
| **MoL** | *MoL: Mixture of Layers in Cross-Tokenizer Embedding Model Distillation* — Vu et al. | Knowledge-Based Systems 2026 | [ScienceDirect](https://doi.org/10.1016/j.knosys.2026.116001) | Dynamic routing across teacher layers with cross-tokenizer projection heads. |

### Scenario 2: Same-Tokenizer Instruction Distillation Baselines

| Baseline | Paper Title & Authors | Venue | Primary Source Link | Core Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **FKL** | *Distilling the Knowledge in a Neural Network* — Hinton et al. | NeurIPS DL Workshop 2014 | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) | Forward-KL divergence matching student probabilities to softened teacher distributions. |
| **RKL** | *MiniLLM: Knowledge Distillation of Large Language Models* — Gu et al. | ICLR 2024 | [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html) | Reverse-KL divergence with policy gradient optimization to prevent hallucination. |
| **JSD / TVD** | *f-Divergence Minimization for Sequence-Level Knowledge Distillation* — Wen et al. | ACL 2023 | [ACL Anthology](https://aclanthology.org/2023.acl-long.605/) | Sequence-level KD optimizing symmetric Jensen-Shannon divergence or Total Variation distance. |
| **SFKL / SRKL** | *DistiLLM: Towards Streamlined Distillation for Large Language Models* — Ko et al. | ICML 2024 | [PMLR Proceedings](https://proceedings.mlr.press/v235/ko24c.html) | Skew forward/reverse KL divergences on adaptive off-policy student generations. |
| **CSD** | *Distillation of Large Language Models via Concrete Score Matching* — Kim et al. | ICLR 2026 | [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1bfc9f74afa91b9b8add5a97a97001a1-Abstract-Conference.html) | Concrete score distillation matching relative logit differences under logit-shift invariance. |
| **GKD** | *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* — Agarwal et al. | ICLR 2024 | [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html) | On-policy distillation training on student-generated sequences scored by teacher probabilities. |
| **FDD** | *Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation* — Gong et al. | ACL 2025 | [ACL Anthology](https://aclanthology.org/2025.acl-long.1125/) | Feature dynamics distillation aligning intermediate hidden-state velocity vectors between layers. |
| **AMiD** | *AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution* — Shin et al. | ICLR 2026 | [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/54a468d19bc5727800096d9c7c51a423-Abstract-Conference.html) | Interpolates an α-mixture assistant policy between student and teacher to bridge capacity gaps. |
| **MTA** | *MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation* — Chi et al. | ACL 2026 | [ACL Anthology](https://aclanthology.org/2026.acl-long.1507/) | Multi-granular alignment matching hidden-state trajectories across token, span, and sequence levels. |

### Scenario 3: Cross-Tokenizer Representation Distillation Baselines

| Baseline | Paper Title & Authors | Venue | Primary Source Link | Core Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **DSKD / DSKD-CMA** | *Dual-Space Knowledge Distillation for Large Language Models* — Zhang et al. | EMNLP 2024 | [ACL Anthology](https://aclanthology.org/2024.emnlp-main.1010/) | Projects representations into shared semantic spaces with cross-model attention (CMA). |
| **DSKDv2 / ETA** | *A Dual-Space Framework for General Knowledge Distillation of Large Language Models* — Zhang et al. | arXiv 2025 | [arXiv:2504.11426](https://arxiv.org/abs/2504.11426) | Extends DSKD with Exact Token Alignment (ETA) via character-level byte boundaries. |
| **ULD** | *Towards Cross-Tokenizer Distillation: The Universal Logit Distillation Loss for LLMs* — Boizard et al. | TMLR 2025 | [OpenReview / TMLR](https://openreview.net/forum?id=buBxXiG09A) | Maps logits across mismatched vocabularies via string-prefix overlap and soft probability transfer. |
| **MinED** | *Knowledge Fusion of Large Language Models* — Wan et al. | ICLR 2024 | [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4fd5cfd2e31bebbccfa5ffa354c04bdc-Abstract-Conference.html) | Dynamic programming minimum-edit-distance alignment between distinct tokenizer token sequences. |
| **MultiLevelOT** | *Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models* — Cui et al. | AAAI 2025 | [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/34543) | Multi-level optimal transport aligning token-level and sequence-level semantic transport plans. |
| **E-SCOT** | *Entropy-aware Span-Constrained Optimal Transport for Robust Cross-Tokenizer Knowledge Distillation* — Liu et al. | ICML 2026 | [PMLR Proceedings](https://proceedings.mlr.press/v306/liu26bw.html) | Span-constrained optimal transport regularized by token prediction entropy to filter noise. |
| **ALM** | *Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching* — Minixhofer et al. | NeurIPS 2025 | [NeurIPS Proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/720f9f5dc751eb56952ae4fee2398f73-Abstract-Conference.html) | Minimizes sequence-level divergence bounds between disparate token vocabularies. |
| **SRA** | *SRA: Span Representation Alignment for Large Language Model Distillation* — Dao et al. | ACL 2026 | [ACL Anthology](https://aclanthology.org/2026.acl-long.1522/) | Aligns pooled representations across contiguous semantic spans across tokenizer boundaries. |
| **MCW-KD** | *MCW-KD: Multi-Cost Wasserstein Knowledge Distillation for Large Language Models* — Vuong et al. | AAAI 2026 | [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40619) | Multi-cost Wasserstein distillation fusing cosine similarity and positional distance transport costs. |
| **ResidualKD** | *Knowledge Distillation for Large Language Models through Residual Learning* — On et al. | ICLR 2026 | [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/608fe7e32f7b773545cc1d656a0fdc98-Abstract-Conference.html) | Two-stage projector learning residual differences between student and teacher representations. |
| **DWA-KD** | *DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation* — Vu et al. | Findings EACL 2026 | [ACL Anthology](https://aclanthology.org/2026.findings-eacl.181/) | Dynamic time warping (DTW) with dual-space weighting across asynchronous token sequences. |

### Scenario 4: Long-CoT Reasoning Chain Baselines

| Baseline | Paper Title & Authors | Venue | Primary Source Link | Core Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **SSFT** | *Segment-Level Attribution for Selective Learning of Long Reasoning Traces* — Wang, Liu & Ren | ICLR 2026 | [OpenReview](https://openreview.net/pdf/7ba314ffe4e9d5c90878656e1656c7d6a34d2a4e.pdf) | Segment-level attribution identifying key reasoning transitions and filtering redundant steps. |
| **P-ALIGN** | *Long-Chain Reasoning Distillation via Adaptive Prefix Alignment* — Liu et al. | ACL 2026 | [ACL Anthology](https://aclanthology.org/2026.acl-long.822/) | Dynamic prefix matching and adaptive thresholding to distill long reasoning trajectories. |
| **SGL-Spectral** | *Uncovering the Gradient Geometry of Long CoT: A Spectral-guided Approach to Reasoning Distillation* — Fan et al. | ICML 2026 | [PMLR Proceedings](https://proceedings.mlr.press/v306/fan26e.html) | Spectral analysis of loss-subspace eigenvalues to prioritize critical deduction tokens. |

### Scenario 5: On-Policy Self-Distillation Baselines

| Baseline | Paper Title & Authors | Venue | Primary Source Link | Core Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **OPSD** | *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models* — Zhao et al. | ICML 2026 | [PMLR Proceedings](https://proceedings.mlr.press/v306/zhao26be.html) | On-policy self-distillation using privileged hints/contexts to construct target distributions. |
| **SDPO** | *Reinforcement Learning via Self-Distillation* — Hübotter et al. | ICML 2026 | [PMLR Proceedings](https://proceedings.mlr.press/v306/hubotter26a.html) | Self-distillation policy optimization using rich textual feedback as dense learning signals. |
| **RLSD** | *Self-Distilled RLVR* — Yang et al. | arXiv 2026 | [arXiv:2604.03128](https://arxiv.org/abs/2604.03128) | Combines verifiable-reward reinforcement learning with token-level self-distillation. |

### Scenario 6: Reward-Model-Free Preference Distillation Baselines

| Baseline | Paper Title & Authors | Venue | Primary Source Link | Core Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **PAD (VPD / PPD)** | *Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models* — Gu et al. | Findings ACL 2025 | [ACL Anthology](https://aclanthology.org/2025.findings-acl.822/) | Preference-aligned distillation via Vanilla (VPD) and Probabilistic (PPD) loss objectives. |
| **DCKD / ADPA** | *Advantage-Guided Distillation for Preference Alignment in Small Language Models* — Gao et al. | ICLR 2025 | [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html) | Dual-constrained distillation (DCKD) and advantage-guided policy alignment (ADPA). |
| **TVKD** | *Preference Distillation via Value-based Reinforcement Learning* — Kwon et al. | NeurIPS 2025 | [NeurIPS Proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/98e967164ae2f6811b975d686dece3eb-Abstract-Conference.html) | Uses teacher token-level state-value estimates as dense preference targets. |
| **CTPD** | *CTPD: Cross Tokenizer Preference Distillation* — Nguyen et al. | AAAI 2026 | [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/41114) | Extends direct preference distillation across mismatched student and teacher tokenizers. |
| **DP3O** | *Towards Bridging the Gap Between Offline and Iterative Alignment via Preference Distillation* — Zhang et al. | TMLR 2026 | [arXiv:2609.06893](https://arxiv.org/abs/2609.06893) | Distilled preference probability policy optimization bridging offline pairs and online rollouts. |
| **PBSD** | *Preference-Based Self-Distillation: Beyond KL Matching via Reward Regularization* — Yu et al. | arXiv 2026 | [arXiv:2605.05040](https://arxiv.org/abs/2605.05040) | Preference-based self-distillation with implicit reward regularization without external teachers. |

### Scenario 7: Self-Play Fine-Tuning Baselines

| Baseline | Paper Title & Authors | Venue | Primary Source Link | Core Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **SPIN** | *Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models* — Chen et al. | ICML 2024 | [PMLR Proceedings](https://proceedings.mlr.press/v235/chen24j.html) | Iterative self-play where the model discriminates its own previous rollouts from reference data. |
| **T-SPIN** | *Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs* — Wang et al. | NeurIPS 2025 | [NeurIPS Proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/3a797b10ff20562b1ecee0d4e914c1c7-Abstract-Conference.html) | Triplet contrastive self-play with historical checkpoints and entropy regularization. |
| **SPACE** | *SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models* — Wang et al. | NeurIPS 2025 | [NeurIPS Proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/5692c7dbc4abcaa50f9ce609819212e5-Abstract-Conference.html) | Noise contrastive estimation stabilizing iterative self-play training dynamics. |
| **SPPO** | *Self-Play Preference Optimization for Language Model Alignment* — Wu et al. | ICLR 2025 | [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e48fa1c4f08fd1ae35d5df8352c3106d-Abstract-Conference.html) | Two-player constant-sum game reaching the game-theoretic Nash equilibrium. |
| **S-SPPO** | *S-SPPO: Semantic-Calibrated Self-Play Preference Optimization* — Chen et al. | ICML 2026 | [PMLR Proceedings](https://proceedings.mlr.press/v306/chen26hc.html) | Semantic gating and representation calibration to prevent reward hacking. |
| **RSPO** | *RSPO: Regularized Self-Play Alignment of Large Language Models* — Tang et al. | ICML 2026 | [PMLR Proceedings](https://proceedings.mlr.press/v306/tang26ab.html) | Regularized self-play optimization with bounded policy updates and reference anchor preservation. |

---

## Key Principles of the Unified Scenarios

1. **Controlled Protocol:** All competing baselines within each scenario share identical training data, prompt formatting, random seeds, and evaluation harnesses.
2. **Exact Results:** Results tables contain complete, unrounded benchmark measurements.
3. **No Hidden Hyperparameters:** Hardware setups and engineering runtime arguments are separated cleanly into runnable YAML recipes under [`recipes/`](../../recipes/).
