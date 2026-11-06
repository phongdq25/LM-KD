# Sentence & Dense Embedding Distillation

Embedding distillation focuses on compressing massive encoder representation models (such as [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) with $D = 1024$ or [`Qwen/Qwen3-Embedding-0.6B/4B`](https://huggingface.co/Qwen)) into compact encoders (such as [`nreimers/MiniLMv2-L6-H384`](https://huggingface.co/nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base) with 22M parameters and $D = 384$) while preserving retrieval, semantic textual similarity (STS), and transfer classification performance across the Massive Text Embedding Benchmark (MTEB).

LM-KD supports all **5 core sentence embedding baseline methods**.

---

## Baseline Summary & Paper Registry

| Baseline | Official Paper Title | Authors & Venue | Official Record / Paper Link | Key Focus |
| :--- | :--- | :--- | :--- | :--- |
| **TALAS** | *TALAS: Teacher-Anchored Layer Alignment with Adaptive Sharpness-Aware Minimization for Embedding Distillation* | Dao et al. (ACL 2026) | [ACL Anthology](https://aclanthology.org/2026.acl-long.1509/) | Teacher-Anchored Multi-Layer Distillation (TAMD) + ASAM |
| **DistillCSE** | *DistillCSE: Distilled Contrastive Learning for Sentence Embeddings* | Xu et al. (Findings EMNLP 2023) | [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.547/) | Pairwise batch relational cosine topology distillation + SimCSE |
| **EMO** | *EMO: Embedding Model Distillation via Intra-Model Relation and Optimal Transport Alignments* | Truong et al. (EMNLP 2025) | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.385/) | Intra-model relational CKA + importance-scored Optimal Transport |
| **Jasper** | *Jasper and Stella: Distillation of SOTA Embedding Models* | Zhang et al. (arXiv 2024; preprint) | [arXiv:2412.19048](https://arxiv.org/abs/2412.19048) · [code](https://github.com/NLPJCL/RAG-Retrieval) | Decoupled 2-stage projector pretraining & InfoNCE fine-tuning |
| **MoL** | *MoL: Mixture of Layers in Cross-Tokenizer Embedding Model Distillation* | Vu et al. (Knowledge-Based Systems 2026) | [ScienceDirect / journal record](https://www.sciencedirect.com/science/article/pii/S0950705126007276) · [DOI](https://doi.org/10.1016/j.knosys.2026.116001) | Learnable dynamic mixture across intermediate teacher layers |

---

## 1. TALAS (Teacher-Anchored Layer Alignment with ASAM)

- **Official Paper:** *TALAS: Teacher-Anchored Layer Alignment with Adaptive Sharpness-Aware Minimization for Embedding Distillation*
- **Authors:** Quoc Phong Dao, et al.
- **Publication / Venue:** ACL 2026
- **Links:** [ACL Anthology](https://aclanthology.org/2026.acl-long.1509/)

### Idea Summary
Compressing large foundation models into compact sentence encoders encounters both a severe architectural capacity gap and significant computational overhead if intermediate representations are forced to match naively. TALAS resolves this through three core principles:
1. **Teacher-Anchored Multi-Layer Distillation ($\mathcal{L}_{\text{TAMD}}$):** Instead of matching every teacher layer, the teacher's final pooled representation acts as a stationary semantic anchor, selectively projected only into the student's upper layers via learnable projectors.
2. **Layer-Aligned Self-Distillation ($\mathcal{L}_{\text{LASD}}$):** A top-down chain self-distills relational structures between the student's own adjacent intermediate layers, ensuring smooth continuous feature propagation without needing teacher hidden states.
3. **Adaptive Sharpness-Aware Minimization (ASAM):** Guides updates toward flat minima in the loss landscape, preventing overfitting to teacher pointwise variance and maximizing zero-shot MTEB generalization.

### Mathematical Formulation
$$\mathcal{L}_{\text{TALAS}}(\theta) = \mathcal{L}_{\text{TAMD}}(\theta) + \lambda_1 \mathcal{L}_{\text{LASD}}(\theta) + \lambda_2 \mathcal{L}_{\text{SimCSE}}(\theta)$$
where ASAM seeks flat minima via perturbation radius $\rho$:
$$\min_\theta \max_{\|\epsilon\| \le \rho} \mathcal{L}_{\text{TALAS}}(\theta + \epsilon) + \lambda \|\theta\|^2$$

### Recipe References
- **TALAS-M (Muon Optimizer):** [`recipes/embedding/talas_m.yaml`](../../recipes/embedding/talas_m.yaml)
- **TALAS-ASAM (Adaptive SAM):** [`recipes/embedding/talas_asam.yaml`](../../recipes/embedding/talas_asam.yaml)

---

## 2. DistillCSE (Distilled Contrastive Learning for Sentence Embeddings)

- **Official Paper:** *DistillCSE: Distilled Contrastive Learning for Sentence Embeddings*
- **Authors:** Shengjie Xu, et al.
- **Publication / Venue:** Findings EMNLP 2023
- **Links:** [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.547/)

### Idea Summary
DistillCSE transfers the pairwise relational topology of the Teacher embedding space directly to the Student. For a mini-batch of sentences, it calculates the complete $B \times B$ cross-sentence cosine similarity matrix under the Teacher and enforces that the Student reproduces these relational affinity scores via mean squared error, combined with unsupervised SimCSE contrastive training.

### Mathematical Formulation
$$\mathcal{L}_{\text{DistillCSE}}(\theta) = \mathcal{L}_{\text{SimCSE}}(\theta) + \frac{\alpha}{B^2} \sum_{i=1}^B \sum_{j=1}^B \left( \cos(e_{S, i}, e_{S, j}) - \cos(e_{T, i}, e_{T, j}) \right)^2$$

### Recipe Reference
[`recipes/embedding/distillcse.yaml`](../../recipes/embedding/distillcse.yaml)

---

## 3. EMO (Embedding Model Distillation via Intra-Model Relation and OT)

- **Official Paper:** *EMO: Embedding Model Distillation via Intra-Model Relation and Optimal Transport Alignments*
- **Authors:** Minh-Phuc Truong, Hai An Vu, Tu Vu, et al.
- **Publication / Venue:** EMNLP 2025
- **Links:** [ACL Anthology](https://aclanthology.org/2025.emnlp-main.385/)

### Idea Summary
EMO addresses cross-architecture and cross-tokenizer embedding model distillation:
1. **Intra-Model Relational Distillation:** Aligns intra-model token-to-token similarity geometries using Centered Kernel Alignment (CKA), ensuring the internal relational manifold of the student reflects the teacher without requiring identical token boundaries.
2. **Optimal Transport (OT) Alignment:** Employs entropic Optimal Transport with importance-weighted mass assignment to match token embedding distributions across differing dimensions in linear or polynomial time.

### Mathematical Formulation
$$\mathcal{L}_{\text{EMO}}(\theta) = \mathcal{L}_{\text{OT}}(\mu_S, \mu_T) + \mu \cdot \text{CKA}(H_S, H_T) + \mathcal{L}_{\text{SimCSE}}(\theta)$$

### Recipe Reference
[`recipes/embedding/emo.yaml`](../../recipes/embedding/emo.yaml)

---

## 4. Jasper (Jasper and Stella Embedding Model Distillation)

- **Official Paper:** *Jasper and Stella: Distillation of SOTA Embedding Models*
- **Authors:** Zhang et al.
- **Publication / Venue:** arXiv 2024 (Preprint)
- **Links:** [arXiv:2412.19048](https://arxiv.org/abs/2412.19048) · [code](https://github.com/NLPJCL/RAG-Retrieval)

### Idea Summary
The Jasper framework uses a decoupled two-stage distillation pipeline:
- **Stage 1 (Projector Pretraining):** Freezes the student encoder and trains an intermediate Multi-Layer Perceptron (MLP) projector to map student hidden representations into the frozen teacher embedding space ([`recipes/embedding/jasper_stage1.yaml`](../../recipes/embedding/jasper_stage1.yaml)).
- **Stage 2 (Joint End-to-End Distillation):** Unfreezes the student encoder and trains all parameters jointly with cosine distance and contrastive InfoNCE objectives ([`recipes/embedding/jasper_stage2.yaml`](../../recipes/embedding/jasper_stage2.yaml)).

### Mathematical Formulation
$$\text{Stage 1: } \min_{\phi} \frac{1}{B} \sum_{i=1}^B \left( 1 - \cos\left( \phi(\text{Pool}(H_S^{(i)})),\, e_T^{(i)} \right) \right)$$
$$\text{Stage 2: } \min_\theta \mathcal{L}_{\text{InfoNCE}}(e_S(\theta)) + \beta \left( 1 - \cos\left( \phi^*(e_S(\theta)),\, e_T \right) \right)$$

### Recipe Reference
[`recipes/embedding/jasper.yaml`](../../recipes/embedding/jasper.yaml)

---

## 5. MoL (Mixture of Layers in Cross-Tokenizer Embedding Distillation)

- **Official Paper:** *MoL: Mixture of Layers in Cross-Tokenizer Embedding Model Distillation*
- **Authors:** Hai An Vu, Minh-Phuc Truong, Tu Vu, Linh Ngo Van
- **Publication / Venue:** Knowledge-Based Systems 2026
- **Links:** [ScienceDirect / journal record](https://www.sciencedirect.com/science/article/pii/S0950705126007276) · [DOI:10.1016/j.knosys.2026.116001](https://doi.org/10.1016/j.knosys.2026.116001)

### Idea Summary
Different levels of semantic abstraction reside in different intermediate layers of large foundation teachers. Rather than supervising student bi-encoders solely from the teacher's final output layer, MoL learns a dynamic mixture across intermediate teacher layers:

$$\bar{H}_T = \sum_{l=1}^L \alpha_l \cdot H_T^{(l)}, \qquad \sum_{l=1}^L \alpha_l = 1$$

This dynamic Mixture-of-Layers representation provides multi-scale semantic guidance across cross-tokenizer boundaries.

### Mathematical Formulation
$$\mathcal{L}_{\text{MoL}}(\theta) = \left\| \text{Pool}(H_S) - W_{\text{proj}} \left( \sum_{l=1}^L \alpha_l \cdot \text{Pool}(H_T^{(l)}) \right) \right\|_2^2 + \mathcal{L}_{\text{SimCSE}}(\theta)$$

### Recipe Reference
[`recipes/embedding/mol.yaml`](../../recipes/embedding/mol.yaml)
