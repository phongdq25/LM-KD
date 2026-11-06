# Cross-Tokenizer Knowledge Distillation

Cross-tokenizer distillation transfers capabilities between models with completely different tokenizers, subword segmentations, and vocabulary spaces ($|\mathcal{V}_S| \neq |\mathcal{V}_T|$, e.g., transferring from [`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B) with $|\mathcal{V}_T| = 152,064$ to [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) with $|\mathcal{V}_S| = 128,256$, or [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) with $|\mathcal{V}_S| = 50,257$).

Because token positions and token IDs do not correspond 1:1, naive logit cross-entropy is mathematically undefined.

LM-KD supports all **14 core cross-tokenizer baseline methods**.

---

## Baseline Summary & Paper Registry

| Baseline | Official Paper Title | Authors & Venue | Official Record / Paper Link | Key Focus |
| :--- | :--- | :--- | :--- | :--- |
| **DSKD** | *Dual-Space Knowledge Distillation for Large Language Models* | Zhang et al. (EMNLP 2024) | [ACL Anthology](https://aclanthology.org/2024.emnlp-main.1010/) · [arXiv:2406.17328](https://arxiv.org/abs/2406.17328) | Dual projection matrices distilling across both vocabulary spaces |
| **DSKD-CMA** | Cross-Model Attention in *Dual-Space KD for LLMs* | Zhang et al. (EMNLP 2024; component) | [Same EMNLP paper](https://aclanthology.org/2024.emnlp-main.1010/) | Cross-attention alignment mechanism; component of DSKD |
| **DSKDv2** | *A Dual-Space Framework for General Knowledge Distillation of Large Language Models* | Zhang et al. (arXiv 2025; preprint) | [arXiv:2504.11426](https://arxiv.org/abs/2504.11426) · [code](https://github.com/songmzhang/DSKDv2) | Pseudoinverse head initialization + agreement mask |
| **DSKDv2-ETA** | Exact Token Alignment in *A Dual-Space Framework...* | Zhang et al. (arXiv 2025; component) | [Same DSKDv2 paper](https://arxiv.org/abs/2504.11426) | Deterministic 1:1 token boundary alignment module |
| **ULD** | *Towards Cross-Tokenizer Distillation: The Universal Logit Distillation Loss for LLMs* | Boizard et al. (TMLR 2025) | [OpenReview / TMLR paper](https://openreview.net/forum?id=buBxXiG09A) · [arXiv:2402.12030](https://arxiv.org/abs/2402.12030) | Vocabulary-independent sorted logit rank matching |
| **MinED** | *Knowledge Fusion of Large Language Models* | Wan et al. (ICLR 2024) | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4fd5cfd2e31bebbccfa5ffa354c04bdc-Abstract-Conference.html) · [arXiv:2401.10491](https://arxiv.org/abs/2401.10491) | Minimum-edit-distance monotonic token alignment (FuseLLM) |
| **MultiLevelOT** | *Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models* | Cui et al. (AAAI 2025) | [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/34543) · [arXiv:2412.14528](https://arxiv.org/abs/2412.14528) | Multi-level entropic Sinkhorn transport at token and sequence tiers |
| **E-SCOT** | *Entropy-aware Span-Constrained Optimal Transport for Robust Cross-Tokenizer Knowledge Distillation* | Liu et al. (ICML 2026) | [PMLR proceedings](https://proceedings.mlr.press/v306/liu26bw.html) | Linear-time span-constrained OT with Rényi-entropy reweighting |
| **ALM** | *Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching* | Minixhofer et al. (NeurIPS 2025) | [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/720f9f5dc751eb56952ae4fee2398f73-Abstract-Conference.html) · [arXiv:2503.20083](https://arxiv.org/abs/2503.20083) | Approximate Likelihood Matching for pure cross-tokenizer transfer |
| **SRA** | *SRA: Span Representation Alignment for Large Language Model Distillation* | Dao et al. (ACL 2026) | [ACL Anthology](https://aclanthology.org/2026.acl-long.1522/) · [PDF](https://aclanthology.org/2026.acl-long.1522.pdf) | Span Center-of-Mass alignment via Longest Common Subsequence |
| **MCW-KD** | *MCW-KD: Multi-Cost Wasserstein Knowledge Distillation for Large Language Models* | Vuong et al. (AAAI 2026) | [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40619) | Multi-cost Wasserstein metric aligning hidden states & logits |
| **ResidualKD** | *Knowledge Distillation for Large Language Models through Residual Learning* | On et al. (ICLR 2026) | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/608fe7e32f7b773545cc1d656a0fdc98-Abstract-Conference.html) | Two-stage decoupled projector pretraining and residual learning |
| **DWA-KD** | *DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation* | Vu et al. (Findings EACL 2026) | [ACL Anthology](https://aclanthology.org/2026.findings-eacl.181/) · [arXiv:2602.21669](https://arxiv.org/abs/2602.21669) | Dual-space entropy weighting + Soft-DTW sequence alignment |
| **DSKDv2 + MTA** | *MTA: Multi-Granular Trajectory Alignment for LLM Distillation* | Chi et al. (ACL 2026) | [ACL Anthology](https://aclanthology.org/2026.acl-long.1507/) | Cross-tokenizer integration combining DSKDv2 and MTA |

---

## 1. DSKD (Dual-Space Knowledge Distillation)

- **Official Paper:** *Dual-Space Knowledge Distillation for Large Language Models*
- **Authors:** Songming Zhang, et al.
- **Publication / Venue:** EMNLP 2024
- **Links:** [ACL Anthology](https://aclanthology.org/2024.emnlp-main.1010/) · [arXiv:2406.17328](https://arxiv.org/abs/2406.17328)

### Idea Summary
Parent paper for both DSKD and DSKD-CMA. It introduces a dual-space distillation paradigm: trains two linear projection matrices, $W_{T \to S}$ to map Teacher representations into Student vocabulary space, and $W_{S \to T}$ to map Student representations into Teacher vocabulary space. Distillation is performed simultaneously across both discrete spaces, minimizing bi-directional divergence.

### Mathematical Formulation
$$\mathcal{L}_{\text{DSKD}}(\theta) = D_{\text{KL}}\left( W_{T \to S}(P_T) \parallel P_S \right) + D_{\text{KL}}\left( P_T \parallel W_{S \to T}(P_S) \right)$$

### Recipe Reference
[`recipes/cross_tokenizer/dskd.yaml`](../../recipes/cross_tokenizer/dskd.yaml)

---

## 2. DSKDv2 (Dual-Space KD Version 2)

- **Official Paper:** *A Dual-Space Framework for General Knowledge Distillation of Large Language Models*
- **Authors:** Songming Zhang, et al.
- **Publication / Venue:** arXiv 2025 (Preprint)
- **Links:** [arXiv:2504.11426](https://arxiv.org/abs/2504.11426) · [official code](https://github.com/songmzhang/DSKDv2)

### Idea Summary
An extended version of EMNLP 2024 DSKD. It analytically initializes the cross-vocabulary projection matrix using the Moore-Penrose pseudoinverse of the teacher's LM head ($W_S \cdot (W_T)^\dagger$), removing the unstable initial phase of random linear mapping. It incorporates an agreement mask mechanism that selectively suppresses positions lacking valid cross-vocabulary correspondence.

### Mathematical Formulation
$$W_{\text{proj}}^{(0)} = W_{\text{head}}^S \left( W_{\text{head}}^T \right)^\dagger$$
$$\mathcal{L}_{\text{DSKDv2}}(\theta) = \sum_{i=1}^{L_S} M_i \cdot D_{\text{KL}}\left( \text{proj}(P_{T, \pi(i)}) \parallel P_{S, i} \right) + \mathcal{L}_{\text{rep}}$$

### Recipe Reference
[`recipes/cross_tokenizer/dskdv2.yaml`](../../recipes/cross_tokenizer/dskdv2.yaml)

---

## 3. DSKD-CMA (Cross-Model Attention)

- **Official Paper:** *Dual-Space Knowledge Distillation for Large Language Models* (Component)
- **Authors:** Songming Zhang, et al.
- **Publication / Venue:** EMNLP 2024
- **Links:** [ACL Anthology](https://aclanthology.org/2024.emnlp-main.1010/)

### Idea Summary
Cross-Model Attention (CMA) is a component of the DSKD paper (not a standalone publication). It applies multi-head cross-attention between Student queries $Q_S$ and Teacher keys/values $(K_T, V_T)$ across hidden states. This learns soft contextual correlations between token sequences of non-uniform length, enabling flexible contextual information transfer without enforcing strict 1:1 token boundary alignment.

### Mathematical Formulation
$$A = \text{Softmax}\left( \frac{Q_S K_T^\top}{\sqrt{d}} \right)$$
$$\mathcal{L}_{\text{CMA}}(\theta) = \left\| H_S - A H_T W_v \right\|_F^2$$

### Recipe Reference
[`recipes/cross_tokenizer/dskd_cma.yaml`](../../recipes/cross_tokenizer/dskd_cma.yaml)

---

## 4. DSKDv2-ETA (Exact Token Alignment)

- **Official Paper:** *A Dual-Space Framework for General Knowledge Distillation of Large Language Models* (Component)
- **Authors:** Songming Zhang, et al.
- **Publication / Venue:** arXiv 2025
- **Links:** [arXiv:2504.11426](https://arxiv.org/abs/2504.11426) · [official code](https://github.com/songmzhang/DSKDv2)

### Idea Summary
Exact Token Alignment (ETA) is a deterministic sub-method described in the DSKDv2 preprint. A two-pointer character-span scanner identifies exact 1:1 identical token substring boundaries between student and teacher tokenizations. Strong cosine hidden-state and logit supervision are concentrated exclusively at these verified aligned positions, while ambiguous partial-token boundaries are cleanly bypassed.

### Mathematical Formulation
$$\mathcal{L}\_{\text{ETA}}(\theta) = \frac{1}{|\mathcal{A}|} \sum_{(i, j) \in \mathcal{A}} \left( 1 - \cos(h_{S,i}, h_{T,j}) \right)$$

where $\mathcal{A} = \{ (i, j) \mid \text{span}_S(i) \equiv \text{span}_T(j) \}$.

### Recipe Reference
[`recipes/cross_tokenizer/dskdv2_eta.yaml`](../../recipes/cross_tokenizer/dskdv2_eta.yaml)

---

## 5. ULD (Universal Logit Distillation)

- **Official Paper:** *Towards Cross-Tokenizer Distillation: The Universal Logit Distillation Loss for LLMs*
- **Authors:** Alexis Boizard, et al.
- **Publication / Venue:** TMLR 2025 (Transactions on Machine Learning Research)
- **Links:** [OpenReview / TMLR paper](https://openreview.net/forum?id=buBxXiG09A) · [arXiv:2402.12030](https://arxiv.org/abs/2402.12030)

### Idea Summary
A completely vocabulary-independent logit distillation objective. ULD sorts the predictive probability distributions in descending order on both student and teacher sides (sorted rank profiles), and minimizes the $L_1$ distance across the top-$K$ sorted probability components. This forces the student to reproduce the teacher's prediction entropy and confidence distribution without requiring vocabulary mapping.

### Mathematical Formulation
$$\mathcal{L}_{\text{ULD}}(\theta) = \frac{1}{L} \sum_{j=1}^L \sum_{k=1}^K \left| \text{sort}(P_T(\cdot \mid s_j))_k - \text{sort}(P_S(\cdot \mid s_j))_k \right|$$

### Recipe Reference
[`recipes/cross_tokenizer/uld.yaml`](../../recipes/cross_tokenizer/uld.yaml)

---

## 6. MinED (Minimum Edit Distance Distillation / FuseLLM)

- **Official Paper:** *Knowledge Fusion of Large Language Models*
- **Authors:** Fanqi Wan, Xinting Huang, Deng Cai, Xiaojun Quan, Wei Bi, Shuming Shi
- **Publication / Venue:** ICLR 2024
- **Links:** [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4fd5cfd2e31bebbccfa5ffa354c04bdc-Abstract-Conference.html) · [arXiv:2401.10491](https://arxiv.org/abs/2401.10491)

### Idea Summary
Minimum Edit Distance (MinED) token alignment is an algorithm developed in FuseLLM (Wan et al., ICLR 2024). It applies dynamic programming (Levenshtein distance) over character tokens to find an optimal monotonic sequence alignment path $\pi^*$ between student and teacher tokenizations. Forward KL divergence is then computed over the mapped position pairs.

### Mathematical Formulation
$$\pi^* = \arg\min_\pi \sum_{(i, j) \in \pi} \text{EditCost}(t_{S,i}, t_{T,j})$$
$$\mathcal{L}_{\text{MinED}}(\theta) = \sum_{(i, j) \in \pi^*} D_{\text{KL}}\left( \tilde{P}_{T,j} \parallel P_{S,i} \right)$$

### Recipe Reference
[`recipes/cross_tokenizer/mined.yaml`](../../recipes/cross_tokenizer/mined.yaml)

---

## 7. MultiLevelOT (Multi-Level Optimal Transport)

- **Official Paper:** *Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models*
- **Authors:** Ganqu Cui, et al.
- **Publication / Venue:** AAAI 2025
- **Links:** [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/34543) · [arXiv:2412.14528](https://arxiv.org/abs/2412.14528)

### Idea Summary
MultiLevelOT formulates cross-tokenizer knowledge distillation as an entropic Optimal Transport problem across two distinct hierarchical tiers: token-level and sequence-level. Using the entropic Sinkhorn-Knopp algorithm, it finds the minimal transport plan between the student's and teacher's unaligned token feature distributions using cosine distance as the ground metric.

### Mathematical Formulation
$$\mathcal{L}_{\text{MLOT}}(\theta) = \min_{P \in \Pi} \langle P, C \rangle_F - \epsilon H(P)$$

where $C_{ij} = 1 - \cos(h_{S,i}, h_{T,j})$ is the token pairwise cosine distance matrix.

### Recipe Reference
[`recipes/cross_tokenizer/mlot.yaml`](../../recipes/cross_tokenizer/mlot.yaml)

---

## 8. E-SCOT (Entropy-aware Span-Constrained Optimal Transport)

- **Official Paper:** *Entropy-aware Span-Constrained Optimal Transport for Robust Cross-Tokenizer Knowledge Distillation*
- **Authors:** Zheyuan Liu, et al.
- **Publication / Venue:** ICML 2026
- **Links:** [PMLR proceedings](https://proceedings.mlr.press/v306/liu26bw.html)

### Idea Summary
Cross-tokenizer KD in low-overlap regimes suffers from negative transfer. E-SCOT solves this via:
1. **Linear-Time Span Alignment:** Replaces expensive quadratic $O(N^2)$ sequence-level OT with linear-time $O(N)$ span-anchored lexical alignment, bounding optimal transport within character spans.
2. **Rényi Entropy Reweighting:** Dynamically assigns higher transport mass to informative token positions exhibiting large uncertainty gaps between teacher and student, preventing alignment noise on unambiguous tokens.

### Mathematical Formulation
$$\mathcal{L}_{\text{E-SCOT}}(\theta) = \sum_{k} w_k \cdot \text{Sinkhorn}\left( H_S^{(k)}, H_T^{(k)};\, C \right)$$

where $w_k = \exp\left( -H_\alpha(P_T^{(k)}) \right)$ scales by teacher Rényi entropy.

### Recipe Reference
[`recipes/cross_tokenizer/escot.yaml`](../../recipes/cross_tokenizer/escot.yaml)

---

## 9. ALM (Approximate Likelihood Matching)

- **Official Paper:** *Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching*
- **Authors:** Benjamin Minixhofer, Ivan Vulić, Edoardo M. Ponti
- **Publication / Venue:** NeurIPS 2025
- **Links:** [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/720f9f5dc751eb56952ae4fee2398f73-Abstract-Conference.html) · [arXiv:2503.20083](https://arxiv.org/abs/2503.20083) · [code](https://github.com/bminixhofer/tokenkit)

### Idea Summary
ALM performs pure cross-tokenizer distillation without relying on standard next-token prediction cross-entropy as its primary engine. It maps probability mass across disparate token boundaries by matching the likelihood assigned to equivalent text spans via TokenKit chunk matrices. This enables distilling between completely incompatible tokenizers (e.g., standard BPE to byte-level tokenizers).

### Mathematical Formulation
$$\mathcal{L}_{\text{ALM}}(\theta) = \mathcal{L}_{\text{match}}\left( P_S(\text{span}),\, P_T(\text{span}) \right) + \gamma_{\text{GradMag}} \cdot \mathcal{L}_{\text{aux}}$$

### Recipe Reference
[`recipes/cross_tokenizer/alm.yaml`](../../recipes/cross_tokenizer/alm.yaml)

---

## 10. SRA (Span Representation Alignment)

- **Official Paper:** *SRA: Span Representation Alignment for Large Language Model Distillation*
- **Authors:** Quoc Phong Dao, et al.
- **Publication / Venue:** ACL 2026
- **Links:** [ACL Anthology](https://aclanthology.org/2026.acl-long.1522/) · [PDF](https://aclanthology.org/2026.acl-long.1522.pdf)

### Idea Summary
SRA shifts the fundamental unit of alignment from individual subword tokens to robust, tokenizer-agnostic text spans. Borrowing principles from multi-particle dynamical systems, SRA represents each text span by its **Center of Mass (CoM)**—an attention-weighted average over span tokens. It identifies optimal span correspondences using Longest Common Subsequence (LCS) and minimizes span CoM distance across teacher and student representations.

### Mathematical Formulation
$$\text{CoM}(H_{\text{span}}) = \sum_{t \in \text{span}} \alpha_t \cdot h_t, \qquad \alpha_t = \text{Softmax}(\text{Attn}(h_t))$$
$$\mathcal{L}_{\text{SRA}}(\theta) = \left\| \text{CoM}(H_S) - W_p \cdot \text{CoM}(H_T) \right\|_F^2 + \lambda \|A_S - A_T\|_F^2$$

### Recipe Reference
[`recipes/cross_tokenizer/sra.yaml`](../../recipes/cross_tokenizer/sra.yaml)

---

## 11. MCW-KD (Multi-Cost Wasserstein KD)

- **Official Paper:** *MCW-KD: Multi-Cost Wasserstein Knowledge Distillation for Large Language Models*
- **Authors:** Quoc-Huy Vuong, et al.
- **Publication / Venue:** AAAI 2026
- **Links:** [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40619)

### Idea Summary
MCW-KD addresses the limitation of conventional Optimal Transport methods that rely on a single cost metric. When student and teacher models possess different architectures and tokenizers, a single metric cannot simultaneously capture semantic hidden state discrepancy and output distribution divergence. MCW-KD formulates a unified Multi-Cost Wasserstein distance combining representation cosine distances and logit divergences into a composite transport plan.

### Mathematical Formulation
$$C_{ij}^{\text{total}} = \alpha \cdot \left( 1 - \cos(h_{S,i}, h_{T,j}) \right) + (1 - \alpha) \cdot D_{\text{JS}}\left( \text{proj}(P_{S,i}), P_{T,j} \right)$$
$$\mathcal{L}_{\text{MCW}}(\theta) = \text{Wasserstein}(H_S, H_T; C^{\text{total}})$$

### Recipe Reference
[`recipes/cross_tokenizer/mcw_kd.yaml`](../../recipes/cross_tokenizer/mcw_kd.yaml)

---

## 12. ResidualKD (Residual Learning KD)

- **Official Paper:** *Knowledge Distillation for Large Language Models through Residual Learning*
- **Authors:** Gyeongman On, et al.
- **Publication / Venue:** ICLR 2026
- **Links:** [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/608fe7e32f7b773545cc1d656a0fdc98-Abstract-Conference.html)

### Idea Summary
A two-stage residual learning pipeline designed for cross-tokenizer and same-tokenizer distillation:
- **Phase 1 (Projector Pretraining):** Freezes the student backbone and optimizes a projection head $\phi$ to fit teacher representations: $\min_\phi \|\phi(H_S^{\text{frozen}}) - H_T\|_2^2$.
- **Phase 2 (Residual Distillation):** Unfreezes the student backbone and trains it to learn the *residual difference* that the projector failed to capture, avoiding negative gradient interference between representation alignment and supervised token prediction.

### Mathematical Formulation
$$\text{Phase 1: } \min_{\phi} \left\| \phi(H_S^{\text{frozen}}) - H_T \right\|_2^2$$
$$\text{Phase 2: } \min_\theta \left\| H_T - \phi^*(H_S(\theta)) \right\|_2^2 + \mathcal{L}_{\text{CE}}(\theta)$$

### Recipe Reference
[`recipes/cross_tokenizer/residualkd_full.yaml`](../../recipes/cross_tokenizer/residualkd_full.yaml)

---

## 13. DWA-KD (Dual-Space Weighting and Time-Warped Alignment KD)

- **Official Paper:** *DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation*
- **Authors:** Hai An Vu, et al.
- **Publication / Venue:** Findings EACL 2026
- **Links:** [ACL Anthology](https://aclanthology.org/2026.findings-eacl.181/) · [arXiv:2602.21669](https://arxiv.org/abs/2602.21669)

### Idea Summary
DWA-KD combines two complementary mechanisms:
1. **Token-Level Dual-Space Weighting:** Applies an entropy-based weighting scheme to dual-space KL distillation, upweighting positions where the student is uncertain and the teacher is confident.
2. **Sequence-Level Time-Warped Alignment:** Employs **Soft Dynamic Time Warping (Soft-DTW)** across embedding and final hidden-state layers, aligning continuous semantic trajectories despite disparate sequence lengths and tokenizations.

### Mathematical Formulation
$$\mathcal{L}_{\text{DWA-KD}}(\theta) = \mathcal{L}_{\text{dual-weighted}}(\theta) + \gamma D_{\text{sDTW}}(H_S, H_T)$$

### Recipe Reference
[`recipes/cross_tokenizer/dwa.yaml`](../../recipes/cross_tokenizer/dwa.yaml)

---

## 14. MTA (Multi-Granular Trajectory Alignment Integration, DSKDv2 + MTA)

- **Official Paper:** *MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation*
- **Authors:** Pham Khanh Chi, Quoc Phong Dao, Thuat Nguyen, Linh Ngo Van, Trung Le, Thanh Hong Nguyen
- **Publication / Venue:** ACL 2026
- **Links:** [ACL Anthology](https://aclanthology.org/2026.acl-long.1507/)

### Idea Summary
In cross-tokenizer settings, `DSKDv2 + MTA` couples DSKDv2's pseudoinverse projection with MTA's phrase-level Dynamic Structural Alignment (DSA). Character-span boundary projections allow MTA to align hierarchical trajectory geometry across disparate tokenizers without requiring a common vocabulary.

### Mathematical Formulation
$$\mathcal{L}_{\text{DSKDv2+MTA}}(\theta) = \mathcal{L}_{\text{DSKDv2}}(\theta) + \beta \mathcal{L}_{\text{DSA}}(\theta)$$

### Recipe Reference
[`recipes/cross_tokenizer/dskdv2_mta.yaml`](../../recipes/cross_tokenizer/dskdv2_mta.yaml)
