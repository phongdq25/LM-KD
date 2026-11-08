# Same-Tokenizer Knowledge Distillation

Same-tokenizer distillation operates when the student and teacher models share the identical vocabulary and subword tokenizer ($\mathcal{V}_S \equiv \mathcal{V}_T$). Both models output probability vectors lying in the same discrete probability simplex $\Delta^{|\mathcal{V}|-1}$, enabling direct token-level distribution matching.

LM-KD supports all **12 core same-tokenizer baseline methods**.

---

## Baseline Summary & Paper Registry

| Baseline | Official Paper Title | Authors & Venue | Official Record / Paper Link | Key Focus |
| :--- | :--- | :--- | :--- | :--- |
| **FKL** | *Distilling the Knowledge in a Neural Network* | Hinton et al. (NeurIPS DL Workshop 2014; arXiv 2015) | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) | Foundational forward-KL soft-target distillation |
| **RKL** | *MiniLLM: Knowledge Distillation of Large Language Models* | Gu et al. (ICLR 2024) | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html) · [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) | Reverse KL mode-seeking distillation for generative LLMs |
| **JSD** | *f-Divergence Minimization for Sequence-Level Knowledge Distillation* | Wen et al. (ACL 2023) | [ACL Anthology](https://aclanthology.org/2023.acl-long.605/) · [arXiv:2307.15190](https://arxiv.org/abs/2307.15190) | Symmetric bounded f-divergence sequence distillation |
| **TVD** | *f-Divergence Minimization for Sequence-Level Knowledge Distillation* | Wen et al. (ACL 2023) | [ACL Anthology](https://aclanthology.org/2023.acl-long.605/) · [arXiv:2307.15190](https://arxiv.org/abs/2307.15190) | Total variation distance minimization under f-divergence KD |
| **SFKL** | *DistiLLM: Towards Streamlined Distillation for Large Language Models* | Ko et al. (ICML 2024) | [PMLR proceedings](https://proceedings.mlr.press/v235/ko24c.html) · [arXiv:2402.03898](https://arxiv.org/abs/2402.03898) | Skewed forward KL objective in DistiLLM framework |
| **SRKL** | *DistiLLM: Towards Streamlined Distillation for Large Language Models* | Ko et al. (ICML 2024) | [PMLR proceedings](https://proceedings.mlr.press/v235/ko24c.html) · [arXiv:2402.03898](https://arxiv.org/abs/2402.03898) | Skewed reverse KL stabilizing mode-seeking gradients |
| **CSD** | *Distillation of Large Language Models via Concrete Score Matching* | Kim et al. (ICLR 2026) | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1bfc9f74afa91b9b8add5a97a97001a1-Abstract-Conference.html) · [arXiv:2509.25837](https://arxiv.org/abs/2509.25837) | Concrete Score Distillation via logit shift-invariant matching |
| **DistiLLM** | *DistiLLM: Towards Streamlined Distillation for Large Language Models* | Ko et al. (ICML 2024) | [PMLR proceedings](https://proceedings.mlr.press/v235/ko24c.html) · [arXiv:2402.03898](https://arxiv.org/abs/2402.03898) | Adaptive Selective Generation Optimization (SGO) + SRKL |
| **GKD** | *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* | Agarwal et al. (ICLR 2024) | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html) · [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) | Generalized on-policy distillation mixing data and student rollouts |
| **FDD** | *Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation* | Gong et al. (ACL 2025) | [ACL Anthology](https://aclanthology.org/2025.acl-long.1125/) | Layer-wise feature progression and dynamical trajectory alignment |
| **AMiD** | *AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution* | Shin et al. (ICLR 2026) | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/54a468d19bc5727800096d9c7c51a423-Abstract-Conference.html) · [arXiv:2510.15982](https://arxiv.org/abs/2510.15982) | Continuous α-mixture assistant distribution interpolation |
| **MTA** | *MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation* | Chi et al. (ACL 2026) | [ACL Anthology](https://aclanthology.org/2026.acl-long.1507/) | Multi-Granular Trajectory Alignment across layer hierarchies |

---

## 1. Forward KL Divergence (FKL)

- **Official Paper:** *Distilling the Knowledge in a Neural Network*
- **Authors:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean
- **Publication / Venue:** NeurIPS Deep Learning Workshop 2014; arXiv 2015
- **Links:** [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

### Idea Summary
The classic knowledge distillation loss (Hinton et al., 2015). It optimizes the forward Kullback-Leibler divergence $\text{KL}(P_T \parallel P_S)$. FKL is **mode-covering** (zero-avoiding): it forces the student to place probability mass everywhere the teacher assigns non-zero probability. While this imparts broad multi-task coverage, in small-capacity students it can spread probability too thin, causing softer or hallucinated tokens if over-smoothed.

### Mathematical Formulation
$$\mathcal{L}_{\text{FKL}}(\theta) = \sum_{j=1}^L \sum_{v \in \mathcal{V}} P_T(v \mid s_j) \log \frac{P_T(v \mid s_j)}{P_\theta(v \mid s_j)}$$
Combined with the standard Supervised Fine-Tuning (SFT) hard cross-entropy loss:
$$\mathcal{L}_{\text{total}}(\theta) = \alpha_{\text{kd}} \cdot T^2 \cdot \mathcal{L}_{\text{FKL}}(\theta) + \alpha_{\text{sft}} \cdot \mathcal{L}_{\text{SFT}}(\theta)$$

### Algorithmic Implementation
```python
def forward_kl_loss(student_logits, teacher_logits, temperature=1.0):
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    log_p_s = F.log_softmax(student_logits / temperature, dim=-1)
    log_p_t = F.log_softmax(teacher_logits / temperature, dim=-1)
    kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
    return (kl * (temperature ** 2)).sum()
```

### Recipe Reference
[`recipes/same_tokenizer/fkl.yaml`](../../recipes/same_tokenizer/fkl.yaml)

---

## 2. Reverse KL Divergence (RKL)

- **Official Paper:** *MiniLLM: Knowledge Distillation of Large Language Models*
- **Authors:** Yuxian Gu, Li Dong, Furu Wei, Minlie Huang
- **Publication / Venue:** ICLR 2024
- **Links:** [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html) · [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)

### Idea Summary
Optimizes $\text{KL}(P_S \parallel P_T)$ on generative language models. Reverse KL is **mode-seeking** (zero-forcing): it heavily penalizes the student for placing probability mass where the teacher assigns low probability ($P_T \approx 0$). This forces the student to focus on high-probability modes, resulting in sharper, more confident responses and reducing hallucinations. MiniLLM derives a stable policy gradient algorithm to optimize RKL on student rollouts.

### Mathematical Formulation
$$\mathcal{L}_{\text{RKL}}(\theta) = \sum_{j=1}^L \sum_{v \in \mathcal{V}} P_\theta(v \mid s_j) \log \frac{P_\theta(v \mid s_j)}{P_T(v \mid s_j)}$$

### Algorithmic Implementation
```python
def reverse_kl_loss(student_logits, teacher_logits, temperature=1.0):
    p_s = F.softmax(student_logits / temperature, dim=-1)
    log_p_s = F.log_softmax(student_logits / temperature, dim=-1)
    log_p_t = F.log_softmax(teacher_logits / temperature, dim=-1)
    rkl = (p_s * (log_p_s - log_p_t)).sum(dim=-1)
    return (rkl * (temperature ** 2)).sum()
```

### Recipe Reference
[`recipes/same_tokenizer/rkl.yaml`](../../recipes/same_tokenizer/rkl.yaml)

---

## 3. Jensen-Shannon Divergence (JSD)

- **Official Paper:** *f-Divergence Minimization for Sequence-Level Knowledge Distillation*
- **Authors:** Yuqiao Wen, Zichao Li, Wenyu Du, Lili Mou
- **Publication / Venue:** ACL 2023
- **Links:** [ACL Anthology](https://aclanthology.org/2023.acl-long.605/) · [arXiv:2307.15190](https://arxiv.org/abs/2307.15190)

### Idea Summary
Derived and evaluated by Wen et al. (ACL 2023) as an $f$-divergence formulation for sequence-level distillation. Measures a symmetric divergence via a mixture distribution $M = \lambda P_S + (1-\lambda) P_T$ (typically $\lambda = 0.5$). JSD eliminates the loss blow-up and infinite penalties that occur when either model's probability approaches zero, providing smooth, bounded gradients.

### Mathematical Formulation
$$\mathcal{L}_{\text{JSD}}(\theta) = \frac{1}{2} D_{\text{KL}}(P_S \parallel M) + \frac{1}{2} D_{\text{KL}}(P_T \parallel M)$$

### Recipe Reference
[`recipes/same_tokenizer/jsd.yaml`](../../recipes/same_tokenizer/jsd.yaml)

---

## 4. Total Variation Distance (TVD)

- **Official Paper:** *f-Divergence Minimization for Sequence-Level Knowledge Distillation*
- **Authors:** Yuqiao Wen, Zichao Li, Wenyu Du, Lili Mou
- **Publication / Venue:** ACL 2023
- **Links:** [ACL Anthology](https://aclanthology.org/2023.acl-long.605/) · [arXiv:2307.15190](https://arxiv.org/abs/2307.15190)

### Idea Summary
Derived in Wen et al. (ACL 2023) within the generalized $f$-divergence distillation family. Computes the $L_1$-norm distance between student and teacher probability vectors: $0.5 \sum_{v \in \mathcal{V}} |P_T(v) - P_S(v)|$. It provides direct linear constraints on absolute deviations across the vocabulary simplex, avoiding exponential scaling penalties on outlier logits.

### Mathematical Formulation
$$\mathcal{L}_{\text{TVD}}(\theta) = \frac{1}{2} \sum_{j=1}^L \sum_{v \in \mathcal{V}} \left| P_T(v \mid s_j) - P_\theta(v \mid s_j) \right|$$

### Recipe Reference
[`recipes/same_tokenizer/tvd.yaml`](../../recipes/same_tokenizer/tvd.yaml)

---

## 5. Skewed Forward KL (SFKL)

- **Official Paper:** *DistiLLM: Towards Streamlined Distillation for Large Language Models*
- **Authors:** Jongwoo Ko, Seungjoon Park, Minwoo Jeong, Kyumin Park, Se-Young Yun
- **Publication / Venue:** ICML 2024
- **Links:** [PMLR proceedings](https://proceedings.mlr.press/v235/ko24c.html) · [arXiv:2402.03898](https://arxiv.org/abs/2402.03898)

### Idea Summary
SFKL is a loss component within the DistiLLM paper. It mixes the Teacher's distribution with the Student's using a skew parameter $\alpha \in (0, 1)$: $M = \alpha P_T + (1-\alpha) P_S$, and computes $-\sum P_T \log M$. This interpolation stabilizes optimization and prevents vanishing gradients on out-of-distribution prefixes.

### Mathematical Formulation
$$\mathcal{L}_{\text{SFKL}}(\theta) = D_{\text{KL}}\left( P_T \parallel \alpha P_T + (1 - \alpha) P_S \right)$$

### Recipe Reference
[`recipes/same_tokenizer/sfkl.yaml`](../../recipes/same_tokenizer/sfkl.yaml)

---

## 6. Skewed Reverse KL (SRKL)

- **Official Paper:** *DistiLLM: Towards Streamlined Distillation for Large Language Models*
- **Authors:** Jongwoo Ko, Seungjoon Park, Minwoo Jeong, Kyumin Park, Se-Young Yun
- **Publication / Venue:** ICML 2024
- **Links:** [PMLR proceedings](https://proceedings.mlr.press/v235/ko24c.html) · [arXiv:2402.03898](https://arxiv.org/abs/2402.03898)

### Idea Summary
SRKL is the primary divergence component introduced in DistiLLM (Ko et al., ICML 2024). It mixes the Student's distribution with the Teacher's: $M = (1-\alpha) P_S + \alpha P_T$, computing $\sum P_S (\log P_S - \log M)$. This stabilizes reverse-KL training without unbounded gradient spikes when student rollouts explore low-density teacher regions.

### Mathematical Formulation
$$\mathcal{L}_{\text{SRKL}}(\theta) = D_{\text{KL}}\left( P_S \parallel (1 - \alpha) P_S + \alpha P_T \right)$$

### Recipe Reference
[`recipes/same_tokenizer/srkl.yaml`](../../recipes/same_tokenizer/srkl.yaml)

---

## 7. Concrete Score Distillation (CSD)

- **Official Paper:** *Distillation of Large Language Models via Concrete Score Matching*
- **Authors:** Yeongmin Kim, Donghyeok Shin, Mina Kang, Byeonghu Na, Il-Chul Moon
- **Publication / Venue:** ICLR 2026
- **Links:** [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1bfc9f74afa91b9b8add5a97a97001a1-Abstract-Conference.html) · [arXiv:2509.25837](https://arxiv.org/abs/2509.25837) · [code](https://github.com/aailab-kaist/CSD)

### Idea Summary
CSD formulates distillation as discrete score matching directly at the logit level. While standard softmax-based KL divergences smooth out fine-grained logit details and direct logit distillation (MSE) restricts the student's solution space by violating logit shift-invariance (adding any scalar $c$ to all logits preserves probabilities), CSD matches the **pairwise logit differences** (the "concrete score"):
$$S(z)_{uv} = z(u) - z(v)$$
This preserves the exact relative ranking geometry of teacher tokens, respects shift invariance, and supports flexible weighting to achieve either mode-seeking or mode-covering behavior stably in reduced precision.

### Mathematical Formulation
$$\mathcal{L}\_{\text{CSD}}(\theta) = \sum_{j=1}^L \sum_{u, v \in \mathcal{V}} w(u, v) \left( [z_S(u \mid s_j) - z_S(v \mid s_j)] - [z_T(u \mid s_j) - z_T(v \mid s_j)] \right)^2$$
where $w(u, v)$ is an adaptive weighting function balancing token frequency and informative tail margins.

### Recipe Reference
[`recipes/same_tokenizer/csd.yaml`](../../recipes/same_tokenizer/csd.yaml)

---

## 8. DistiLLM (Adaptive SGO)

- **Official Paper:** *DistiLLM: Towards Streamlined Distillation for Large Language Models*
- **Authors:** Jongwoo Ko, Seungjoon Park, Minwoo Jeong, Kyumin Park, Se-Young Yun
- **Publication / Venue:** ICML 2024
- **Links:** [PMLR proceedings](https://proceedings.mlr.press/v235/ko24c.html) · [arXiv:2402.03898](https://arxiv.org/abs/2402.03898)

### Idea Summary
DistiLLM combines Skewed Reverse KL (SRKL) with **Adaptive Selective Generation Optimization (Adaptive SGO)**. Rather than generating expensive on-policy rollouts at every single step, it monitors validation loss metrics to trigger rollouts only on challenging sequence regions where the student encounters high loss. Combined with an off-policy replay buffer, it delivers up to $4.3\times$ speedups over standard on-policy distillation while preventing exposure bias.

### Mathematical Formulation
$$\mathcal{L}_{\text{DistiLLM}}(\theta) = \mathcal{L}_{\text{SRKL}}(\theta; \alpha) + \mathbb{I}(\Delta \mathcal{L}_{\text{val}} > \tau) \cdot \mathcal{L}_{\text{on-policy}}(\theta)$$

### Recipe Reference
[`recipes/same_tokenizer/distillm.yaml`](../../recipes/same_tokenizer/distillm.yaml)

---

## 9. Generalized Knowledge Distillation (GKD)

- **Official Paper:** *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*
- **Authors:** Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, Olivier Bachem
- **Publication / Venue:** ICLR 2024
- **Links:** [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html) · [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

### Idea Summary
GKD directly addresses exposure bias by mixing demonstration prefixes (off-policy) with student self-generated rollouts (on-policy) at probability ratio $\lambda \in [0, 1]$:
$$y \sim (1 - \lambda) \cdot P_{\text{data}} + \lambda \cdot P_\theta$$
On student rollouts, the teacher model evaluates the student's prefix step-by-step and provides online guidance, teaching the student how to recover when it drifts from optimal trajectories.

### Mathematical Formulation
$$\mathcal{L}_{\text{GKD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim (1-\lambda) P_{\text{data}} + \lambda P_\theta} \left[ \sum_{j=1}^{|y|} D\left( P_T(\cdot \mid x, y_{<j}) \parallel P_\theta(\cdot \mid x, y_{<j}) \right) \right] \right]$$

### Recipe Reference
[`recipes/same_tokenizer/gkd.yaml`](../../recipes/same_tokenizer/gkd.yaml)

---

## 10. FDD (Feature Dynamics Distillation)

- **Official Paper:** *Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation*
- **Authors:** Yichen Gong, et al.
- **Publication / Venue:** ACL 2025
- **Links:** [ACL Anthology](https://aclanthology.org/2025.acl-long.1125/)

### Idea Summary
Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation (ACL 2025). Rather than supervising only the final probability distribution, FDD aligns the layer-to-layer velocity and dynamical trajectory of hidden states as representations evolve through transformer blocks, ensuring internal computational trajectories mirror the teacher.

### Mathematical Formulation
$$\mathcal{L}\_{\text{FDD}}(\theta) = \mathcal{L}\_{\text{distill}} + \mu \sum_{l=1}^{L-1} \left\| \Delta h_S^{(l)} - W_l \Delta h_T^{(l)} \right\|_2^2$$
where $\Delta h^{(l)} = h^{(l+1)} - h^{(l)}$ captures feature progression dynamics across layers.

### Recipe Reference
[`recipes/same_tokenizer/distillm_fdd.yaml`](../../recipes/same_tokenizer/distillm_fdd.yaml)

---

## 11. AMiD (α-mixture Assistant Distribution)

- **Official Paper:** *AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution*
- **Authors:** Donghyeok Shin, Yeongmin Kim, Suhyeon Jo, Byeonghu Na, Il-Chul Moon
- **Publication / Venue:** ICLR 2026
- **Links:** [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/54a468d19bc5727800096d9c7c51a423-Abstract-Conference.html) · [arXiv:2510.15982](https://arxiv.org/abs/2510.15982)

### Idea Summary
AMiD constructs a generalized assistant distribution family by introducing a continuous distribution design variable $\alpha$. The $\alpha$-mixture assistant distribution dynamically interpolates between the Teacher distribution $P_T$ and Student distribution $P_S$ in unnormalized logit space. By aligning the student with this theoretically grounded intermediate assistant distribution, AMiD eliminates the capacity gap and prevents distribution blow-ups on complex generation tasks.

### Mathematical Formulation
$$\mathcal{L}_{\text{AMiD}}(\theta) = D\left( P_S \parallel r(\lambda, \alpha) \right) + \mathcal{L}_{\text{SFT}}$$
where the unnormalized assistant distribution $r_v$ is defined as:
$$\log r_v = \frac{2}{1 - \alpha} \log \left[ \lambda P_T(v)^{\frac{1-\alpha}{2}} + (1 - \lambda) P_S(v)^{\frac{1-\alpha}{2}} \right]$$

### Recipe Reference
[`recipes/same_tokenizer/amid.yaml`](../../recipes/same_tokenizer/amid.yaml)

---

## 12. MTA (Multi-Granular Trajectory Alignment)

- **Official Paper:** *MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation*
- **Authors:** Pham Khanh Chi, Quoc Phong Dao, Thuat Nguyen, Linh Ngo Van, Trung Le, Thanh Hong Nguyen
- **Publication / Venue:** ACL 2026
- **Links:** [ACL Anthology](https://aclanthology.org/2026.acl-long.1507/)

### Idea Summary
MTA aligns student and teacher internal representations along their layer-wise transformation trajectory. It employs a layer-adaptive granularity strategy:
1. **Lower layers** are aligned at the **word level** to capture fine-grained lexical signals.
2. **Higher layers** are aligned on **phrase-level syntactic spans** (noun chunks, verb phrases) to preserve high-level compositional semantics.
MTA optimizes a Dynamic Structural Alignment (DSA) loss matching relative geometry alongside a Hidden Representation Alignment (HRA) loss across intermediate layers.

### Mathematical Formulation
$$\mathcal{L}\_{\text{MTA}}(\theta) = \mathcal{L}_{\text{base-distill}} + \beta \mathcal{L}_{\text{DSA}} + \gamma \mathcal{L}_{\text{HRA}}$$
where:
$$\mathcal{L}_{\text{DSA}} = \sum_k \left( 1 - \frac{\text{SpanRep}_S^{(k)} \cdot \text{SpanRep}_T^{(k)}}{\|\text{SpanRep}_S^{(k)}\|_2 \|\text{SpanRep}_T^{(k)}\|_2} \right)$$

### Recipe Reference
- **DistiLLM + MTA:** [`recipes/same_tokenizer/distillm_mta.yaml`](../../recipes/same_tokenizer/distillm_mta.yaml)
- **AMiD + MTA:** Combined via `token_kd` (divergence: `amid`) with span-relational MTA aggregator.
