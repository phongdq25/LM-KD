# Long-CoT Reasoning Chain Distillation

Long-CoT distillation compresses extensive step-by-step reasoning chains (Chain-of-Thought, e.g. 4k–16k tokens) from frontier reasoning models (such as DeepSeek-R1, OpenAI o1-style models) into compact student models without sacrificing mathematical or algorithmic accuracy.

Standard uniform negative log-likelihood (NLL) treats all intermediate tokens equally, squandering model capacity on filler words while diluting gradients across critical reasoning steps.

LM-KD strictly supports the **3 core Long-CoT baseline methods**.

---

## Baseline Summary & Paper Registry

| Baseline | Official Paper Title | Authors & Venue | Official Record / Paper Link | Key Focus |
| :--- | :--- | :--- | :--- | :--- |
| **SSFT** | *Segment-Level Attribution for Selective Learning of Long Reasoning Traces* | Shenglong Wang, Yixin Liu, Xiang Ren (ICLR 2026) | [OpenReview PDF](https://openreview.net/pdf/7ba314ffe4e9d5c90878656e1656c7d6a34d2a4e.pdf) · [arXiv:2602.00425](https://arxiv.org/abs/2602.00425) | Integrated-gradients segment attribution & selective reasoning trace SFT |
| **P-ALIGN** | *Long-Chain Reasoning Distillation via Adaptive Prefix Alignment* | Liu et al. (ACL 2026) | [ACL Anthology](https://aclanthology.org/2026.acl-long.822/) · [arXiv:2601.10064](https://arxiv.org/abs/2601.10064) | Adaptive prefix-alignment distillation stripping redundant suffix chains |
| **SGL-Spectral** | *Uncovering the Gradient Geometry of Long CoT: A Spectral-guided Approach to Reasoning Distillation* | Sinan Fan et al. (ICML 2026) | [PMLR proceedings](https://proceedings.mlr.press/v306/fan26e.html) | SVD spectral filtering isolating consensus reasoning gradient geometry |

---

## 1. SSFT (Segment-Level Attribution for Selective Learning)

- **Official Paper:** *Segment-Level Attribution for Selective Learning of Long Reasoning Traces*
- **Authors:** Shenglong Wang, Yixin Liu, Xiang Ren
- **Publication / Venue:** ICLR 2026
- **Links:** [OpenReview PDF](https://openreview.net/pdf/7ba314ffe4e9d5c90878656e1656c7d6a34d2a4e.pdf) · [arXiv:2602.00425](https://arxiv.org/abs/2602.00425)

### Idea Summary
In lengthy reasoning trajectories (4k–16k tokens), many tokens represent superficial filler, repetition, or abandoned dead ends that dilute gradient updates. SSFT segments the reasoning trace into syntactic logical steps and uses **Integrated Gradients (IG)** to measure two key properties per segment:
1. **Attribution Strength:** The magnitude of gradient influence on predicting the final correct boxed answer.
2. **Direction Consistency:** Whether token attributions within the segment are consistently directed.

Segments exhibiting high attribution strength and reflective consistency are retained as vital deductive steps, while uninformative or noisy segments are masked out during fine-tuning.

### Mathematical Formulation
$$\text{IG}_k = (x_k - x_k') \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_k} d\alpha$$

$$\mathcal{L}_{\text{SSFT}}(\theta) = -\sum_{k \in \mathcal{S}_{\text{vital}}} \sum_{j \in \text{seg}_k} \log P_\theta(y_j \mid x, y_{<j})$$

where $\mathcal{S}_{\text{vital}} = \{ k \mid \text{AttributionStrength}(k) > \tau \text{ and } \text{Consistency}(k) > \eta \}$.

### Recipe Reference
[`recipes/long_cot/ssft.yaml`](../../recipes/long_cot/ssft.yaml)

---

## 2. P-ALIGN (Prefix-Alignment Reasoning Distillation)

- **Official Paper:** *Long-Chain Reasoning Distillation via Adaptive Prefix Alignment*
- **Authors:** Liu et al.
- **Publication / Venue:** ACL 2026
- **Links:** [ACL Anthology](https://aclanthology.org/2026.acl-long.822/) · [arXiv:2601.10064](https://arxiv.org/abs/2601.10064) · [code](https://github.com/NEUIR/P-ALIGN)

### Idea Summary
Direct distillation of sprawling teacher reasoning traces overtaxes smaller student models. P-ALIGN discovers that students only require the foundational, structural prefix of the reasoning chain to reach the correct answer independently:
1. **Adaptive Trajectory Truncation:** Evaluates the student model conditioned on varying prefix lengths of the teacher's thought process.
2. **Prefix Search:** Identifies the minimal inflection point $k^*$ from which the student can autonomously roll out the remaining steps to verify the correct solution.
3. **Prefix Alignment:** The student is supervised exclusively on the high-quality teacher prefix $1 \dots k^*$, cutting out redundant post-derivation verifications and training autonomous problem-solving.

### Mathematical Formulation
$$
k^* = \min \left\lbrace k \in [1, K] \quad\middle|\quad \mathbb{E}_{y \sim P_\theta(\cdot \mid x, \text{prefix}_k)} [\text{Verify}(y, y^*) = 1] \ge 1 - \epsilon \right\rbrace
$$

$$\mathcal{L}_{\text{P-ALIGN}}(\theta) = -\sum_{j=1}^{k^*} \log P_\theta(y_j^{\text{teacher}} \mid x, y_{<j}^{\text{teacher}})$$

### Recipe Reference
[`recipes/long_cot/palign.yaml`](../../recipes/long_cot/palign.yaml)

---

## 3. SGL-Spectral (Spectral-Guided Learning)

- **Official Paper:** *Uncovering the Gradient Geometry of Long CoT: A Spectral-guided Approach to Reasoning Distillation*
- **Authors:** Sinan Fan, et al.
- **Publication / Venue:** ICML 2026
- **Links:** [PMLR proceedings](https://proceedings.mlr.press/v306/fan26e.html)

### Idea Summary
SGL-Spectral examines the gradient geometry of long reasoning chains via **Loss Subspace Attribution**. It shows that gradients corresponding to genuine, reusable deductive logic naturally align into a **low-rank consensus subspace**, whereas instance-specific noise, erratic backtracking, and superficial tokens occupy a turbulent residual subspace.

Using **Singular Value Decomposition (SVD)** across sequence gradient vectors $G = [g_1, \dots, g_T] = U \Sigma V^\top$, SGL-Spectral projects step-level loss updates onto the dominant spectral consensus subspace, dynamically upweighting core deductive milestones while suppressing spectral noise.

### Mathematical Formulation
$$\mathcal{L}_{\text{SGL}}(\theta) = -\sum_{t=1}^T \omega_t^{\text{spectral}} \cdot \log P_\theta(y_t \mid x, y_{<t})$$

where $\omega_t^{\text{spectral}} = \| U_r^\top g_t \|_2^2$ projects the decoding gradient $g_t$ onto the top-$r$ principal spectral consensus subspace.

### Recipe Reference
[`recipes/long_cot/sgl_spectral.yaml`](../../recipes/long_cot/sgl_spectral.yaml)
