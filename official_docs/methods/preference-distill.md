# Preference Distillation (Reward-Model-Free Alignment)

Traditional Reinforcement Learning from Human Feedback (RLHF) requires an expensive, multi-stage pipeline: training a separate Reward Model (RM) followed by unstable policy gradient optimization (PPO). Preference Distillation aligns student models directly by transferring a capable Teacher's preference distribution over pairs $(x, y^+, y^-)$ without requiring a separate reward model.

LM-KD supports all **8 core preference distillation baseline methods**.

---

## Baseline Summary & Paper Registry

| Baseline | Official Paper Title | Authors & Venue | Official Record / Paper Link | Key Focus |
| :--- | :--- | :--- | :--- | :--- |
| **VPD (PAD)** | *Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models* | Gu et al. (Findings ACL 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-acl.822/) | Vanilla Preference Distillation objective in PAD |
| **PPD (PAD)** | *Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models* | Gu et al. (Findings ACL 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-acl.822/) | Probabilistic Preference Distillation objective in PAD |
| **DCKD** | *Advantage-Guided Distillation for Preference Alignment in Small Language Models* | Gao et al. (ICLR 2025) | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html) · [arXiv:2502.17927](https://arxiv.org/abs/2502.17927) | Dual-Constrained Knowledge Distillation from aligned teacher |
| **ADPA** | *Advantage-Guided Distillation for Preference Alignment in Small Language Models* | Gao et al. (ICLR 2025) | [Same ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html) · [arXiv:2502.17927](https://arxiv.org/abs/2502.17927) | Advantage-Guided Distillation for Preference Alignment |
| **TVKD** | *Preference Distillation via Value-based Reinforcement Learning* | Minchan Kwon, Junwon Ko, et al. (NeurIPS 2025) | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/98e967164ae2f6811b975d686dece3eb-Abstract-Conference.html) | Teacher Value-based KD via potential-based reward shaping |
| **CTPD** | *CTPD: Cross Tokenizer Preference Distillation* | Truong Nguyen, et al. (AAAI 2026) | [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/41114) | Cross-tokenizer preference alignment via span projection & TIS-DPO |
| **DP3O** | *Towards Bridging the Gap Between Offline and Iterative Alignment via Preference Distillation* | Wenbo Zhang, et al. (TMLR 2026) | [arXiv:2609.06893](https://arxiv.org/abs/2609.06893) | Distilled Preference Probability Policy Optimization |
| **PBSD** | *Preference-Based Self-Distillation: Beyond KL Matching via Reward Regularization* | Xin Yu, et al. (arXiv 2026; preprint) | [arXiv:2605.05040](https://arxiv.org/abs/2605.05040) | Preference-based self-distillation via reward-regularized target |

---

## 1. VPD (Vanilla Preference Distillation / PAD)

- **Official Paper:** *Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models*
- **Authors:** Yuxian Gu, et al.
- **Publication / Venue:** Findings ACL 2025
- **Links:** [ACL Anthology](https://aclanthology.org/2025.findings-acl.822/)

### Idea Summary
VPD distills preference information from an aligned teacher by computing forward KL divergence on the preferred completion $y^+$ while penalizing probability mass on the dispreferred completion $y^-$, forcing the student to align with the teacher's preference margin:
$$\Delta_T(x, y^+, y^-) = \log P_T(y^+ \mid x) - \log P_T(y^- \mid x)$$

### Mathematical Formulation
$$\mathcal{L}_{\text{VPD}}(\theta) = D_{\text{KL}}\left( P_T(\cdot \mid x, y^+) \parallel P_\theta(\cdot \mid x, y^+) \right) - \lambda \cdot D_{\text{KL}}\left( P_T(\cdot \mid x, y^-) \parallel P_\theta(\cdot \mid x, y^-) \right)$$

### Recipe Reference
[`recipes/preference/pad.yaml`](../../recipes/preference/pad.yaml) (or [`recipes/preference/vpd.yaml`](../../recipes/preference/vpd.yaml))

---

## 2. PPD (Probabilistic Preference Distillation / PAD)

- **Official Paper:** *Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models*
- **Authors:** Yuxian Gu, et al.
- **Publication / Venue:** Findings ACL 2025
- **Links:** [ACL Anthology](https://aclanthology.org/2025.findings-acl.822/)

### Idea Summary
Rather than supervising hard binary completion margins, PPD converts the Teacher's log-probability difference into a soft Plackett-Luce preference probability:
$$P_T(y^+ \succ y^- \mid x) = \sigma\left( \frac{\log P_T(y^+ \mid x) - \log P_T(y^- \mid x)}{\tau} \right)$$
The student policy minimizes cross-entropy against this soft teacher preference distribution, capturing nuanced preference degrees without over-fitting to extreme margins.

### Mathematical Formulation
$$\mathcal{L}_{\text{PPD}}(\theta) = -\left[ P_T(y^+ \succ y^- \mid x) \log P_\theta(y^+ \succ y^- \mid x) + P_T(y^- \succ y^+ \mid x) \log P_\theta(y^- \succ y^+ \mid x) \right]$$

### Recipe Reference
[`recipes/preference/ppd.yaml`](../../recipes/preference/ppd.yaml)

---

## 3. DCKD (Dual-Constrained Knowledge Distillation)

- **Official Paper:** *Advantage-Guided Distillation for Preference Alignment in Small Language Models*
- **Authors:** Shiping Gao, Fanqi Wan, Jiajian Guo, Xiaojun Quan, Qifan Wang
- **Publication / Venue:** ICLR 2025 (Spotlight)
- **Links:** [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html) · [arXiv:2502.17927](https://arxiv.org/abs/2502.17927) · [code](https://github.com/SLIT-AI/ADPA)

### Idea Summary
Direct preference optimization can cause small models to suffer from the "alignment tax," degrading basic language capabilities. DCKD applies dual constraints:
1. Standard forward KL distillation from the aligned teacher on the chosen response $y^+$ to preserve core generative quality.
2. A margin-based contrastive penalty ensuring the student's log-ratio difference matches or exceeds the teacher's margin on $(y^+, y^-)$.

### Mathematical Formulation
$$\mathcal{L}_{\text{DCKD}}(\theta) = D_{\text{KL}}\left( P_T(\cdot \mid x, y^+) \parallel P_\theta(\cdot \mid x, y^+) \right) + \max\left( 0,\, \gamma - \left[ \log \frac{P_\theta(y^+ \mid x)}{P_\theta(y^- \mid x)} - \log \frac{P_T(y^+ \mid x)}{P_T(y^- \mid x)} \right] \right)$$

### Recipe Reference
[`recipes/preference/dckd.yaml`](../../recipes/preference/dckd.yaml)

---

## 4. TVKD (Teacher Value-based Knowledge Distillation)

- **Official Paper:** *Preference Distillation via Value-based Reinforcement Learning*
- **Authors:** Minchan Kwon, Junwon Ko, Kangil Kim, Junmo Kim
- **Publication / Venue:** NeurIPS 2025
- **Links:** [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/98e967164ae2f6811b975d686dece3eb-Abstract-Conference.html) · [arXiv:2506.01234](https://arxiv.org/abs/2406.19830)

### Idea Summary
Standard DPO provides sparse sequence-level binary win/loss signals. TVKD derives an auxiliary value function from the internal states of the teacher model. Using **potential-based reward shaping**, it injects dense intermediate value guidance into the student's optimization trajectory while mathematically preserving the global reward structure and optimal policy of the base DPO framework.

### Mathematical Formulation
$$\mathcal{L}_{\text{TVKD}}(\theta) = \mathcal{L}_{\text{DPO}}(\theta) + \beta \left( \Phi_T(y^+) - \Phi_T(y^-) - \left[ \Phi_S(y^+) - \Phi_S(y^-) \right] \right)^2$$

where $\Phi_T(y)$ is the teacher's potential-based value function estimate.

### Recipe Reference
[`recipes/preference/tvkd.yaml`](../../recipes/preference/tvkd.yaml)

---

## 5. ADPA (Advantage-Guided Distillation for Preference Alignment)

- **Official Paper:** *Advantage-Guided Distillation for Preference Alignment in Small Language Models*
- **Authors:** Shiping Gao, Fanqi Wan, Jiajian Guo, Xiaojun Quan, Qifan Wang
- **Publication / Venue:** ICLR 2025 (Spotlight)
- **Links:** [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html) · [arXiv:2502.17927](https://arxiv.org/abs/2502.17927) · [code](https://github.com/SLIT-AI/ADPA)

### Idea Summary
ADPA extracts an implicit advantage function $A_T(x, y)$ from the aligned teacher model to supply fine-grained distribution-level reward signals. Rather than weighting all preference pairs equally, it weights updates by the teacher's advantage differential $A_T(x, y^+) - A_T(x, y^-)$, focusing student learning capacity on pairs where the teacher has clear, reliable preference certainty.

### Mathematical Formulation
$$w(x, y^+, y^-) = \sigma\left( \frac{A_T(x, y^+) - A_T(x, y^-)}{\tau} \right)$$
$$\mathcal{L}_{\text{ADPA}}(\theta) = w(x, y^+, y^-) \cdot \mathcal{L}_{\text{DPO}}(\theta; y^+, y^-) + \alpha \mathcal{L}_{\text{KD}}(P_T, P_\theta)$$

### Recipe Reference
[`recipes/preference/adpa.yaml`](../../recipes/preference/adpa.yaml)

---

## 6. CTPD (Cross Tokenizer Preference Distillation)

- **Official Paper:** *CTPD: Cross Tokenizer Preference Distillation*
- **Authors:** Truong Nguyen, Phi Van Dat, Ngan Nguyen, et al.
- **Publication / Venue:** AAAI 2026
- **Links:** [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/41114)

### Idea Summary
CTPD solves preference alignment when student and teacher use completely different tokenizers. It combines two core mechanisms:
1. **Aligned Span Projection:** Projects teacher and student sequences into shared character-level spans, allowing token log-probabilities to be compared across subword boundaries.
2. **Token-level Importance Sampling (TIS-DPO):** Uses importance sampling ratios to reweight DPO sequence gradients, ensuring unbiased credit assignment across disparate token length spaces.

### Mathematical Formulation
$$\mathcal{L}_{\text{CTPD}}(\theta) = -\log \sigma \left( \beta \sum_{k \in \text{spans}} w_k^{\text{IS}} \left[ \log \frac{P_\theta(\text{span}_k^+)}{P_{\text{ref}}(\text{span}_k^+)} - \log \frac{P_\theta(\text{span}_k^-)}{P_{\text{ref}}(\text{span}_k^-)} \right] \right)$$

### Recipe Reference
[`recipes/preference/ctpd.yaml`](../../recipes/preference/ctpd.yaml)

---

## 7. DP3O (Distilled Preference Probability Policy Optimization)

- **Official Paper:** *Towards Bridging the Gap Between Offline and Iterative Alignment via Preference Distillation*
- **Authors:** Wenbo Zhang, Wenzhuo Zhou, Hengrui Cai, Zhengling Qi
- **Publication / Venue:** TMLR 2026
- **Links:** [arXiv:2609.06893](https://arxiv.org/abs/2609.06893)

### Idea Summary
Iterative alignment methods (e.g., iterative DPO) achieve superior alignment over static offline DPO, but require costly online generation rollouts. DP3O bridges this gap completely offline: it first trains an explicit parameterized preference model $P_{\text{pref}}$ on static preference data, and then distills this preference probability directly into the policy optimization objective. It cuts training time by ~42% compared to iterative DPO while matching iterative alignment performance.

### Mathematical Formulation
$$\mathcal{L}_{\text{DP3O}}(\theta) = \mathbb{E}_{(x, y^+, y^-)} \left[ \mathcal{L}_{\text{DPO}}(\theta) + \mu \cdot D_{\text{KL}}\left( P_{\text{pref}}(\cdot \mid x) \parallel \sigma\left( \beta \log \frac{\pi_\theta}{\pi_{\text{ref}}} \right) \right) \right]$$

### Recipe Reference
[`recipes/preference/dp3o.yaml`](../../recipes/preference/dp3o.yaml)

---

## 8. PBSD (Preference-Based Self-Distillation)

- **Official Paper:** *Preference-Based Self-Distillation: Beyond KL Matching via Reward Regularization*
- **Authors:** Xin Yu, Liuchen Liao, Yiwen Zhang, Yingchen Yu, Lingzhou Xue, Qinzhen Guo
- **Publication / Venue:** arXiv 2026 (Preprint)
- **Links:** [arXiv:2605.05040](https://arxiv.org/abs/2605.05040)

### Idea Summary
Existing self-distillation frameworks reduce learning to strict KL divergence matching against a context-augmented teacher, causing training instability and loss of diversity. PBSD reformulates self-distillation through a **reward-regularized perspective**: treating the teacher as a reference measure, the optimal target policy is a reward-reweighted distribution provably superior to the original teacher. PBSD optimizes preference gaps between teacher- and student-generated rollouts on-policy.

### Mathematical Formulation
$$\mathcal{L}_{\text{PBSD}}(\theta) = \mathbb{E}_{y_T \sim \pi_T, y_S \sim \pi_S} \left[ -\log \sigma\left( r_\phi(x, y_T) - r_\phi(x, y_S) \right) \right] + \lambda D_{\text{KL}}\left( \pi_\theta \parallel \pi_{\text{anchor}} \right)$$

### Recipe Reference
[`recipes/preference/pbsd.yaml`](../../recipes/preference/pbsd.yaml)
