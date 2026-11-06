# Self-Play Fine-Tuning (Adversarial Iterative Alignment)

Self-play fine-tuning formulates model alignment as an iterative two-player game. Rather than relying on external human preference datasets or fixed reward models, the student model plays against earlier iterations of itself, progressively discriminating human demonstrations from its own synthetic rollouts.

LM-KD supports all **6 core self-play fine-tuning baseline methods**.

Self-play methods are categorized under **Lagged-Policy Supervision** (Branch 3 of the LM-KD Taxonomy), operating via self-generated supervision rather than distilling an external teacher distribution into a student.

---

## Baseline Summary & Paper Registry

| Baseline | Official Paper Title | Authors & Venue | Official Record / Paper Link | Key Focus |
| :--- | :--- | :--- | :--- | :--- |
| **SPIN** | *Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models* | Zixiang Chen, et al. (ICML 2024) | [PMLR proceedings](https://proceedings.mlr.press/v235/chen24j.html) · [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) | Minimax two-player game discriminating data from rollouts |
| **T-SPIN** | *Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs* | Wang et al. (NeurIPS 2025) | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a797b10ff20562b1ecee0d4e914c1c7-Abstract-Conference.html) · [arXiv:2601.08198](https://arxiv.org/abs/2601.08198) | Triplet-based self-play with historical advantage anchoring |
| **SPACE** | *SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models* | Wang et al. (NeurIPS 2025) | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5692c7dbc4abcaa50f9ce609819212e5-Abstract-Conference.html) · [arXiv:2512.07175](https://arxiv.org/abs/2512.07175) | Noise Contrastive Estimation stabilizing gap-based self-play |
| **SPPO** | *Self-Play Preference Optimization for Language Model Alignment* | Yue Wu, et al. (ICLR 2025) | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e48fa1c4f08fd1ae35d5df8352c3106d-Abstract-Conference.html) · [arXiv:2405.00675](https://arxiv.org/abs/2405.00675) | Constant-sum zero-sum game with soft win-rate regression |
| **S-SPPO** | *S-SPPO: Semantic-Calibrated Self-Play Preference Optimization* | Xiwen Chen, et al. (ICML 2026) | [PMLR proceedings](https://proceedings.mlr.press/v306/chen26hc.html) · [arXiv:2606.01561](https://arxiv.org/abs/2606.01561) | Semantic gating and latent repulsion against manifold collapse |
| **RSPO** | *RSPO: Regularized Self-Play Alignment of Large Language Models* | Xiaohang Tang, et al. (ICML 2026) | [PMLR proceedings](https://proceedings.mlr.press/v306/tang26ab.html) · [arXiv:2503.00030](https://arxiv.org/abs/2503.00030) | Regularized self-play optimization with reference policy bounds |

---

## 1. SPIN (Self-Play Fine-Tuning)

- **Official Paper:** *Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models*
- **Authors:** Zixiang Chen, Tengyang Xie, Win-Yen Chen, Alexander Zhou, Arthur Douillard, et al.
- **Publication / Venue:** ICML 2024
- **Links:** [PMLR proceedings](https://proceedings.mlr.press/v235/ko24c.html) · [PMLR paper](https://proceedings.mlr.press/v235/chen24j.html) · [arXiv:2401.01335](https://arxiv.org/abs/2401.01335)

### Idea Summary
SPIN pits the current policy $\pi_\theta$ against an opponent policy parameterized by previous iteration rollouts $y' \sim \pi_{\text{ref}}(\cdot \mid x)$. The training objective encourages $\pi_\theta$ to assign higher relative likelihood to ground-truth human demonstrations $y$ than to its own earlier rollouts $y'$, converging toward the true data distribution as a minimax Nash equilibrium without external reward models.

### Mathematical Formulation
$$\mathcal{L}_{\text{SPIN}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}, y' \sim \pi_{\text{ref}}(\cdot \mid x)} \left[ \log \sigma \left( \beta \left[ \left(\log \pi_\theta(y \mid x) - \log \pi_\theta(y' \mid x)\right) - \left(\log \pi_{\text{ref}}(y \mid x) - \log \pi_{\text{ref}}(y' \mid x)\right) \right] \right) \right]$$

### Recipe Reference
[`recipes/self_play/spin.yaml`](../../recipes/self_play/spin.yaml)

---

## 2. T-SPIN (Triplet-Based Self-Play Fine-Tuning)

- **Official Paper:** *Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs*
- **Authors:** Wang et al.
- **Publication / Venue:** NeurIPS 2025
- **Links:** [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a797b10ff20562b1ecee0d4e914c1c7-Abstract-Conference.html) · [arXiv:2601.08198](https://arxiv.org/abs/2601.08198)

### Idea Summary
Traditional pair-based SPIN suffers from vanishing reward advantages as synthetic rollouts converge toward expert data. T-SPIN replaces pairs with triplets $(y, y', y_0)$:
- $y$: Expert-annotated demonstration
- $y'$: Current synthetic rollout
- $y_0$: Proto-synthetic rollout from the base/initial policy $\pi_0$

By anchoring against proto-synthetic outputs, T-SPIN maintains non-vanishing **historical advantages** alongside an entropy-regularized, reference-free objective, achieving strong stability with only 25% of annotated data.

### Mathematical Formulation
$$\mathcal{L}_{\text{T-SPIN}}(\theta) = -\log \sigma\left( \alpha \left[\log \pi_\theta(y \mid x) - \log \pi_\theta(y' \mid x)\right] \right) - \beta \log \sigma\left( \alpha \left[\log \pi_\theta(y' \mid x) - \log \pi_\theta(y_0 \mid x)\right] \right)$$

### Recipe Reference
[`recipes/self_play/tspin.yaml`](../../recipes/self_play/tspin.yaml)

---

## 3. SPACE (Self-Play via Noise Contrastive Estimation)

- **Official Paper:** *SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models*
- **Authors:** Wang et al.
- **Publication / Venue:** NeurIPS 2025
- **Links:** [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5692c7dbc4abcaa50f9ce609819212e5-Abstract-Conference.html) · [arXiv:2512.07175](https://arxiv.org/abs/2512.07175)

### Idea Summary
Existing self-play methods optimize relative probability gaps between real and synthetic data, which degenerate as the gap collapses. SPACE reframes self-play through Noise Contrastive Estimation: treating synthetic rollouts as an auxiliary noise distribution, it optimizes decoupled unnormalized reward likelihoods from a binary classification perspective, preventing objective collapse.

### Mathematical Formulation
$$\mathcal{L}_{\text{SPACE}}(\theta) = -\mathbb{E}_{y \sim \mathcal{D}} \left[ \log \frac{\exp(f_\theta(x, y))}{\exp(f_\theta(x, y)) + \nu P_{\text{noise}}(y \mid x)} \right] - \nu \mathbb{E}_{y' \sim P_{\text{noise}}} \left[ \log \frac{P_{\text{noise}}(y' \mid x)}{\exp(f_\theta(x, y')) + \nu P_{\text{noise}}(y' \mid x)} \right]$$

### Recipe Reference
[`recipes/self_play/space.yaml`](../../recipes/self_play/space.yaml)

---

## 4. SPPO (Self-Play Preference Optimization)

- **Official Paper:** *Self-Play Preference Optimization for Language Model Alignment*
- **Authors:** Yue Wu, Zhiqing Sun, Huizhuo Yuan, Jiayi Ji, Yiming Yang, Quanquan Gu
- **Publication / Venue:** ICLR 2025
- **Links:** [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e48fa1c4f08fd1ae35d5df8352c3106d-Abstract-Conference.html) · [arXiv:2405.00675](https://arxiv.org/abs/2405.00675)

### Idea Summary
SPPO models preference optimization as a symmetric two-player zero-sum game, finding the Nash equilibrium policy through iterative soft win-rate regression. Rather than binary chosen/rejected classification, it computes exact theoretical win probabilities $P_{\text{win}}$ against the opponent population and regresses log-odds margins directly to this soft game-theoretic equilibrium.

### Mathematical Formulation
$$\mathcal{L}_{\text{SPPO}}(\theta) = \frac{1}{2} \left( \Delta \log \pi_w - \frac{1}{\beta}(P_{\text{win}} - 0.5) \right)^2 + \frac{1}{2} \left( \Delta \log \pi_l - \frac{1}{\beta}(P_{\text{lose}} - 0.5) \right)^2$$

where $\Delta \log \pi = \log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)$.

### Recipe Reference
[`recipes/self_play/sppo.yaml`](../../recipes/self_play/sppo.yaml)

---

## 5. S-SPPO (Semantic-Calibrated Self-Play Preference Optimization)

- **Official Paper:** *S-SPPO: Semantic-Calibrated Self-Play Preference Optimization*
- **Authors:** Xiwen Chen, et al.
- **Publication / Venue:** ICML 2026
- **Links:** [PMLR proceedings](https://proceedings.mlr.press/v306/chen26hc.html) · [arXiv:2606.01561](https://arxiv.org/abs/2606.01561)

### Idea Summary
Standard SPPO degenerates when preference oracles assign confident win/loss scores to responses that are semantically identical. S-SPPO introduces a dual-space semantic calibration framework:
1. **Semantic Gating:** Dynamically adjusts win-rate regression targets toward maximum entropy as semantic overlap $(1 - \text{sim-score})$ increases, preventing overconfidence on semantic twins.
2. **Latent Repulsion:** Enforces geometric diversity $\lambda \langle h_w, h_l \rangle$ in intermediate representation spaces to prevent latent manifold collapse.

### Mathematical Formulation
$$\mathcal{L}_{\text{S-SPPO}}(\theta) = \mathcal{L}_{\text{SPPO}}(\theta; (1 - \text{sim-score})) + \lambda \langle h_w, h_l \rangle$$

### Recipe Reference
[`recipes/self_play/s_sppo.yaml`](../../recipes/self_play/s_sppo.yaml)

---

## 6. RSPO (Regularized Self-Play Optimization)

- **Official Paper:** *RSPO: Regularized Self-Play Alignment of Large Language Models*
- **Authors:** Xiaohang Tang, Sangwoong Yoon, Seongho Son, Huizhuo Yuan, Quanquan Gu, Ilija Bogunovic
- **Publication / Venue:** ICML 2026
- **Links:** [PMLR proceedings](https://proceedings.mlr.press/v306/tang26ab.html) · [arXiv:2503.00030](https://arxiv.org/abs/2503.00030) · [code](https://github.com/xiaohangt/RSPO)

### Idea Summary
RSPO provides a modular framework incorporating reference policy regularizers into self-play game optimization with provable convergence to regularized Nash equilibria:
- **Forward KL regularization:** Controls runaway generation verbosity and length inflation.
- **Reverse KL regularization:** Directly enhances competitive win rates on open-ended generation benchmarks (AlpacaEval-2).

### Mathematical Formulation
$$\mathcal{L}_{\text{RSPO}}(\theta) = \mathcal{L}_{\text{SPPO}}(\theta) + \mu_1 \cdot D_{\text{KL}}\left( \pi_{\text{ref}} \parallel \pi_\theta \right) + \mu_2 \cdot D_{\text{KL}}\left( \pi_\theta \parallel \pi_{\text{ref}} \right)$$

### Recipe Reference
[`recipes/self_play/rspo.yaml`](../../recipes/self_play/rspo.yaml)
