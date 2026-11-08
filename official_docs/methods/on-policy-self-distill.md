# On-Policy & Self-Distillation

On-policy self-distillation enables a language model to bootstrap its own reasoning and problem-solving abilities by generating exploratory rollouts during training. Rather than passively imitating static offline text, the student samples its own solutions, receives verification or environment feedback, and aligns with an updated self-teacher or reference distribution.

LM-KD strictly supports the **3 core on-policy & self-distillation baseline methods**.

Unlike GKD (Agarwal et al., ICLR 2024)—which relies on an external Teacher model guiding student rollouts—OPSD, SDPO, and RLSD are self-distillation methods where the **same model serves as its own supervisory source** under privileged conditioning or verification feedback.

---

## Baseline Summary & Paper Registry

| Baseline | Official Paper Title | Authors & Venue | Official Record / Paper Link | Key Focus |
| :--- | :--- | :--- | :--- | :--- |
| **OPSD** | *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models* | Zhao et al. (ICML 2026) | [PMLR proceedings](https://proceedings.mlr.press/v306/zhao26be.html) · [arXiv:2601.18734](https://arxiv.org/abs/2601.18734) | Privileged-context teacher rationalizing ordinary student on-policy rollouts |
| **SDPO** | *Reinforcement Learning via Self-Distillation* | Jonas Hübotter, Frederike Lübeck, et al. (ICML 2026) | [PMLR proceedings](https://proceedings.mlr.press/v306/hubotter26a.html) · [arXiv:2601.20802](https://arxiv.org/abs/2601.20802) | Self-Distillation Policy Optimization leveraging rich textual feedback |
| **RLSD** | *Self-Distilled RLVR* | Yang et al. (arXiv 2026; preprint) | [arXiv:2604.03128](https://arxiv.org/abs/2604.03128) · [code](https://github.com/iie-ycx/RLSD) | RLVR feedback for update direction + self-distillation for token magnitude |

---

## 1. OPSD (On-Policy Self-Distillation)

- **Official Paper:** *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*
- **Authors:** Zhao et al.
- **Publication / Venue:** ICML 2026
- **Links:** [PMLR proceedings](https://proceedings.mlr.press/v306/zhao26be.html) · [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)

### Idea Summary
OPSD eliminates the need for an external teacher by having a single LLM act as both teacher and student under different prompt conditions:
- **Student Policy:** Receives standard problem context $x$ and generates solutions on-policy ($y \sim P_\theta(\cdot \mid x)$) via high-throughput rollout engines (vLLM).
- **Privileged Teacher Policy:** The same model receives privileged information $x_{\text{priv}}$ (such as reference answers or verification hints) to generate a dense supervisory target distribution $P_\theta(\cdot \mid x_{\text{priv}}, y_{<t})$.

Instead of standard forward KL, OPSD distills using **Generalized JSD** or **Top-$k$ Reverse KL** over the student's on-policy rollouts, penalizing mode collapse and achieving 4–8x greater token efficiency than reinforcement learning.

### Mathematical Formulation
$$\mathcal{L}_{\text{OPSD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim P_\theta(\cdot \mid x)} \left[ \sum_{t=1}^{|y|} D_{\text{GenJSD}}\left( P_{\theta_{\text{frozen}}}(\cdot \mid x_{\text{priv}}, y_{<t}) \parallel P_\theta(\cdot \mid x, y_{<t});\, \beta \right) \right]$$

where $D_{\text{GenJSD}}(P \parallel Q; \beta) = \beta D_{\text{KL}}(P \parallel M) + (1-\beta) D_{\text{KL}}(Q \parallel M)$ with $M = \beta P + (1-\beta) Q$.

### Recipe Reference
[`recipes/self_distill/opsd.yaml`](../../recipes/self_distill/opsd.yaml)

---

## 2. SDPO (Self-Distillation Policy Optimization)

- **Official Paper:** *Reinforcement Learning via Self-Distillation*
- **Authors:** Jonas Hübotter, Frederike Lübeck, Lejs Deen Behric, et al.
- **Publication / Venue:** ICML 2026
- **Links:** [PMLR proceedings](https://proceedings.mlr.press/v306/hubotter26a.html) · [arXiv:2601.20802](https://arxiv.org/abs/2601.20802)

### Idea Summary
Standard RLVR relies on sparse scalar rewards (0 or 1), which creates an severe credit assignment bottleneck. SDPO formalizes **Reinforcement Learning with Rich Feedback (RLRF)**:
1. When an on-policy rollout fails or encounters runtime compiler errors, unit test assertions, or judge critiques, the detailed textual feedback is preserved.
2. The model is conditioned retrospectively on this rich feedback in-context to analyze and rectify its own mistakes:
   $$x_{\text{correct}} = (x,\, y^-,\, \text{Feedback: compiler/test failure})$$
3. SDPO distills these feedback-informed, corrected next-token predictions directly into the base policy as dense, token-level supervisory signals.

### Mathematical Formulation
$$\mathcal{L}_{\text{SDPO}}(\theta) = -\mathbb{E}_{(x, y^-) \sim \mathcal{D}_{\text{rollout}}} \left[ \sum_{t=1}^{|y^+|} \log P_\theta(y_t^+ \mid x_{\text{correct}}, y_{<t}^+) \right] + \beta D_{\text{KL}}\left( P_\theta \parallel P_{\text{ref}} \right)$$

### Recipe Reference
[`recipes/self_distill/sdpo.yaml`](../../recipes/self_distill/sdpo.yaml)

---

## 3. RLSD (Self-Distilled RLVR)

- **Official Paper:** *Self-Distilled RLVR*
- **Authors:** Yang et al.
- **Publication / Venue:** arXiv 2026 (Preprint)
- **Links:** [arXiv:2604.03128](https://arxiv.org/abs/2604.03128) · [code](https://github.com/iie-ycx/RLSD)

### Idea Summary
RLSD couples Reinforcement Learning with Verifiable Rewards (RLVR) and Self-Distillation into a synergistic loop. While RLVR provides overall policy update direction based on verifiable answer correctness ($\boxed{\dots}$ verification), it suffers from high gradient variance. RLSD uses self-distillation from high-reward sampled trajectories to determine exact token-level update magnitudes, constraining policy exploration entropy and preventing policy collapse.

### Mathematical Formulation
$$\mathcal{L}_{\text{RLSD}}(\theta) = -\frac{1}{n} \sum_{i=1}^n \hat{A}_i \cdot \log P_\theta(y_i \mid x) + \alpha \cdot D_{\text{KL}}\left( P_{\text{self-teacher}}(\cdot \mid x) \parallel P_\theta(\cdot \mid x) \right)$$

where $\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}$ is the group-relative advantage estimated across sampled rollouts.

### Recipe Reference
[`recipes/self_distill/rlsd.yaml`](../../recipes/self_distill/rlsd.yaml)
