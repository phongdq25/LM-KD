# Hands-On Tutorials & Walkthroughs

Step-by-step interactive tutorials demonstrating end-to-end distillation workflows.

---

## Tutorial Catalog

- **[Cross-Tokenizer Distillation: Qwen to OPT](distilling-llama-to-opt.md):** Complete walkthrough transferring knowledge from a 152k-vocab Qwen model to a 50k-vocab OPT student using DSKDv2 and subword DTW alignment.
- **[On-Policy Reasoning Self-Distillation](custom-policy-loss.md):** Implementing an interactive rollout loop where a model solves math problems, evaluates its own answers, and updates its policy in real time.
- **[Building a New Method with `lmkd init` (MTA + DistiLLM Adaptive Off-Policy + Privileged Teacher)](building-a-method-with-lmkd-init.md):** Complete walkthrough for scaffolding and customizing a new KD method using `lmkd init`, combining Multi-granularity Text Alignment (`MTA`), DistiLLM's adaptive off-policy producer (`AdaptiveSGOProducer`), and a Privileged Teacher receiving both prompt and reference response.

