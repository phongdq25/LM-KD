# Developer Guides & How-To Manuals

This section provides practical, step-by-step guides for extending the LM-KD framework, implementing new algorithms, and customizing training pipelines.

---

## Guide Index

- **[Adding a New Baseline](adding-a-baseline.md):** The end-to-end checklist for adding a published baseline paper to the toolkit.
- **[Developing a Custom Loss](developing-custom-loss.md):** Complete walkthrough of authoring, validating, and testing a new distillation loss from scratch.
- **[Adding Models & API Teachers](adding-models.md):** How to integrate new open-weights models and proprietary black-box API teachers (Gemini, Claude, GPT-4).
- **[Adding Datasets & Custom Metrics](adding-datasets-and-metrics.md):** Registering data readers, prompt formatters, and custom evaluation metrics.
- **[LoRA & Parameter-Efficient Distillation](lora-and-peft.md):** Memory-efficient distillation with LoRA, QLoRA, and frozen teacher adapters.
- **[Cross-Tokenizer Adaptation](cross-tokenizer-adaptation.md):** Configuring subword aligners, learned projectors, and optimal transport mappings.
- **[Debugging & Performance Profiling](debugging-and-profiling.md):** Diagnosing CUDA out-of-memory errors, gradient NaN spikes, and profiling GPU throughput.

