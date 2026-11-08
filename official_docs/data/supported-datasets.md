# Supported Datasets & Benchmarks Catalog

This catalog details all supported training and evaluation datasets in LM-KD, including task category, local workspace path, and official Hugging Face Hub link.

---

## Master Datasets Directory

| Dataset Name | Role & Domain | Local Path in `lm-kd/` | Hugging Face Hub URL |
| :--- | :--- | :--- | :--- |
| **Banking77** | Dev, Test (Embedding Classification) | [`datasets/embedding_data/multi-data/banking77_test.csv`](../../datasets/embedding_data/multi-data/banking77_test.csv) | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) |
| **TweetEval** | Dev, Test (Embedding Classification) | [`datasets/embedding_data/multi-data/tweet_test.csv`](../../datasets/embedding_data/multi-data/tweet_test.csv) | [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) |
| **Emotion** | Train, Dev, Test (Embedding In-domain) | [`datasets/embedding_data/multi-data/emotion_validation.csv`](../../datasets/embedding_data/multi-data/emotion_validation.csv) | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) |
| **MRPC** | Dev, Test (Embedding Pairwise) | [`datasets/embedding_data/multi-data/mrpc_test.csv`](../../datasets/embedding_data/multi-data/mrpc_test.csv) | [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) |
| **SciTail** | Dev, Test (Embedding Entailment) | [`datasets/embedding_data/multi-data/scitail_test.csv`](../../datasets/embedding_data/multi-data/scitail_test.csv) | [allenai/scitail](https://huggingface.co/datasets/allenai/scitail) |
| **WiC** | Train, Dev, Test (Embedding In-domain) | [`datasets/embedding_data/multi-data/wic_validation.csv`](../../datasets/embedding_data/multi-data/wic_validation.csv) | [super_glue](https://huggingface.co/datasets/super_glue) |
| **SICK** | Dev, Test (Embedding STS) | [`datasets/embedding_data/multi-data/sick_validation.csv`](../../datasets/embedding_data/multi-data/sick_validation.csv) | [sick](https://huggingface.co/datasets/sick) |
| **STS12** | Dev, Test (Embedding STS) | [`datasets/embedding_data/multi-data/sts12_validation.csv`](../../datasets/embedding_data/multi-data/sts12_validation.csv) | [mteb/sts12-sts](https://huggingface.co/datasets/mteb/sts12-sts) |
| **STSB** | Train, Dev, Test (Embedding In-domain) | [`datasets/embedding_data/multi-data/stsb_validation.csv`](../../datasets/embedding_data/multi-data/stsb_validation.csv) | [mteb/stsbenchmark-sts](https://huggingface.co/datasets/mteb/stsbenchmark-sts) |
| **Databricks Dolly 15k** | Train, Dev, Test (Instruction KD) | [`datasets/same_cross_tokenizer_data/dolly/`](../../datasets/same_cross_tokenizer_data/dolly/) | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| **Self-Instruct** | Test (Instruction Following) | [`datasets/same_cross_tokenizer_data/self-inst/valid.jsonl`](../../datasets/same_cross_tokenizer_data/self-inst/valid.jsonl) | [yizhongw/self_instruct](https://huggingface.co/datasets/yizhongw/self_instruct) |
| **Vicuna Evaluation** | Test (Instruction Benchmark) | [`datasets/same_cross_tokenizer_data/vicuna/valid.jsonl`](../../datasets/same_cross_tokenizer_data/vicuna/valid.jsonl) | [lmsys/vicuna-eval](https://huggingface.co/datasets/lmsys/vicuna-eval) |
| **Super-NaturalInstructions** | Test (Diverse Generalization) | [`datasets/same_cross_tokenizer_data/sinst/11_/valid.jsonl`](../../datasets/same_cross_tokenizer_data/sinst/11_/valid.jsonl) | [allenai/super-naturalinstructions](https://huggingface.co/datasets/allenai/super-naturalinstructions) |
| **s1K-1.1** | Train (Reasoning / Long-CoT) | — | [simplescaling/s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1) |
| **LIMO** | Train/Data (Long-CoT) | [`datasets/long_CoT_data/limo/`](../../datasets/long_CoT_data/limo/) | [GAIR/LIMO](https://huggingface.co/datasets/GAIR/LIMO) |
| **AIME 2024** | Benchmark (Math Olympiad) | [`datasets/long_CoT_data/aime24/test.jsonl`](../../datasets/long_CoT_data/aime24/test.jsonl) | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) |
| **AIME 2025** | Benchmark (Math Olympiad) | — | [HuggingFaceH4/aime_2025](https://huggingface.co/datasets/HuggingFaceH4/aime_2025) |
| **AMC 12 / AMC 23** | Benchmark (Competition Math) | [`datasets/long_CoT_data/amc23/test.jsonl`](../../datasets/long_CoT_data/amc23/test.jsonl) | [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math) |
| **MATH-500** | Benchmark (Challenging Math) | [`datasets/long_CoT_data/math500/test.jsonl`](../../datasets/long_CoT_data/math500/test.jsonl) | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| **GSM8K** | Train & Test (Grade School Math) | — | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| **UltraFeedback Binarized**| Train (Preference Distillation) | — (HuggingFace Hub: `train_prefs`) | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) |
| **HellaSwag** | Benchmark (Commonsense Reasoning) | — | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) |
| **ARC Reasoning** | Benchmark (Scientific Reasoning) | — | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) |
| **TruthfulQA** | Benchmark (Hallucination/Truth) | — | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) |
| **Winogrande** | Benchmark (Coreference Reasoning) | — | [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) |
| **UltraChat 200k** | Warmup Train (Self-Play SFT) | — | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
| **SPIN_iter0** | Train Prompts (Self-Play) | — | [UCLA-AGI/SPIN_iter0](https://huggingface.co/datasets/UCLA-AGI/SPIN_iter0) |
| **MMLU** | Benchmark (General Knowledge) | — | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| **MMLU-Pro** | Benchmark (Advanced Knowledge) | — | [TIGER-AI-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-AI-Lab/MMLU-Pro) |
| **IFEval** | Benchmark (Strict Instruction Follow) | — | [google/ifeval](https://huggingface.co/datasets/google/ifeval) |

