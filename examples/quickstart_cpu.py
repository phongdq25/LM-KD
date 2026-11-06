"""Zero-download quickstart example: Trains a tiny student model from a tiny teacher on CPU.

This script demonstrates LM-KD's Python library API (`lmkd.build` and `rt.fit()`)
without requiring any pre-downloaded weights, HuggingFace credentials, or GPU hardware.
Execution finishes in ~2-3 seconds on any modern laptop or workstation.

Usage:
    python examples/quickstart_cpu.py
"""
import tempfile

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

import lmkd


def build_toy_environment():
    """Builds synthetic in-memory GPT-2 student/teacher models and a word-level tokenizer."""
    vocab_words = ["<pad>", "<eos>", "<unk>", "Machine", "learning", "distills", "knowledge", "efficiently"]
    vocab = {word: i for i, word in enumerate(vocab_words)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )

    torch.manual_seed(42)

    def make_gpt2(num_layers: int, hidden_dim: int):
        cfg = GPT2Config(
            vocab_size=len(vocab),
            n_positions=64,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
        return GPT2LMHeadModel(cfg)

    student = make_gpt2(num_layers=1, hidden_dim=16)  # 1-layer student
    teacher = make_gpt2(num_layers=2, hidden_dim=16)  # 2-layer teacher
    return student, teacher, tokenizer


def main():
    print("======================================================================")
    print("LM-KD Quickstart: In-Memory Knowledge Distillation")
    print("======================================================================")

    student, teacher, tokenizer = build_toy_environment()

    # Define minimal in-memory distillation recipe
    with tempfile.TemporaryDirectory() as tmp_dir:
        recipe = {
            "name": "quickstart_demo",
            "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
            "models": {
                "student": {"path": "<injected>"},
                "teacher": {"path": "<injected>"},
            },
            "teacher": {"provider": "hf_frozen", "input": "same"},
            "data": {
                "transforms": [{"type": "tokenize_sft", "use_chat_template": False}],
                "max_length": 32,
            },
            "loss": {
                "aggregator": {"type": "kd_mix", "kd_ratio": 0.5, "ce_alias": "nll"},
                "terms": [
                    {"criterion": "nll", "alias": "nll"},
                    {"criterion": "token_kd", "alias": "kd", "params": {"divergence": "fkl", "temperature": 1.0}},
                ],
            },
            "optim": {"lr": 1.0e-3},
            "train": {
                "max_steps": 5,
                "batch_size": 2,
                "micro_batch_size": 1,
                "output_dir": tmp_dir,
                "log_every": 1,
            },
        }

        # In-memory training dataset
        sample_records = [
            {"prompt": "Machine learning", "response": "distills knowledge"},
            {"prompt": "knowledge", "response": "efficiently"},
            {"prompt": "Machine", "response": "learning efficiently"},
            {"prompt": "distills", "response": "knowledge efficiently"},
        ]

        print("Assembling LM-KD Runtime...")
        rt = lmkd.build(
            recipe,
            student_model=student,
            student_tokenizer=tokenizer,
            teacher_model=teacher,
            teacher_tokenizer=tokenizer,
            train_records=sample_records,
        )

        print("Starting training loop (5 steps)...")
        rt.fit()

        print(f"\nTraining completed! Final Step: {rt.state.step}")
        print(f"Total tokens seen:  {rt.state.tokens_seen}")
        print(f"Total samples seen: {rt.state.samples_seen}")
        print(f"Current epoch:      {rt.state.epoch:.2f}")
        print("======================================================================")


if __name__ == "__main__":
    main()
