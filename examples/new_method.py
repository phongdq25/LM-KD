"""Offline CPU experiment: define a masked KD loss, compose it with NLL and train.

Run: .venv/bin/python examples/new_method.py --steps 2 --output /tmp/lmkd-new-method
"""
import argparse

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

import lmkd
from lmkd import Inputs, TokenLoss


@lmkd.loss("example_confident_kd", teacher=True, chunk_size=16, supports_packing=True)
def confident_kd(s: Inputs, t: Inputs, temperature: float = 2.0,
                 confidence_threshold: float = 0.0) -> TokenLoss:
    """Forward KL on confident teacher targets, divided by all supervised tokens."""
    student_lp = (s.logits / temperature).log_softmax(-1)
    teacher_lp = (t.logits / temperature).log_softmax(-1)
    teacher_p = teacher_lp.exp()
    values = (teacher_p * (teacher_lp - student_lp)).sum(-1) * temperature**2
    selected = teacher_p.max(-1).values >= confidence_threshold
    return TokenLoss(values, selected)


def tiny_parts():
    """Random GPT-2 models and a word tokenizer, constructed locally without downloads."""
    vocab = {word: i for i, word in enumerate(["<pad>", "<eos>", "<unk>", "Q", "A", "one", "two", "three"])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                        eos_token="<eos>", unk_token="<unk>")
    torch.manual_seed(7)

    def model(layers):
        cfg = GPT2Config(vocab_size=len(vocab), n_positions=32, n_embd=16, n_layer=layers, n_head=2,
                         bos_token_id=1, eos_token_id=1, pad_token_id=0,
                         resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        return GPT2LMHeadModel(cfg)

    return model(1), model(2), tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", default="outputs/example_new_method")
    args = parser.parse_args()
    student, teacher, tokenizer = tiny_parts()
    recipe = {
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "models": {"student": {"path": "<injected>"}, "teacher": {"path": "<injected>"}},
        "teacher": {"provider": "hf_frozen", "input": "same"},
        "data": {"transforms": [{"type": "tokenize_sft", "use_chat_template": False}], "max_length": 32},
        "loss": {"terms": [{"criterion": "nll"},
                           {"criterion": "example_confident_kd", "weight": 0.5,
                            "params": {"temperature": 2.0, "confidence_threshold": 0.0}}]},
        "optim": {"lr": 0.001},
        "train": {"max_steps": args.steps, "batch_size": 2, "micro_batch_size": 1,
                  "output_dir": args.output, "log_every": 1},
    }
    rows = [{"prompt": "Q one", "response": "A two three"}, {"prompt": "Q two", "response": "A three one"}]
    rt = lmkd.build(recipe, student_model=student, student_tokenizer=tokenizer,
                    teacher_model=teacher, teacher_tokenizer=tokenizer, train_records=rows)
    rt.fit()
    print(f"Completed {rt.state.step} steps; checkpoint: {args.output}/checkpoint-final")


if __name__ == "__main__":
    main()
