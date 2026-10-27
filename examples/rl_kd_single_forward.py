"""Research Example: Joint RL / Preference Alignment + Knowledge Distillation with a SINGLE forward pass.

This example demonstrates how to combine RL (GRPO/PPO) and Preference Optimization (DPO)
with Knowledge Distillation using exactly ONE forward pass on the student model.

Modes covered:
1. Native LM-KD Recipe/Runtime mode:
   - Combines `ppo_clip` (GRPO/PPO policy loss) + `token_kd` (Teacher KD loss) in `loss.terms`.
   - LM-KD's ForwardRunner automatically fuses feature requests into a SINGLE forward pass.
   - Custom advantage-guided KD via `@lmkd.loss`.
2. Pure PyTorch Functional mode:
   - For researchers who want full control without framework abstractions (e.g. inside a custom HF Trainer).
   - Uses `lmkd.losses.functional` primitives: `grpo_advantage`, `grpo_loss`, `reverse_kl`, `dpo_loss`.
   - Explicitly shows: 1 rollout -> 1 student forward -> joint loss -> 1 backward.

Run:
    .venv/bin/python examples/rl_kd_single_forward.py --mode all
"""
from __future__ import annotations

import argparse
import tempfile
from typing import Any

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

import lmkd
from lmkd import Inputs, TokenLoss
from lmkd.core.registry import REWARDS
from lmkd.losses.functional import (
    dpo_loss,
    grpo_advantage,
    grpo_loss,
    reverse_kl,
)
from lmkd.rl.rewards import Verifier


# -----------------------------------------------------------------------------
# 1. Custom Verifier / Reward Function for Rollout
# -----------------------------------------------------------------------------
@REWARDS.register("simple_accuracy_reward", force=True)
class SimpleAccuracyReward(Verifier):
    """Toy reward: gives +1.0 for even-indexed responses, -1.0 for odd-indexed responses."""

    def __call__(self, prompts: list[str], responses: list[str], meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"score": 1.0 if i % 2 == 0 else -1.0, "acc": 1.0 if i % 2 == 0 else 0.0, "feedback": ""}
            for i in range(len(responses))
        ]


# -----------------------------------------------------------------------------
# 2. Custom Advantage-Guided KD Loss (Deep Fusion)
# -----------------------------------------------------------------------------
@lmkd.loss("advantage_guided_kd", teacher=True, chunked=False)
def advantage_guided_kd(s: Inputs, t: Inputs, temperature: float = 2.0) -> TokenLoss:
    """Novel method example: Only distill from teacher when rollout response has positive advantage."""
    assert s.logits is not None and t.logits is not None
    s_lp = (s.logits / temperature).log_softmax(-1)
    t_lp = (t.logits / temperature).log_softmax(-1)
    t_p = t_lp.exp()
    token_kl = (t_p * (t_lp - s_lp)).sum(-1) * (temperature**2)

    # Access rollout advantage from the shared batch
    adv = s.batch.get("rollout.advantage", torch.ones_like(token_kl)) if s.batch is not None else torch.ones_like(token_kl)
    # Modulation: scale KD loss by positive advantage (clamp negative advantage to 0)
    pos_adv = torch.clamp(adv, min=0.0)
    weighted_kl = token_kl * pos_adv
    return TokenLoss(weighted_kl)


# -----------------------------------------------------------------------------
# Helper: Build Tiny Models for Offline CPU Run
# -----------------------------------------------------------------------------
def make_tiny_parts() -> tuple[GPT2LMHeadModel, GPT2LMHeadModel, PreTrainedTokenizerFast]:
    """Create tiny random GPT-2 student/teacher and word tokenizer without network downloads."""
    vocab = {word: i for i, word in enumerate(["<pad>", "<eos>", "<unk>", "Q", "A", "w0", "w1", "w2", "+", "="])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )

    def create_model(layers: int, seed: int) -> GPT2LMHeadModel:
        torch.manual_seed(seed)
        cfg = GPT2Config(
            vocab_size=len(vocab),
            n_positions=32,
            n_embd=16,
            n_layer=layers,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        return GPT2LMHeadModel(cfg)

    return create_model(1, seed=42), create_model(2, seed=100), tokenizer


# -----------------------------------------------------------------------------
# Demo 1: Native LM-KD GRPO + KD with SINGLE Forward
# -----------------------------------------------------------------------------
def demo_native_grpo_kd(student: GPT2LMHeadModel, teacher: GPT2LMHeadModel, tok: PreTrainedTokenizerFast) -> None:
    """Run GRPO + KD through LM-KD Native Runtime.

    ForwardRunner executes the student forward pass EXACTLY ONCE, feeding the resulting
    LogitsView to both `ppo_clip` (policy gradient) and `advantage_guided_kd` (teacher KD).
    """
    print("\n" + "=" * 70)
    print("1. Native LM-KD Mode: GRPO Policy Gradient + Advantage-Guided KD")
    print("=" * 70)

    recipe = {
        "engine": {"type": "local", "device": "cpu", "mixed_precision": "no"},
        "models": {"student": {"path": "<injected>"}, "teacher": {"path": "<injected>"}},
        "teacher": {"provider": "hf_frozen", "input": "same"},
        "data": {
            "transforms": [
                {"type": "tokenize_prompt", "template": "{prompt}", "use_chat_template": False, "keep_fields": ["response"]}
            ],
            "max_length": 32,
        },
        "producer": {
            "type": "rollout_group",
            "n": 2,  # 2 completions per prompt
            "sampling": {"max_new_tokens": 4},
            "verifier": {"type": "simple_accuracy_reward"},
        },
        "rollout": {"type": "hf"},
        "algorithm": {"type": "rl"},
        "step": {"type": "ppo_epochs", "epochs": 1, "mini_batches": 1},
        "loss": {
            "terms": [
                {"criterion": "ppo_clip", "weight": 1.0, "alias": "grpo_pg"},
                {"criterion": "advantage_guided_kd", "weight": 0.5, "alias": "adv_kd"},
            ]
        },
        "optim": {"lr": 1e-3},
        "train": {"max_steps": 2, "batch_size": 2, "micro_batch_size": 1, "output_dir": tempfile.mkdtemp()},
    }

    rows = [{"prompt": "Q w0", "response": "A w1"}, {"prompt": "Q w1", "response": "A w2"}]
    rt = lmkd.build(
        recipe,
        student_model=student,
        student_tokenizer=tok,
        teacher_model=teacher,
        teacher_tokenizer=tok,
        train_records=rows,
    )
    rt.begin()
    result = rt.train_step()
    print("Step 1 metrics:")
    for k, v in sorted(result.metrics.items()):
        if "loss" in k or "reward" in k:
            print(f"  {k:28s}: {float(v):.6f}")
    rt.end()
    print("[Native Mode] Completed step with 1 student forward pass across all criteria!")


# -----------------------------------------------------------------------------
# Demo 2: Pure PyTorch / Custom Loop Mode with SINGLE Forward
# -----------------------------------------------------------------------------
def demo_functional_pure_pytorch(student: GPT2LMHeadModel, teacher: GPT2LMHeadModel, tok: PreTrainedTokenizerFast) -> None:
    """Run GRPO + KD in pure PyTorch using `lmkd.losses.functional` primitives.

    Demonstrates how to write a custom loop or Trainer that forwards the student
    EXACTLY ONCE and computes both GRPO loss and KD loss without any framework wrapper.
    """
    print("\n" + "=" * 70)
    print("2. Pure PyTorch Mode: 1 Forward -> GRPO Loss + KD Loss -> 1 Backward")
    print("=" * 70)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Simulate a generated batch of rollouts (2 prompts x 2 completions = 4 sequences)
    # [B, T] with prompt length = 2, completion length = 3
    input_ids = torch.tensor(
        [
            [3, 5, 4, 6, 1],  # prompt: Q w0, completion: A w1 <eos>
            [3, 5, 4, 7, 1],  # prompt: Q w0, completion: A w2 <eos>
            [3, 6, 4, 7, 1],  # prompt: Q w1, completion: A w2 <eos>
            [3, 6, 4, 5, 1],  # prompt: Q w1, completion: A w0 <eos>
        ]
    )
    # Target-aligned labels: prompt is masked with -100, index 0 is always -100
    labels = torch.tensor(
        [
            [-100, -100, 4, 6, 1],
            [-100, -100, 4, 7, 1],
            [-100, -100, 4, 7, 1],
            [-100, -100, 4, 5, 1],
        ]
    )
    rewards = torch.tensor([1.0, -1.0, 1.0, -1.0])
    group_uids = ["group_0", "group_0", "group_1", "group_1"]

    # 1. Compute group-relative advantage (GRPO)
    advantages = grpo_advantage(rewards, group_uids, norm_by_std=True)
    # Expand to token level [B, 1]
    adv_tokens = advantages.unsqueeze(-1).expand_as(labels)

    # 2. STUDENT FORWARD PASS: EXACTLY ONCE
    optimizer.zero_grad()
    student_outputs = student(input_ids)
    # Target-aligned student logits: position t predicts token at t
    # HF model logits[:, t] predicts input_ids[:, t+1], so shift right by 1
    shift_student_logits = torch.cat(
        [student_outputs.logits.new_zeros(input_ids.size(0), 1, student_outputs.logits.size(-1)), student_outputs.logits[:, :-1]],
        dim=1,
    )

    # 3. TEACHER FORWARD PASS: Once without gradients
    with torch.no_grad():
        teacher_outputs = teacher(input_ids)
        shift_teacher_logits = torch.cat(
            [
                teacher_outputs.logits.new_zeros(input_ids.size(0), 1, teacher_outputs.logits.size(-1)),
                teacher_outputs.logits[:, :-1],
            ],
            dim=1,
        )

    # 4. COMPUTE LOSSES FROM THE SAME FORWARD PASS
    old_logp = shift_student_logits.detach().log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)

    # Term A: GRPO Policy Gradient Loss
    loss_grpo = grpo_loss(shift_student_logits, old_logp, labels, adv_tokens, clip_low=0.2, clip_high=0.28)

    # Term B: Knowledge Distillation Loss (Reverse KL)
    loss_kd = reverse_kl(shift_student_logits, shift_teacher_logits)
    mask = labels.ne(-100)
    loss_kd_mean = (loss_kd * mask).sum() / mask.sum().clamp_min(1)

    # Fused Total Loss
    total_loss = loss_grpo + 0.5 * loss_kd_mean

    # 5. BACKWARD PASS: EXACTLY ONCE
    total_loss.backward()
    optimizer.step()

    print(f"GRPO Loss : {loss_grpo.item():.6f}")
    print(f"KD Loss   : {loss_kd_mean.item():.6f}")
    print(f"Total Loss: {total_loss.item():.6f}")
    print("[Functional Mode] Gradients successfully updated student parameters!")


# -----------------------------------------------------------------------------
# Demo 3: Joint DPO + KD with SINGLE Forward
# -----------------------------------------------------------------------------
def demo_dpo_kd_single_forward(student: GPT2LMHeadModel, teacher: GPT2LMHeadModel, tok: PreTrainedTokenizerFast) -> None:
    """Run DPO + KD on chosen completion with a single forward pass."""
    print("\n" + "=" * 70)
    print("3. Joint DPO + KD Mode: Preference Loss + Distillation on Chosen")
    print("=" * 70)

    # Pairwise preference batch: Row 0 is chosen (yw), Row 1 is rejected (yl)
    # [2, T]
    input_ids = torch.tensor(
        [
            [3, 5, 4, 6, 1],  # Chosen: prompt Q w0 -> completion A w1 <eos>
            [3, 5, 4, 7, 1],  # Rejected: prompt Q w0 -> completion A w2 <eos>
        ]
    )
    labels = torch.tensor(
        [
            [-100, -100, 4, 6, 1],
            [-100, -100, 4, 7, 1],
        ]
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    optimizer.zero_grad()

    # Forward Student ONCE
    s_out = student(input_ids)
    s_logits = torch.cat(
        [s_out.logits.new_zeros(2, 1, s_out.logits.size(-1)), s_out.logits[:, :-1]],
        dim=1,
    )

    # Forward Teacher (frozen reference)
    with torch.no_grad():
        t_out = teacher(input_ids)
        t_logits = torch.cat(
            [t_out.logits.new_zeros(2, 1, t_out.logits.size(-1)), t_out.logits[:, :-1]],
            dim=1,
        )

    # Compute sequence log-probabilities using lmkd sequence utility
    # logp on completion tokens
    mask = labels.ne(-100)
    s_logp = (s_logits.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1) * mask).sum(-1)
    with torch.no_grad():
        t_logp = (t_logits.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1) * mask).sum(-1)

    # DPO implicit reward: log(pi_student) - log(pi_reference)
    log_ratio = s_logp - t_logp
    lr_chosen, lr_rejected = log_ratio[0:1], log_ratio[1:2]

    # Term A: DPO Loss
    loss_dpo_val = dpo_loss(lr_chosen, lr_rejected, beta=0.1)

    # Term B: Distill teacher distribution specifically on the CHOSEN response (Row 0)
    kd_chosen = reverse_kl(s_logits[0:1], t_logits[0:1])
    loss_kd_chosen = (kd_chosen * mask[0:1]).sum() / mask[0:1].sum().clamp_min(1)

    # Joint Objective
    total_loss = loss_dpo_val + 0.5 * loss_kd_chosen
    total_loss.backward()
    optimizer.step()

    print(f"DPO Loss  : {loss_dpo_val.item():.6f}")
    print(f"KD Chosen : {loss_kd_chosen.item():.6f}")
    print(f"Total Loss: {total_loss.item():.6f}")
    print("[DPO+KD Mode] Completed joint preference alignment + distillation!")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["native", "functional", "dpo", "all"],
        default="all",
        help="Which single-forward demo to execute.",
    )
    args = parser.parse_args()

    student, teacher, tokenizer = make_tiny_parts()

    if args.mode in ("native", "all"):
        demo_native_grpo_kd(student, teacher, tokenizer)
    if args.mode in ("functional", "all"):
        demo_functional_pure_pytorch(student, teacher, tokenizer)
    if args.mode in ("dpo", "all"):
        demo_dpo_kd_single_forward(student, teacher, tokenizer)


if __name__ == "__main__":
    main()
