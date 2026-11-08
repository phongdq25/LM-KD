"""Reference implementation of Generalized JSD loss.

Converted to a module-level function.
"""
import torch
import torch.nn.functional as F


def generalized_jsd_loss(
    student_logits,
    teacher_logits,
    labels=None,
    beta=0.5,
    temperature=1.0,
    reduction="batchmean",
    logits_are_probs=False,
    top_k=None,
    token_clip=None,
):
    """
    Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
    of https://huggingface.co/papers/2306.13649 for the definition.

    Args:
        student_logits:
            Tensor of shape (batch_size, sequence_length, vocab_size)
        teacher_logits:
            Tensor of shape (batch_size, sequence_length, vocab_size)
        labels:
            Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
            loss
        beta:
            Interpolation coefficient between 0 and 1 (default: 0.5)
        temperature:
            Softmax temperature (default: 1.0)
        reduction:
            Specifies the reduction to apply to the output (default: 'batchmean')
        top_k:
            If set, restricts the loss to only the top-k tokens of the teacher distribution. Both student and
            teacher distributions are renormalized over these k tokens before computing JSD. This reduces memory
            and focuses distillation on the teacher's most probable tokens. (default: None = full vocabulary)
        token_clip:
            if set, clips per-token divergence values to this maximum before reduction. Prevents style tokens from dominating the gradient signal over math tokens.

    Returns:
        loss: Scalar tensor with the generalized JSD loss
    """

    if logits_are_probs:
        student_log_probs = torch.log(student_logits.clamp_min(1e-8))
        teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
    else:
        # Apply temperature scaling to logits before computing probabilities
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if top_k is not None and top_k > 0:
            # Restrict to top-k tokens of the teacher distribution and renormalize.
            # Shape: [batch, seq_len, top_k]
            _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        # Compute the log of the mixture distribution
        # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
        beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
            dim=0,
        )

        # Compute KL divergences using F.kl_div
        # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

        # Compute the Generalized Jensen-Shannon Divergence
        jsd = beta * kl_teacher + (1 - beta) * kl_student

    # Per-token clipping: cap each token's divergence value
    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)

    # Masking
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]

    # Apply reduction
    if reduction == "batchmean":
        return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    else:
        return jsd

