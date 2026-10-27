"""Porting a baseline loss into LM-KD in minutes: tier 0 and tier 1 (README "Adding a new method").

Load this file with ``imports: [path/to/my_losses.py]`` in a recipe, ``--plugins my_losses.py``
on the CLI, or ``lmkd.build(..., imports=["my_losses.py"])``. No list to edit, no package to install.
"""
import torch
import torch.nn.functional as F

import lmkd


# ----------------------------------------------------------------------------- tier 1
# A function on HF-style outputs, registered as a normal criterion: composable in `loss.terms`
# with any built-in criterion, weight and schedule. Hyperparameters = keyword args (strict Config).
@lmkd.register_loss("example_fkl", teacher=True, reduction="token_sum",
                    source="white-box-distillation/same-tokenizer/distillm/losses.py:4")
def example_fkl(s, t, batch, temperature: float = 1.0):
    """Forward KL in cross-entropy form, -sum_v p_T log q_S, written like the original code."""
    s_logits = s.logits[:, :-1] / temperature           # HF convention: row t predicts token t+1
    t_logits = t.logits[:, :-1] / temperature
    mask = (batch["student.labels"][:, 1:] != -100).float()
    per_token = -(F.softmax(t_logits, -1) * F.log_softmax(s_logits, -1)).sum(-1)
    return (per_token * mask).sum()                     # token_sum: framework divides by #tokens of the step


# ----------------------------------------------------------------------------- tier 0
# Like transformers.Trainer.compute_loss: run the models yourself, return the loss.
# Recipe: algorithm: {type: compute_loss, fn: example_ce_kd, kwargs: {kd_weight: 0.5}}
@lmkd.compute_loss("example_ce_kd")
def example_ce_kd(batch, rt, kd_weight: float = 0.5, temperature: float = 1.0):
    """CE + weighted forward KL; each value is a mean over the micro-batch."""
    s = rt.forward(batch)                               # student (grad, DeepSpeed-safe)
    t = rt.teacher_forward(batch)                       # teacher (no grad)
    labels = batch["student.labels"][:, 1:]
    mask = labels != -100
    ce = F.cross_entropy(s.logits[:, :-1][mask], labels[mask])
    t_p = F.softmax(t.logits[:, :-1][mask] / temperature, -1)
    s_logp = F.log_softmax(s.logits[:, :-1][mask] / temperature, -1)
    kd = -(t_p * s_logp).sum(-1).mean()
    return {"ce": ce, "kd": kd_weight * kd}             # logged as loss/loss_fn/ce, loss/loss_fn/kd


@torch.no_grad()
def token_accuracy(rt, n: int = 8) -> dict[str, float]:
    """Custom evaluation callable for ``lmkd.build(..., evaluators=[token_accuracy])``."""
    samples = [rt.train_dataset[i] for i in range(min(n, len(rt.train_dataset)))]
    batch = rt.collator(samples).to(rt.engine.device)
    out = rt.forward(batch, grad=False)                 # through the engine wrapper (safe under ZeRO-3)
    pred = out.logits[:, :-1].argmax(-1)
    labels = batch["student.labels"][:, 1:]
    m = labels != -100
    return {"my_eval/token_acc": float((pred[m] == labels[m]).float().mean())}
