"""A hand-written training loop in one file, assembled from LM-KD parts.

    python examples/custom_loop.py -c recipes/same_tokenizer/fkl.yaml --set train.max_steps=20

``rt.train_step`` keeps every framework semantic (micro-batching, global loss normalization,
DeepSpeed, grad clipping, schedulers, logging, checkpoints); everything else is plain Python.
"""
import argparse
import sys
from pathlib import Path

import torch.nn.functional as F

import lmkd

sys.path.insert(0, str(Path(__file__).parent))
from my_losses import token_accuracy  # noqa: E402  (also registers example_fkl / example_ce_kd)


def reverse_kl(batch, rt):
    """Tier-0 loss added to the recipe's criteria at every step (mean over the micro-batch)."""
    s = rt.forward(batch)
    t = rt.teacher_forward(batch)
    mask = batch["student.labels"][:, 1:] != -100
    s_logp = F.log_softmax(s.logits[:, :-1][mask], -1)
    t_logp = F.log_softmax(t.logits[:, :-1][mask], -1)
    return {"rkl": (s_logp.exp() * (s_logp - t_logp)).sum(-1).mean()}


def main(recipe, overrides=None, rkl_weight: float = 0.1, eval_every: int = 10):
    rt = lmkd.build(recipe, overrides=overrides, evaluators=[token_accuracy])

    def loss_fn(batch, rt_):
        return {k: rkl_weight * v for k, v in reverse_kl(batch, rt_).items()}

    rt.begin()
    while not rt.done:
        out = rt.train_step(loss_fn=loss_fn)          # one optimizer step
        if rt.state.step % eval_every == 0 or rt.done:
            print(f"step {rt.state.step} loss {float(out.metrics['loss/total']):.4f} eval {rt.evaluate()}")
    rt.end()                                          # final checkpoint (checkpoint callback)
    return rt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    main(a.config, a.set)
