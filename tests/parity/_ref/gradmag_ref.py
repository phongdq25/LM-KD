"""Reference implementation for multi-task gradient magnitude balancing."""
import torch


def approximate_gradmag_weights(losses, params, mode, epsilon=1e-8):
    norms = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        sq = sum(g.pow(2).sum() for g in grads if g is not None)
        norms.append(torch.sqrt(sq + 1e-12))
    norms = torch.stack(norms)
    inv_norms = 1.0 / (norms + epsilon)
    if mode == "approx_gradmag_preserve_mag":
        inv_norms = inv_norms / inv_norms.sum()
    return inv_norms, norms
