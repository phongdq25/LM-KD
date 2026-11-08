"""Independent reference for ResidualKD utilities."""
import math

import torch
import torch.nn.functional as F


def residual_core_ref(hs, ht, sa, ta, as_, labels_pos, teacher_logits, cross, threshold=0.5):
    hsa, hta = sa(hs), ta.encode(ht)
    q = hsa / hsa.std(-1, keepdim=True).clamp(min=1e-5)
    k = hta / hta.std(-1, keepdim=True).clamp(min=1e-5)
    a = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(hsa.shape[-1]), -1)
    proj = as_(a @ hta)
    resp = labels_pos.ne(-100)
    if cross:
        prob = teacher_logits.float().softmax(-1)
        entropy = -(prob * prob.clamp(min=1e-9).log()).sum(-1) / math.log(prob.shape[-1])
        mask = ((a.float() @ entropy.unsqueeze(-1)).squeeze(-1) > threshold) & resp
    else:
        mask = teacher_logits.argmax(-1).ne(labels_pos) & resp
    ratio = hs.float().norm(dim=-1) / proj.float().norm(dim=-1).clamp(min=1e-5)
    beta = (math.sqrt(hs.shape[-1] / hsa.shape[-1]) *
            ((ratio * resp).sum(-1) / resp.sum(-1).clamp(min=1)).mean()).detach().clamp(0.05, 10)
    return hs - beta * proj * mask.unsqueeze(-1), beta, mask


def pretrain_ref(hidden, labels, projector, head):
    """Pretrain wrapper: reconstruct hidden, apply detached head, shift CE."""
    _, reconstructed = projector(hidden.detach())
    logits = F.linear(reconstructed[:, :-1], head.weight.detach())
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
