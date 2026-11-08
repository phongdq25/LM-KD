"""Reference implementation of SRA span pooling and loss."""
import torch
import torch.nn.functional as F


def pool(hidden, attention, idx):
    mask = idx >= 0
    safe = idx.clamp_min(0)
    w = attention.sum(1)[:, -1].detach()
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
    b = torch.arange(idx.shape[0])[:, None, None]
    span_w = w[b, safe] * mask
    gathered = hidden[b, safe] * mask[..., None] * span_w[..., None]
    return gathered.sum(2) / span_w.sum(2, keepdim=True).clamp_min(1e-5), span_w.sum(2)


def components(s_pool, t_pool, t_w, projector, s_head, t_head, temperature,
               geom_weight, span_power, ids=None):
    cos = 1 - F.cosine_similarity(projector(s_pool).float(), t_pool.float(), dim=-1, eps=1e-5)
    span = (cos * (t_w / t_w.sum(-1, keepdim=True).clamp_min(1e-9))).sum(-1).mean()
    w = t_w.pow(span_power)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
    pair_w = w[:, :, None] * w[:, None, :]
    pair_w[:, torch.eye(w.shape[1], dtype=torch.bool)] = 0
    pair_w = pair_w / pair_w.sum((1, 2), keepdim=True).clamp_min(1e-5)
    s_sim = F.normalize(s_pool.float(), dim=-1, eps=1e-5)
    t_sim = F.normalize(t_pool.float(), dim=-1, eps=1e-5)
    geom = (((s_sim @ s_sim.transpose(1, 2) - t_sim @ t_sim.transpose(1, 2)).square()) * pair_w).sum()
    geom = geom / s_pool.shape[0] * geom_weight
    s_logits, t_logits = s_head(s_pool), t_head(t_pool)
    if ids is not None:
        s_logits, t_logits = s_logits[..., ids[0]], t_logits[..., ids[1]]
    mask = (s_logits.abs().sum(-1) != 0).float()
    kl = F.kl_div(F.log_softmax(s_logits.float() / temperature, -1),
                  F.softmax(t_logits.float() / temperature, -1), reduction="none").sum(-1)
    return span, geom, (kl * mask).sum() / s_pool.shape[0]
