"""Small explicit references from IMPACT MCW_KD.py:118-257, with documented fixes."""
import torch
import torch.nn.functional as F

from tests.parity._ref.etp_ref import etp_ref


def context_ref(seq, window):
    return torch.stack([seq[max(0, i - window):min(len(seq), i + window + 1)].mean(0)
                        for i in range(len(seq))])


def logits_ref(slog, tlog, hs, ht, salience, projector, weights, tau=2.0, progress=0.0, k=300):
    k = min(k, slog.shape[-1], tlog.shape[-1])
    s = slog.topk(k, -1).values
    t = tlog.topk(k, -1).values
    ratio = 0.1 + 0.9 * progress
    sp = F.softmax(s / tau, -1)
    tp = F.softmax(((1 - ratio) * s + ratio * t) / tau, -1)
    c2 = torch.cdist(tp, sp)
    c4 = (tp[:, None] * torch.log(tp[:, None] / (sp[None] + 1e-7))).sum(-1)
    ss = torch.sigmoid(salience(hs)).squeeze(-1)
    st = torch.sigmoid(salience(projector(ht))).squeeze(-1)
    cs = (st[:, None] - ss[None]).abs()
    return etp_ref(weights[0] * c2 + weights[1] * c4 + weights[2] * cs)[0]


def hidden_ref(s, t, pairs_s2t, costs_s2t, pairs_t2s, costs_t2s, weights, window=4):
    n, m = len(s), len(t)
    cst = s.new_zeros(n, m)
    cts = s.new_zeros(m, n)
    for (i, j), cost in zip(pairs_s2t, costs_s2t):
        if i < n and j < m:
            cst[i, j] = cost
    for (i, j), cost in zip(pairs_t2s, costs_t2s):
        if i < m and j < n:
            cts[i, j] = cost
    c2 = (cst.T + cts) / 2
    cs, ct = context_ref(s, window), context_ref(t, window)
    c5 = torch.cdist(ct, cs)
    c5 = c5 / (c5.max() + 1e-7)
    ns = cs / (cs.norm(dim=-1, keepdim=True) + 1e-7)
    nt = ct / (ct.norm(dim=-1, keepdim=True) + 1e-7)
    c6 = 1 - nt @ ns.T
    return etp_ref(sum(w * c for w, c in zip(weights, (c2, c5, c6))))[0]
