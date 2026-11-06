"""Pure reference from IMPACT/code/criterions/ALM.py:14-34,133-289."""
import torch
import torch.nn.functional as F


def _last_index(matrix):
    present = matrix.bool()
    last = present.clone()
    last[:, :-1] = present[:, :-1] & ~present[:, 1:]
    return last.long().argmax(dim=-2)


def _log1mexp(x):
    half = -torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
    return torch.where(x < half, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


def _merge(t_main, s_main, t_space, s_space, count, a, threshold):
    b, c = t_space.shape
    space_mask = torch.exp(t_space) > threshold
    sums = torch.cumsum(space_mask.flip(-1), -1).flip(-1)
    dest = sums.max(-1, keepdim=True).values - sums
    merge = torch.zeros((b * c, c), dtype=a.dtype, device=a.device)
    merge[torch.arange(b * c, device=a.device), dest.reshape(-1).long()] = (count > 0).reshape(-1).to(a.dtype)
    merge = merge.reshape(b, c, c)
    last = _last_index(merge)
    return ((t_main[:, None] @ merge.float()).squeeze(1),
            (s_main[:, None] @ merge.float()).squeeze(1),
            t_space.gather(-1, last), s_space.gather(-1, last),
            (count[:, None].float() @ merge.float()).squeeze(1))


def alm_loss(s_logits, t_logits, s_ids, t_ids, a, b, s_loss_mask, t_loss_mask,
             s_space_mask, t_space_mask, mode, temp=100.0, threshold=0.1, epsilon=1e-5):
    """Reference ALM with dense positional alignment matrices."""
    slp = F.log_softmax(s_logits.float(), -1).clamp(max=0)
    tlp = F.log_softmax(t_logits.float(), -1).clamp(max=0)
    a = a * s_loss_mask[..., None]
    b = b * t_loss_mask[..., None]
    s_main = slp[:, :-1].gather(-1, s_ids[:, 1:, None]).squeeze(-1)
    t_main = tlp[:, :-1].gather(-1, t_ids[:, 1:, None]).squeeze(-1)
    s_main = (s_main[:, None] @ a[:, 1:].float()).squeeze(1).clamp(max=0)
    t_main = (t_main[:, None] @ b[:, 1:].float()).squeeze(1).clamp(max=0)
    s_space = torch.log(slp.exp() @ s_space_mask.float() + 1e-10).clamp(max=0)
    t_space = torch.log(tlp.exp() @ t_space_mask.float() + 1e-10).clamp(max=0)
    s_space = s_space.gather(-1, _last_index(a))
    t_space = t_space.gather(-1, _last_index(b))
    count = b[:, 1:].sum(-2)
    if "merge_by_space_prob" in mode:
        t_main, s_main, t_space, s_space, count = _merge(
            t_main, s_main, t_space, s_space, count, a, threshold)
    valid = (count > 0).float()
    s_main = s_main * valid
    t_space = t_space * valid
    s_space = s_space * valid
    if "append_space" in mode:
        rev = torch.flip(count > 0, [-1])
        last = torch.flip(torch.cumsum(rev.long(), -1) == 1, [-1])
        s_main = s_main + s_space * last.float()
        t_main = t_main + t_space * last.float()
    neg = torch.tensor(-0.7 * torch.finfo(torch.float32).max, device=s_main.device)
    s_full = torch.where(count > 0, s_main, neg)
    t_full = torch.where(count > 0, t_main, neg)
    y = t_full.float() / temp - epsilon
    pred = s_full.float() / temp - epsilon
    bce = -(torch.exp(y) * pred + (-torch.expm1(y) * _log1mexp(pred)))
    numerator = (count > 0).float()
    return (bce * numerator).mean() / (numerator.mean() + epsilon)
