"""PO utilities ported from the audited human_alignment preference/losses.py.

Source: Awesome-Human-AI-Alignment-main/src/human_alignment/mechanisms/training/preference/losses.py.
All logits and labels here are TARGET aligned through LogitsView. Pair ordering
is chosen block then rejected block (the Criterion converts LM-KD interleaving).
"""
import torch
import torch.nn.functional as F


def preference_loss(method, logits, reference_logits, labels, config, *, state=None):
    """Per-pair risk matching upstream, using target-aligned LogitsView tensors."""
    ratio, fkl, _, mask = token_statistics(logits, reference_logits, labels)
    if len(ratio) % 2:
        raise ValueError("Expected equally sized chosen and rejected batches")
    n = len(ratio) // 2
    state = {} if state is None else state
    margins = ratio.sum(-1)
    chosen, rejected = margins[:n], margins[n:]
    if method == "beta_dpo":
        return beta_dpo_loss(chosen, rejected, state, config.beta, config.beta_dpo_a,
                             config.beta_dpo_filter_ratio, config.beta_dpo_momentum)
    if method == "epsilon_dpo":
        steps = epsilon_dpo_steps(logits, reference_logits, labels, config.epsilon)
        state["last_steps"] = steps
        beta = state.get("beta", config.beta) / (1 + config.epsilon * steps)
        return -F.logsigmoid(beta * (chosen - rejected))
    if method == "tdpo":
        kl = fkl.sum(-1)
        return tdpo_loss(chosen, rejected, kl[:n], kl[n:], config.beta, config.alpha, config.tdpo2)
    if method == "ra_dpo":
        risk = ra_dpo_risk_ratio(logits, reference_logits, labels, config.ra_dpo_confidence,
                                 config.ra_dpo_split_vocabulary)
        penalty = (config.ra_dpo_alpha * (risk[n:] - risk[:n].detach()) if config.ra_dpo_variant == 2
                   else risk[n:] - risk[:n])
        return -F.logsigmoid(config.beta * (chosen - rejected - penalty))
    if method == "adpo":
        if config.adpo_granularity == "static":
            # Source chunks start at its first predicted position. Target position
            # zero is only a LogitsView dummy and must not shift fixed-size chunks.
            ratio, mask = ratio[:, 1:], mask[:, 1:]
        segments = ratio * mask if config.adpo_granularity == "pass" else adpo_segments(
            ratio, mask, config.adpo_granularity, config.adpo_segments,
            config.adpo_static_k, config.adpo_segment_reduce)
        delta = config.beta * (segments[:n] - segments[n:])
        return (-(1 - config.label_smoothing) * F.logsigmoid(delta)
                - config.label_smoothing * F.logsigmoid(-delta)).mean(-1)
    if method == "bpo":
        return _bregman(config.beta * (rejected - chosen), config)
    if method == "tbpo_a":
        values = [pack_responses(v, m) for v, m in
                  ((ratio[:n], mask[:n]), (ratio[n:], mask[n:]), (fkl[:n], mask[:n]), (fkl[n:], mask[n:]))]
        (c, cm), (r, rm), (bc, _), (br, _) = values
        length = min(c.shape[1], r.shape[1])
        valid = cm[:, :length] & rm[:, :length]
        z = config.beta * (r[:, :length] - c[:, :length] + br[:, :length] - bc[:, :length])
        return (_bregman(z.masked_fill(~valid, 0), config) * valid).sum(-1) / valid.sum(-1)
    raise ValueError(f"Unsupported PO method: {method}")


def _bregman(z, config):
    args = (config.generator, config.bregman_lambda, config.bregman_scale, config.log_ratio_clip)
    loss = bregman_loss(z, *args)
    if config.label_smoothing:
        loss = (1 - config.label_smoothing) * loss + config.label_smoothing * bregman_loss(-z, *args)
    return loss

def token_statistics(logits, reference_logits, labels):
    """Return sampled log ratios, forward/reverse KL, and response mask.

    Labels are unshifted; -100 masks prompts/padding. The reference is frozen.
    """
    if logits.shape != reference_logits.shape or logits.shape[:-1] != labels.shape:
        raise ValueError("Policy/reference logits and labels must have matching shapes")
    mask = labels.ne(-100).clone()
    mask[:, 0] = False
    if not mask.any(-1).all():
        raise ValueError("Every sequence must have at least one predicted response token")
    targets = labels.masked_fill(~mask, 0)
    policy = logits.float().log_softmax(-1)
    reference = reference_logits.detach().float().log_softmax(-1)
    delta = policy - reference
    ratios = delta.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    forward_kl = -(reference.exp() * delta).sum(-1)
    reverse_kl = (policy.exp() * delta).sum(-1)
    return tuple(x.masked_fill(~mask, 0) for x in (ratios, forward_kl, reverse_kl)) + (mask,)

def bregman_loss(log_ratio, generator="sba", lam=0.2, scale=4.0, clip=30.0):
    """BPO ratio risk; log_ratio is rejected minus chosen (scaled by beta).

    Additive constants are omitted. LSIF follows h'(R)R-h(R)-h'(1/R),
    correcting the R instead of 1/R typo in the released TBPO LSIF branch.
    """
    if lam <= 0 or scale <= 0 or clip <= 0:
        raise ValueError("lam, scale, and clip must be positive")
    z = log_ratio.clamp(-clip, clip)
    if generator == "logistic":
        return F.softplus(z)
    if generator == "kliep":
        return z.exp() + z
    if generator == "lsif":
        return (2 * z).exp() - 2 * (-z).exp()
    if generator in {"ba", "sba"}:
        risk = ((1 + lam) * z).exp() - (1 + lam) / lam * (-lam * z).exp()
        return risk if generator == "ba" else risk / (scale * (1 + lam))
    raise ValueError(f"Unknown generator: {generator}")

def tdpo_loss(chosen, rejected, chosen_kl, rejected_kl, beta=0.1, alpha=0.5, tdpo2=True):
    correction = (alpha * (rejected_kl - chosen_kl.detach()) if tdpo2
                  else rejected_kl - chosen_kl)
    return F.softplus(-beta * (chosen - rejected - correction))

def pack_responses(values, mask):
    """Left-align response positions without losing the autograd graph."""
    counts = mask.sum(-1)
    length = int(counts.max())
    packed = torch.nn.utils.rnn.pad_sequence(
        [row[m] for row, m in zip(values, mask)], batch_first=True
    )
    valid = torch.arange(length, device=mask.device)[None, :] < counts[:, None]
    return packed, valid

def beta_dpo_loss(chosen, rejected, state, beta=0.1, a=0.6, filter_ratio=0.2, momentum=0.9):
    """beta-DPO (Wu et al., 2024): batch-calibrated beta with beta-guided filtering.

    Follows ``preference_loss``/``update_and_sync_tensor_mean`` in the released
    trainer: the running reward-gap mean and std (``state``, initialized to zero)
    are updated with this batch first; pairs are then sampled without replacement
    with Gaussian weights around the running mean, keeping ``1 - filter_ratio``.
    The returned per-pair losses are rescaled so their mean is the masked mean.
    """
    gap = (chosen - rejected).detach()
    if gap.numel() < 2:
        raise ValueError("beta-DPO needs at least two pairs per micro-batch for its gap statistics")
    if "gap_mean" not in state:
        state["gap_mean"] = gap.new_zeros(())
        state["gap_std"] = gap.new_zeros(())
    state["gap_mean"] = momentum * state["gap_mean"] + (1 - momentum) * gap.mean()
    state["gap_std"] = momentum * state["gap_std"] + (1 - momentum) * gap.std()
    mean, std = state["gap_mean"], state["gap_std"]
    # With policy == reference (the first step) every gap and the std are zero; the
    # released 0/0 weights are NaN, so use their limit exp(0) = 1 (uniform sampling).
    z = (gap - mean) / std if float(std) > 0 else torch.zeros_like(gap)
    weights = torch.exp(-0.5 * z.pow(2))
    if not torch.isfinite(weights).all() or float(weights.sum()) <= 0:
        # Every weight underflowed (gaps far outside the running std); sample uniformly.
        weights = torch.ones_like(gap)
    kept = int(gap.numel() * (1 - filter_ratio))
    if kept < 1:
        raise ValueError("beta_dpo_filter_ratio leaves no pair in this micro-batch")
    index = torch.multinomial(weights, kept, replacement=False)
    mask = torch.zeros_like(gap)
    mask[index] = 1
    beta_used = (beta * (1 + a * (gap[index].mean() - mean))).clamp(min=1e-3).detach()
    state["beta_used"] = float(beta_used)
    losses = -F.logsigmoid(beta_used * (chosen - rejected))
    return losses * mask * (gap.numel() / mask.sum())

def _sequence_logps(logits, targets, mask):
    return (logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask).sum(-1)

def epsilon_dpo_steps(logits, reference_logits, labels, epsilon):
    """Per-pair KL-step direction of epsilon-DPO (Lee et al., 2025): +1, -1, or 0.

    Perturbing policy logits toward ``(1 + eps) * policy - eps * ref`` and
    ``(1 - eps) * policy + eps * ref`` and checking how the chosen-minus-rejected
    log ratio moves, as in the released ``concatenated_forward``.
    """
    mask = labels.ne(-100).clone()
    mask[:, 0] = False
    targets = labels.masked_fill(~mask, 0)
    policy = logits.detach().float()
    reference = reference_logits.detach().float()
    n = len(logits) // 2
    values = []
    for candidate in (policy, (1 + epsilon) * policy - epsilon * reference,
                      (1 - epsilon) * policy + epsilon * reference):
        logps = _sequence_logps(candidate, targets, mask)
        values.append(logps[:n] - logps[n:])
    base, positive, negative = values
    up = (positive > base) & (base > negative)
    down = (negative > base) & (base > positive)
    return up.long() - down.long()

def _linear_quantile(values, q):
    """``torch.quantile(values, q, dim=-1, keepdim=True)`` without its input-size limit."""
    count = values.shape[-1]
    position = q * (count - 1)
    low, high = int(position // 1), min(int(position // 1) + 1, count - 1)
    low_value = values.kthvalue(low + 1, dim=-1, keepdim=True).values
    high_value = values.kthvalue(high + 1, dim=-1, keepdim=True).values
    return low_value + (high_value - low_value) * (position - low)

def ra_dpo_risk_ratio(logits, reference_logits, labels, confidence=0.95, split_vocabulary=True):
    """Ra-DPO (Zhang et al., 2025) sequential risk ratio: summed per-token CVaR.

    Per position, ``d = log p_ref - log p_policy`` over the vocabulary; entries
    above the ``1 - confidence`` quantile are weighted by ``p_ref`` and summed
    (released ``calculate_cvar``). With ``split_vocabulary`` (the released
    ``is_split_risk_ratio: true``) the quantile is taken separately over the two
    halves of the vocabulary axis, exactly as the reference slices ``dim=2``.
    """
    mask = labels.ne(-100).clone()
    mask[:, 0] = False
    policy = logits.float().log_softmax(-1)
    reference = reference_logits.detach().float().log_softmax(-1)
    probabilities = reference.exp()
    difference = reference - policy
    middle = difference.shape[-1] // 2
    parts = ((slice(0, middle), slice(middle, None)) if split_vocabulary else (slice(None),))
    risk = 0
    for part in parts:
        values = difference[..., part]
        threshold = _linear_quantile(values.detach(), 1 - confidence)
        risk = risk + (probabilities[..., part] * values * values.gt(threshold)).sum(-1)
    return (risk * mask).sum(-1)

def adpo_segments(ratio, mask, granularity="adaptive", segments=256, static_k=32, reduce="sum"):
    """ADPO (Oi et al., 2026) segment log ratios, as the released ``_to_segments``.

    ``adaptive`` splits each response's valid tokens into ``segments`` contiguous
    parts (the first ``N % m`` one token longer); ``static`` cuts the padded row
    into chunks of ``static_k`` positions. ``reduce`` sums or averages per segment.
    """
    if granularity == "static":
        width = ratio.shape[1]
        count = -(-width // static_k)
        pad = count * static_k - width
        values = F.pad(ratio * mask, (0, pad)).view(len(ratio), count, static_k)
        valid = F.pad(mask, (0, pad)).view(len(ratio), count, static_k)
        sums = values.sum(-1)
        if reduce == "sum":
            return sums
        counts = valid.sum(-1)
        return torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))
    if granularity != "adaptive":
        raise ValueError(f"Unknown ADPO granularity: {granularity}")
    rows = []
    for row, row_mask in zip(ratio, mask):
        values = row[row_mask]
        total = values.numel()
        sizes = torch.full((segments,), total // segments, dtype=torch.long, device=row.device)
        sizes[: total % segments] += 1
        ids = torch.repeat_interleave(torch.arange(segments, device=row.device), sizes)
        sums = values.new_zeros(segments).index_add(0, ids, values)
        if reduce == "mean":
            sums = torch.where(sizes > 0, sums / sizes.clamp_min(1), torch.zeros_like(sums))
        rows.append(sums)
    return torch.stack(rows)
