"""Recompute linear-head chunks in backward, retaining no vocabulary-sized logits.

Computes chunk_fn(h @ W.T + b) in fp32.
Only first-order gradients are supported. The callback must be deterministic.
Inputs must be full parameters: LogitsView routes ZeRO-3 and wrapped heads through their
module forward instead. This Function does not gather parameters or reduce gradients.
"""
import torch
from torch.autograd.function import once_differentiable


def _head(h, weight, bias, temperature, vocab_limit):
    with torch.autocast(device_type=h.device.type, enabled=False):
        if h.device.type == "cuda" and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        x = h.float() @ weight.float().T
        if bias is not None:
            x = x + bias.float()
        if vocab_limit is not None:
            x = x[..., :vocab_limit]
        return x / temperature


class ChunkedHeadFunction(torch.autograd.Function):
    """Apply a pure per-token callback and recompute its linear projection per chunk."""

    @staticmethod
    def forward(ctx, h, weight, bias, fn, chunk_size, temperature, vocab_limit, skip_mask=None):
        if chunk_size <= 0:
            raise ValueError('chunk_size must be positive')
        out = h.new_zeros(h.shape[:2], dtype=torch.float32)
        for s in range(0, h.shape[1], chunk_size):
            e = min(h.shape[1], s + chunk_size)
            if skip_mask is not None and not bool(skip_mask[:, s:e].any()):
                continue
            out[:, s:e] = fn(_head(h[:, s:e], weight, bias, temperature, vocab_limit), s, e)
        ctx.save_for_backward(h, weight, bias, skip_mask)
        ctx.fn, ctx.size, ctx.temp, ctx.limit = fn, chunk_size, temperature, vocab_limit
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        h, weight, bias, mask = ctx.saved_tensors
        gh = torch.zeros_like(h) if ctx.needs_input_grad[0] else None
        gw = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        gb = torch.zeros_like(bias, dtype=torch.float32) if bias is not None and ctx.needs_input_grad[2] else None
        with torch.autocast(device_type=h.device.type, enabled=False):
            for s in range(0, h.shape[1], ctx.size):
                e = min(h.shape[1], s + ctx.size)
                if mask is not None and not bool(mask[:, s:e].any()):
                    continue
                with torch.enable_grad():
                    raw = _head(h[:, s:e].detach(), weight.detach(), None if bias is None else bias.detach(), 1., None).requires_grad_(True)
                    x = raw if ctx.limit is None else raw[..., :ctx.limit]
                    val = ctx.fn(x / ctx.temp, s, e)
                    g = torch.autograd.grad(val, raw, grad_outputs=grad_out[:, s:e].float())[0]
                if gh is not None:
                    if weight.dtype in (torch.float16, torch.bfloat16):
                        gh[:, s:e] = (g.to(weight.dtype) @ weight).to(h.dtype)
                    else:
                        gh[:, s:e] = (g @ weight.float()).to(h.dtype)
                if gw is not None:
                    if h.dtype in (torch.float16, torch.bfloat16):
                        gw += (g.to(h.dtype).flatten(0, 1).T @ h[:, s:e].flatten(0, 1)).to(gw.dtype)
                    else:
                        gw += g.flatten(0, 1).T @ h[:, s:e].float().flatten(0, 1)
                if gb is not None:
                    gb += g.sum((0, 1))
        return gh, None if gw is None else gw.to(weight.dtype), None if gb is None else gb.to(bias.dtype), None, None, None, None, None
