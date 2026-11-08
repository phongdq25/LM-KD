import pytest
import torch

from lmkd.losses.functional.divergences import get_divergence
from tests.parity._ref import distillm_ref as distillm


@pytest.mark.parity
@pytest.mark.parametrize(
    "name,ref_fn,kw",
    [
        ("fkl", distillm.forward_kl, {}),
        ("rkl", distillm.reverse_kl, {}),
        ("symkl", distillm.symmetric_kl, {"lam": 0.9}),
        ("jsd", distillm.js_distance, {"lam": 0.9}),
        ("tvd", distillm.tv_distance, {}),
        ("sfkl", distillm.skewed_forward_kl, {"alpha": 0.1}),
        ("srkl", distillm.skewed_reverse_kl, {"alpha": 0.1}),
        ("csd", distillm.csd, {}),
    ],
)
def test_distillm_parity(name, ref_fn, kw):
    torch.manual_seed(42)
    B, T, V = 2, 9, 37
    s = torch.randn(B, T, V)
    t = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :3] = -100  # mask prompt tokens

    no_model_batch = {"label": labels}
    mask = (labels != -100).float()

    # DistiLLM uses `lam` argument for sfkl/srkl in ref code
    ref_kw = dict(kw)
    if name in ("sfkl", "srkl"):
        ref_kw = {"lam": kw["alpha"]}

    ref_val = ref_fn(s, t, no_model_batch, **ref_kw)

    our_fn = get_divergence(name, **kw)
    our_per_tok = our_fn(s, t)
    our_val = (our_per_tok * mask).sum() / mask.sum()

    assert torch.allclose(our_val, ref_val, atol=1e-5), f"{name}: our={our_val.item()} != ref={ref_val.item()}"

    if name == "csd":
        s_ref = s.clone().detach().requires_grad_(True)
        ref_loss = distillm.csd(s_ref, t, no_model_batch)
        ref_grad = torch.autograd.grad(ref_loss, s_ref)[0]

        s_our = s.clone().detach().requires_grad_(True)
        our_per_tok_grad = get_divergence("csd")(s_our, t)
        our_loss = (our_per_tok_grad * mask).sum() / mask.sum()
        our_grad = torch.autograd.grad(our_loss, s_our)[0]

        assert torch.allclose(our_loss, ref_loss, atol=1e-5)
        assert torch.allclose(our_grad, ref_grad, atol=1e-5)
