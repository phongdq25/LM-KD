"""Peak allocation comparison for T03; requires a large CUDA GPU."""

import pytest
import torch
from torch import nn

from lmkd.losses.functional.divergences import reverse_kl
from lmkd.models import logits_view as lv


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_peak_memory_halved(monkeypatch):
    if torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3:
        pytest.skip("benchmark requires at least 40 GiB CUDA memory")

    def measure(efficient):
        monkeypatch.setattr(lv, "LOGITS_VIEW_EFFICIENT", efficient)
        torch.cuda.empty_cache()
        h = torch.randn(2, 2048, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        head = nn.Linear(1024, 151936, bias=False, device="cuda", dtype=torch.bfloat16)
        teacher_h = torch.randn_like(h)
        teacher_head = nn.Linear(1024, 151936, bias=False, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(2, 2048, device="cuda")
        mask[:, 0] = 0
        torch.cuda.reset_peak_memory_stats()
        lv.chunked_token_divergence(
            lv.LogitsView(h, head), lv.LogitsView(teacher_h, teacher_head), reverse_kl, mask, 128
        ).sum().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    naive = measure(False)
    efficient = measure(True)
    assert efficient < 0.5 * naive, (efficient, naive)
