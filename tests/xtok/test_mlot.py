"""Unit tests for MultiLevelOT criterion (T13)."""

from __future__ import annotations

import pytest
import torch

from lmkd.losses.cross_tokenizer.mlot import MultiLevelOT, mlot_normalize


def test_two_pass_sort_matches_onepass():
    """Chunked 2-pass top-k permutation matches full tensor computation."""
    torch.manual_seed(42)
    N, V, top_k = 1500, 200, 50
    temperature = 2.0
    s_rows = torch.randn(N, V)

    # One pass on full tensor
    zs = mlot_normalize(s_rows)
    ps = torch.softmax(zs / temperature, dim=-1)
    full_sum = ps.sum(dim=0)
    perm_full = torch.argsort(full_sum, descending=True)[:top_k]

    # Two-pass chunked simulation (chunk size 400)
    chunk_size = 400
    accum_sum = torch.zeros(V, dtype=torch.float32)
    for i in range(0, N, chunk_size):
        chunk = s_rows[i : i + chunk_size]
        z_chunk = mlot_normalize(chunk)
        p_chunk = torch.softmax(z_chunk / temperature, dim=-1)
        accum_sum += p_chunk.sum(dim=0)

    perm_chunked = torch.argsort(accum_sum, descending=True)[:top_k]

    assert torch.allclose(full_sum, accum_sum, atol=1e-5)
    assert torch.equal(perm_full, perm_chunked)


def test_f_not_one_raises():
    """MultiLevelOT with f != 1 raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="f=1"):
        MultiLevelOT(MultiLevelOT.Config(f=2))
