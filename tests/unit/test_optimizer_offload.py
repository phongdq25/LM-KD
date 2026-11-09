"""Unit tests for OptimizerOffloader (Colocate + Offload)."""

import pytest
import torch

from lmkd.train.engine import MasterWeightOptimizer
from lmkd.train.optimizer_offload import OptimizerOffloader


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_optimizer_offload_reload_parity():
    """Verify that offloading and reloading preserves exact optimizer states and allows continued training."""
    device = torch.device("cuda:0")  # in test context, cuda:0 points to visible device
    w = torch.nn.Parameter(torch.randn(64, 64, device=device))
    opt = torch.optim.AdamW([w], lr=1e-3)

    # Step 1: Populate optimizer state
    loss = (w**2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    exp_avg_orig = opt.state[w]["exp_avg"].clone()
    exp_avg_sq_orig = opt.state[w]["exp_avg_sq"].clone()

    offloader = OptimizerOffloader({"default": opt})
    assert not offloader.is_offloaded

    # Offload
    freed_bytes = offloader.offload()
    assert freed_bytes > 0
    assert offloader.is_offloaded

    # Verify CUDA memory was freed
    for k, v in opt.state[w].items():
        if torch.is_tensor(v) and v.device.type == "cuda":
            assert v.numel() == 0

    # Reload
    offloader.reload()
    assert not offloader.is_offloaded

    # Parity assertions
    assert torch.equal(opt.state[w]["exp_avg"], exp_avg_orig)
    assert torch.equal(opt.state[w]["exp_avg_sq"], exp_avg_sq_orig)

    # Step 2: Ensure optimizer continues to step without error
    loss2 = (w**2).sum()
    loss2.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_optimizer_offload_context_manager():
    """Verify context manager syntax ``with offloader:``."""
    device = torch.device("cuda:0")
    w = torch.nn.Parameter(torch.randn(32, 32, device=device))
    opt = torch.optim.AdamW([w], lr=1e-3)

    loss = (w**2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    offloader = OptimizerOffloader(opt)
    with offloader:
        assert offloader.is_offloaded
        assert opt.state[w]["exp_avg"].numel() == 0
    assert not offloader.is_offloaded
    assert opt.state[w]["exp_avg"].numel() == 32 * 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_optimizer_offload_buffer_reuse():
    """Verify pinned memory buffers are allocated once and reused across cycles."""
    device = torch.device("cuda:0")
    w = torch.nn.Parameter(torch.randn(16, 16, device=device))
    opt = torch.optim.AdamW([w], lr=1e-3)

    loss = (w**2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    offloader = OptimizerOffloader(opt)
    offloader.offload()
    buf_ptrs = {k: buf.data_ptr() for k, buf in offloader.buffers.items()}
    offloader.reload()

    # Offload again
    offloader.offload()
    buf_ptrs_cycle2 = {k: buf.data_ptr() for k, buf in offloader.buffers.items()}
    offloader.reload()

    # Pointers must match: zero reallocation!
    assert buf_ptrs == buf_ptrs_cycle2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_master_weight_optimizer_offload():
    """Verify MasterWeightOptimizer FP32 master copy offloading and restoration."""
    device = torch.device("cuda:0")
    w_bf16 = torch.nn.Parameter(torch.randn(16, 16, device=device, dtype=torch.bfloat16))
    inner_opt = torch.optim.AdamW([w_bf16], lr=1e-3)
    opt = MasterWeightOptimizer(inner_opt)

    loss = (w_bf16.float() ** 2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    master_p = opt.param_map[w_bf16]
    orig_master_data = master_p.data.clone()

    offloader = OptimizerOffloader(opt, offload_master_weights=True)
    with offloader:
        assert master_p.data.numel() == 0

    assert master_p.data.numel() == 16 * 16
    assert torch.equal(master_p.data, orig_master_data)

    # Next step succeeds
    loss = (w_bf16.float() ** 2).sum()
    loss.backward()
    opt.step()
