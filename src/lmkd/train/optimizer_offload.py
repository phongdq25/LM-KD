"""Optimizer state offloading to pinned CPU memory (Colocate + Offload).

Allows training models and running vLLM rollout / evaluation on the same GPU without OOM
by offloading heavy FP32 optimizer states (and optional FP32 master weights) to pre-allocated
pinned host memory buffers, freeing VRAM for vLLM KV-Cache and activations.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _extract_leaf_optimizers(opt: Any) -> list[torch.optim.Optimizer]:
    """Recursively extract underlying PyTorch optimizers from wrappers."""
    if hasattr(opt, "optimizers") and isinstance(opt.optimizers, dict):
        # MultiOptimizer
        res = []
        for sub in opt.optimizers.values():
            res.extend(_extract_leaf_optimizers(sub))
        return res
    if hasattr(opt, "optimizer") and isinstance(opt.optimizer, torch.optim.Optimizer):
        # MasterWeightOptimizer or single wrapper
        return [opt]
    if isinstance(opt, torch.optim.Optimizer):
        return [opt]
    return []


class OptimizerOffloader:
    """Colocates training and vLLM by offloading optimizer state to pinned host memory.

    Features:
    1. Persistent Pinned Buffers: Allocates pinned host memory buffers once and reuses them
       across epochs/steps for fast PCIe DMA transfers via ``.copy_(..., non_blocking=True)``.
    2. Selective Offloading: Only offloads CUDA optimizer tensors (exp_avg, exp_avg_sq, etc.)
       and FP32 master weights. Low-precision model parameters (e.g. BF16 student weights)
       remain on the GPU so vLLM can sync them in-process without PCIe host round-trips.
    3. Context Manager: ``with offloader:`` offloads before entering and reloads on exit.
    """

    def __init__(
        self,
        optimizers: dict[str, Any] | list[Any] | Any,
        offload_master_weights: bool = True,
    ) -> None:
        if isinstance(optimizers, dict):
            raw_list = list(optimizers.values())
        elif isinstance(optimizers, (list, tuple)):
            raw_list = list(optimizers)
        else:
            raw_list = [optimizers]

        self.optimizers: list[torch.optim.Optimizer] = []
        for o in raw_list:
            self.optimizers.extend(_extract_leaf_optimizers(o))

        self.offload_master_weights = offload_master_weights
        self._is_offloaded = False

        # Persistent pinned memory buffer cache:
        # key: (id(param), tensor_field_name) -> pinned CPU tensor
        self.buffers: dict[tuple[int, str], torch.Tensor] = {}
        self.meta: dict[tuple[int, str], tuple[torch.Size, torch.dtype, torch.device]] = {}

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def offload(self) -> int:
        """Offloads optimizer CUDA tensors to pinned host memory and frees GPU memory.

        Returns:
            Total bytes offloaded to host memory.
        """
        if self._is_offloaded:
            return 0

        total_bytes = 0

        # Phase 1: Asynchronous DMA copy from CUDA to pinned host memory
        for opt in self.optimizers:
            # Check for DeepSpeed optimizer
            if opt.__class__.__name__ == "DeepSpeedOptimizer":
                continue

            # 1. Offload optimizer state tensors (e.g. exp_avg, exp_avg_sq)
            for p, state in opt.state.items():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        key = (id(p), str(k))
                        nbytes = v.numel() * v.element_size()
                        total_bytes += nbytes

                        # Allocate or reuse pinned host memory buffer
                        if (
                            key not in self.buffers
                            or self.buffers[key].shape != v.shape
                            or self.buffers[key].dtype != v.dtype
                        ):
                            self.buffers[key] = torch.empty(
                                v.shape,
                                dtype=v.dtype,
                                device="cpu",
                                pin_memory=True,
                            )
                            self.meta[key] = (v.shape, v.dtype, v.device)

                        self.buffers[key].copy_(v, non_blocking=True)

            # 2. Optionally offload FP32 master weights if wrapped in MasterWeightOptimizer
            if self.offload_master_weights and hasattr(opt, "param_map"):
                param_map = getattr(opt, "param_map", {})
                for master_p in param_map.values():
                    if isinstance(master_p, torch.nn.Parameter) and master_p.is_cuda and master_p.data.numel() > 0:
                        key = (id(master_p), "__master_weight__")
                        v = master_p.data
                        nbytes = v.numel() * v.element_size()
                        total_bytes += nbytes

                        if (
                            key not in self.buffers
                            or self.buffers[key].shape != v.shape
                            or self.buffers[key].dtype != v.dtype
                        ):
                            self.buffers[key] = torch.empty(
                                v.shape,
                                dtype=v.dtype,
                                device="cpu",
                                pin_memory=True,
                            )
                            self.meta[key] = (v.shape, v.dtype, v.device)

                        self.buffers[key].copy_(v, non_blocking=True)

        if total_bytes == 0:
            # No CUDA tensors to offload (e.g. before step 1 or already on CPU)
            self._is_offloaded = True
            return 0

        # Wait for all asynchronous DMA copies to host pinned memory to finish
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Phase 2: Release CUDA memory references
        for opt in self.optimizers:
            if opt.__class__.__name__ == "DeepSpeedOptimizer":
                continue

            for state in opt.state.values():
                for v in state.values():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        # Release backing storage in PyTorch allocator
                        v.data = torch.empty(0, device=v.device, dtype=v.dtype)

            if self.offload_master_weights and hasattr(opt, "param_map"):
                param_map = getattr(opt, "param_map", {})
                for master_p in param_map.values():
                    if isinstance(master_p, torch.nn.Parameter) and master_p.is_cuda:
                        master_p.data = torch.empty(0, device=master_p.device, dtype=master_p.dtype)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_offloaded = True
        logger.debug("Offloaded %.2f MB of optimizer state to host pinned memory.", total_bytes / (1024 * 1024))
        return total_bytes

    def reload(self, device: torch.device | str | None = None) -> None:
        """Reloads offloaded tensors back to CUDA memory from host pinned buffers."""
        if not self._is_offloaded or not self.buffers:
            self._is_offloaded = False
            return

        # Phase 1: Allocate on CUDA and initiate non-blocking copies from pinned host memory
        for opt in self.optimizers:
            if opt.__class__.__name__ == "DeepSpeedOptimizer":
                continue

            # 1. Reload optimizer state tensors
            for p, state in opt.state.items():
                for k in list(state.keys()):
                    key = (id(p), str(k))
                    if key in self.buffers:
                        shape, dtype, orig_device = self.meta[key]
                        target_device = device or orig_device
                        new_tensor = torch.empty(shape, dtype=dtype, device=target_device)
                        new_tensor.copy_(self.buffers[key], non_blocking=True)
                        state[k] = new_tensor

            # 2. Reload FP32 master weights
            if self.offload_master_weights and hasattr(opt, "param_map"):
                param_map = getattr(opt, "param_map", {})
                for master_p in param_map.values():
                    key = (id(master_p), "__master_weight__")
                    if key in self.buffers:
                        shape, dtype, orig_device = self.meta[key]
                        target_device = device or orig_device
                        master_p.data = torch.empty(shape, dtype=dtype, device=target_device)
                        master_p.data.copy_(self.buffers[key], non_blocking=True)

        # Wait for all DMA transfers back to GPU to finish
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._is_offloaded = False

    def __enter__(self) -> OptimizerOffloader:
        self.offload()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.reload()


def get_or_create_offloader(rt: Any) -> OptimizerOffloader | None:
    """Helper to retrieve or lazily create a persistent OptimizerOffloader on Runtime."""
    optimizers = getattr(rt, "optimizers", None)
    if not optimizers:
        return None

    offloader = getattr(rt, "_optimizer_offloader", None)
    if offloader is None:
        offloader = OptimizerOffloader(optimizers)
        rt._optimizer_offloader = offloader
    return offloader
