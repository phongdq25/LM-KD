"""PyTorch FSDP2 sharding and full checkpoint conversion."""
from typing import Any

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


def shard(module: torch.nn.Module) -> None:
    """Keep full parameters through loss evaluation: LM heads/aux run after backbone forward."""
    fully_shard(module, reshard_after_forward=False)


def full_model_state(model: torch.nn.Module) -> dict:
    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def load_model_state(model: torch.nn.Module, state: dict) -> Any:
    return set_model_state_dict(model, state, options=StateDictOptions(full_state_dict=True, strict=False))


def full_optimizer_state(model: torch.nn.Module, optimizer: Any) -> dict:
    return get_optimizer_state_dict(model, optimizer,
                                    options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def load_optimizer_state(model: torch.nn.Module, optimizer: Any, state: dict) -> None:
    # Stateless SGD has no tensors to distribute. PyTorch's full-state loader
    # otherwise cannot infer a device and raises "Expected device to be set".
    set_optimizer_state_dict(model, optimizer, state,
                             options=StateDictOptions(full_state_dict=bool(state.get("state"))))


def full_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.full_tensor() if isinstance(value, DTensor) else value
