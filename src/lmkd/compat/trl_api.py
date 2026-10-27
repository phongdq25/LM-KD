"""trl glue (see ``lmkd.compat``): the ``VLLMClient`` used by the ``vllm_server`` rollout backend."""
from __future__ import annotations

from typing import Any


def get_vllm_client(
    host: str = "127.0.0.1",
    server_port: int = 8000,
    group_port: int = 51216,
    connection_timeout: float = 600.0,
) -> Any:
    """Import and instantiate TRL's VLLMClient."""
    import importlib

    client_cls = None
    for mod_name in ("trl.generation.vllm_client", "trl.extras.vllm_client"):
        try:
            mod = importlib.import_module(mod_name)
            client_cls = getattr(mod, "VLLMClient", None)
            if client_cls is not None:
                break
        except ImportError:
            pass

    if client_cls is None:
        raise ImportError("trl is required for VLLMClient")

    return client_cls(
        host=host,
        server_port=server_port,
        group_port=group_port,
        connection_timeout=connection_timeout,
    )
