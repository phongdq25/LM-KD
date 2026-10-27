"""vLLM glue (see ``lmkd.compat``): construction, prompts, LoRA requests, in-process model access."""
from __future__ import annotations

import os
from typing import Any


def import_vllm() -> Any:
    """Import vllm with the V1 multiprocessing engine off: weight sync needs the model in THIS process."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import vllm

    return vllm


def tokens_prompt(ids: list[int]) -> Any:
    from vllm.inputs import TokensPrompt  # name/location has moved between vLLM versions

    return TokensPrompt(prompt_token_ids=ids)


def lora_request(name: str, lora_id: int, path: str) -> Any:
    from vllm.lora.request import LoRARequest

    return LoRARequest(name, lora_id, path)


def sampling_params(**kwargs: Any) -> Any:
    from vllm import SamplingParams

    return SamplingParams(**kwargs)


# Where the in-process model lives under ``llm.llm_engine.model_executor`` (layout depends on the version).
_MODEL_PATHS = ("driver_worker.worker.model_runner.model", "driver_worker.model_runner.model")


def in_process_model(llm: Any) -> Any:
    """The in-process vLLM ``nn.Module`` (has ``load_weights``); raises RuntimeError if unreachable."""
    ex = llm.llm_engine.model_executor
    for path in _MODEL_PATHS:
        cur = ex
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if cur is not None:
            return cur
    raise RuntimeError("cannot reach vLLM model in-process; set VLLM_ENABLE_V1_MULTIPROCESSING=0 "
                       "or use vllm_server (T26)")


def worker_load_weights(worker: Any, named: list) -> None:
    """Runs inside a vLLM worker process (tensor parallel): ``named`` = list of (name, cpu tensor)."""
    model = worker.get_model() if hasattr(worker, "get_model") else worker.model_runner.model
    model.load_weights(iter(named))
