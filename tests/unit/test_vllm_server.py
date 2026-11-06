"""Unit tests for VLLMServer rollout engine with fake client (T26)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lmkd.core.config import recipe_from_dict
from lmkd.rollout.base import SamplingSpec
from lmkd.rollout.vllm_server import VLLMServer
from lmkd.rollout.weight_sync import iter_student_weights
from lmkd.train.builder import build_runtime


class FakeClient:
    """Mock client for testing VLLMServer without an external server process."""

    def __init__(self, *args, **kwargs):
        self.synced_params: list[tuple[str, tuple[int, ...]]] = []
        self.prefix_cache_resets = 0
        self.device = None
        self.model = "test_model"

    def init_communicator(self, device):
        self.device = device

    def update_named_param(self, name: str, tensor):
        self.synced_params.append((name, tuple(tensor.shape)))

    def reset_prefix_cache(self):
        self.prefix_cache_resets += 1

    def generate(self, prompts, **kwargs):
        n = kwargs.get("n", 1)
        completions = []
        logprobs = []
        for _ in prompts:
            for _ in range(n):
                completions.append([42, 43])
                logprobs.append([[0.1], [0.2]])
        return {
            "prompt_ids": prompts,
            "completion_ids": completions,
            "logprobs": logprobs,
            "logprob_token_ids": None,
        }


def test_server_sync_and_generate_with_fake_client(base_recipe, monkeypatch):
    """FakeClient records update_named_param calls and returns fixed rollouts:
    - parameter names match the set from iter_student_weights
    - reset_prefix_cache is called exactly once per sync
    - does not sync again when state.step has not changed.
    """
    fake_client = FakeClient()
    monkeypatch.setattr("lmkd.rollout.vllm_server.get_vllm_client", lambda **kw: fake_client)

    base_recipe["rollout"] = {
        "type": "vllm_server",
        "sync_every": 1,
    }
    rt = build_runtime(recipe_from_dict(base_recipe))
    engine = rt.rollout
    assert isinstance(engine, VLLMServer)

    # First generate call -> triggers first sync at step 0
    prompts = [[5, 6, 7], [8, 9]]
    spec = SamplingSpec(n=2, max_new_tokens=4, return_logprobs=True)
    rollouts = engine.generate(prompts, spec)

    assert len(rollouts) == 2
    assert len(rollouts[0]) == 2
    assert rollouts[0][0].response_ids == [42, 43]
    assert rollouts[0][0].response_logp == [0.1, 0.2]

    # Verify synced parameter names match iter_student_weights
    expected_names = {name for name, _ in iter_student_weights(rt, "full")}
    synced_names = {name for name, _ in fake_client.synced_params}
    assert synced_names == expected_names
    assert fake_client.prefix_cache_resets == 1

    # Second generate call at the same step -> should NOT sync again
    n_params_before = len(fake_client.synced_params)
    engine.generate(prompts, spec)
    assert len(fake_client.synced_params) == n_params_before
    assert fake_client.prefix_cache_resets == 1

    # Advance step -> triggers sync again
    rt.state.step += 1
    engine.generate(prompts, spec)
    assert len(fake_client.synced_params) == n_params_before * 2
    assert fake_client.prefix_cache_resets == 2


def test_server_decodes_and_marks_length(base_recipe, monkeypatch):
    monkeypatch.setattr("lmkd.rollout.vllm_server.get_vllm_client", lambda **kw: FakeClient())
    base_recipe["rollout"] = {"type": "vllm_server"}
    rt = build_runtime(recipe_from_dict(base_recipe))
    rolls = rt.rollout.generate([[5]], SamplingSpec(n=2, max_new_tokens=2))
    assert rolls[0][0].text == rt.student_tokenizer.decode([42, 43], skip_special_tokens=True)
    assert rolls[0][0].finish_reason == "length"


def test_server_forwards_stop_and_validates_payload(base_recipe, monkeypatch):
    client = FakeClient()
    seen = {}

    def generate(prompts, **kw):
        seen.update(kw)
        return {"completion_ids": [[3, 1]], "logprobs": [[[-0.1], [-0.2]]]}

    client.generate = generate
    monkeypatch.setattr("lmkd.rollout.vllm_server.get_vllm_client", lambda **kw: client)
    base_recipe["rollout"] = {"type": "vllm_server"}
    rt = build_runtime(recipe_from_dict(base_recipe))
    spec = SamplingSpec(stop=["end"], stop_token_ids=[4], seed=7, max_new_tokens=2)
    roll = rt.rollout.generate([[5]], spec)[0][0]
    assert roll.finish_reason == "stop"
    assert seen["generation_kwargs"] == {"stop": ["end"], "stop_token_ids": [4], "seed": 7}
    with pytest.raises(ValueError, match="completion"):
        rt.rollout.generate([[5]], SamplingSpec(n=2))


def test_sync_collectives_match_on_all_ranks(monkeypatch):
    monkeypatch.setattr("lmkd.rollout.vllm_server.iter_student_weights", lambda *args: iter([]))
    traces = []
    for main in [True, False]:
        trace = []
        engine = SimpleNamespace(
            is_main=main,
            world_size=2,
            all_gather_object=lambda value: trace.append("gather") or [None, None],
            barrier=lambda: trace.append("barrier"),
        )
        server = VLLMServer()
        server.rt = SimpleNamespace(
            engine=engine,
            state=SimpleNamespace(step=0),
            student_model=None,
            config=SimpleNamespace(models=SimpleNamespace(student=SimpleNamespace(lora=None))),
        )
        server.client = FakeClient() if main else None
        server._last_sync_step = -1
        server._maybe_sync()
        traces.append(trace)
    assert traces[0] == traces[1] and "barrier" in traces[0]
