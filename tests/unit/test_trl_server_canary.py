"""Canary for the installed TRL server-client token payload."""

from transformers import AutoTokenizer
from trl.generation.vllm_client import VLLMClient

from lmkd.rollout.base import SamplingSpec
from lmkd.rollout.vllm_server import VLLMServer


def test_trl_server_token_payload_canary(tiny_models):
    client = object.__new__(VLLMClient)
    client.model, client.base_url = "tiny", "http://unused"
    sent = {}

    def post(url, json):
        sent.update(json)
        return {
            "choices": [
                {
                    "prompt_token_ids": [5],
                    "token_ids": [42],
                    "finish_reason": "length",
                    "logprobs": {
                        "tokens": ["token_id:42"],
                        "token_logprobs": [-0.2],
                        "top_logprobs": [{"token_id:3": -0.1}],
                    },
                }
            ]
        }

    client._post = post
    server = VLLMServer()
    server.client = client
    server._tok = AutoTokenizer.from_pretrained(tiny_models["student"])
    roll = server._call_client_generate(
        [[5]], SamplingSpec(max_new_tokens=1, stop=["end"], stop_token_ids=[4], seed=7)
    )[0][0]
    assert sent["prompt"] == [[5]] and sent["return_token_ids"] is True
    assert sent["stop"] == ["end"] and sent["stop_token_ids"] == [4] and sent["seed"] == 7
    assert roll.response_logp == [-0.2] and roll.finish_reason == "length" and roll.text
