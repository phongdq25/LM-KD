"""Two-rank fake-server integration: collective ordering and rank-zero failures."""
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

import lmkd.rollout.vllm_server as module
from lmkd.rollout.base import SamplingSpec
from lmkd.rollout.vllm_server import VLLMServer


class Engine:
    """CPU collectives with a strict deadline, including a collective weight stream."""
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world_size = 2
        self.is_main = self.rank == 0
        self.is_partitioned = False
        self.device = torch.device("cpu")
        self.streamed = 0

    def all_gather_object(self, value):
        values = [None, None]
        dist.all_gather_object(values, value)
        return values

    def barrier(self):
        dist.barrier()

    def iter_full_state_dict(self, model):
        for name, value in model.named_parameters():
            dist.all_reduce(value.detach().clone())
            self.streamed += 1
            yield name, value


class Client:
    """Rank-zero client can fail either weight transfer or generation."""
    def __init__(self):
        self.fail_sync = self.fail_generate = False
        self.syncs = 0

    def init_communicator(self, device):
        pass

    def update_named_param(self, name, tensor):
        if self.fail_sync:
            raise ConnectionError("sync outage")
        self.syncs += 1

    def reset_prefix_cache(self):
        pass

    def generate(self, prompts, **kwargs):
        if self.fail_generate:
            raise ConnectionError("generate outage")
        return {"completion_ids": [[p[0], 1] for p in prompts for _ in range(kwargs["n"])],
                "logprobs": [[[-.1], [-.2]] for p in prompts for _ in range(kwargs["n"])]}


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    try:
        eng, client = Engine(), Client()
        module.get_vllm_client = lambda **kw: client
        runtime = SimpleNamespace(engine=eng, student_model=torch.nn.Linear(2, 2),
            student_tokenizer=SimpleNamespace(eos_token_id=1, decode=lambda ids, **kw: str(ids)),
            state=SimpleNamespace(step=0),
            config=SimpleNamespace(models=SimpleNamespace(student=SimpleNamespace(lora=None))))
        server = VLLMServer()
        server.setup(runtime)
        spec = SamplingSpec(n=2, max_new_tokens=2)
        for step in [0, 0, 1]:
            runtime.state.step = step
            rolls = server.generate([[eng.rank + 3]], spec)
            assert len(rolls) == 1 and len(rolls[0]) == 2
            assert rolls[0][0].response_ids == [eng.rank + 3, 1]
            assert rolls[0][0].text and rolls[0][0].finish_reason == "stop"
        messages = []
        client.fail_generate = True
        try:
            server.generate([[3]], spec)
        except RuntimeError as exc:
            messages.append(str(exc))
        client.fail_generate = False
        client.fail_sync = True
        eng.is_partitioned = True
        runtime.state.step = 2
        try:
            server.generate([[3]], spec)
        except RuntimeError as exc:
            messages.append(str(exc))
        assert len(messages) == 2 and eng.streamed == 2
        assert "generate outage" in messages[0] and "sync outage" in messages[1]
        assert eng.all_gather_object(messages) == [messages, messages]
        Path(os.environ["AUDIT_SERVER_OUTPUT"], f"rank{eng.rank}.json").write_text(json.dumps(messages))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
