import pytest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from lmkd.models.adapters import HFAutoAdapter, HiddenForward
from lmkd.models.forward import ForwardRunner, SideRequest


def make(kind):
    torch.manual_seed(12)
    if kind == "gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(vocab_size=67, n_embd=32, n_layer=2, n_head=4, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        )
    else:
        cfg, cls = {
            "llama": (LlamaConfig, LlamaForCausalLM),
            "qwen2": (Qwen2Config, Qwen2ForCausalLM),
            "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        }[kind]
        model = cls(
            cfg(
                vocab_size=67,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                attention_dropout=0.0,
            )
        )
    model.set_attn_implementation("eager")
    model.eval()
    return (
        model,
        HFAutoAdapter(HFAutoAdapter.Config()),
        {"input_ids": torch.randint(0, 67, (2, 7)), "attention_mask": torch.tensor([[1] * 7, [1] * 4 + [0] * 3])},
    )


@pytest.mark.parametrize("kind", ["gpt2", "llama"])
def test_hooks_match_output_hidden_states(kind, monkeypatch):
    model, adapter, inputs = make(kind)
    ref = adapter.forward_hidden(model, **inputs, output_hidden_states=True)
    original = adapter.forward_hidden

    def checked(*args, **kwargs):
        assert kwargs["output_hidden_states"] is False
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "forward_hidden", checked)
    out = ForwardRunner.run(model, adapter, inputs, SideRequest(hidden=(0, 1, -1, 2)), grad=True)[0]
    for i in (0, 1, -1, 2):
        torch.testing.assert_close(out.hidden(i), ref.hidden_states[i], atol=1e-6, rtol=0.0)


def test_hooks_through_module_and_grad_hook():
    model, adapter, inputs = make("gpt2")
    module = HiddenForward(model, adapter)
    calls = []
    handle = module.register_forward_pre_hook(lambda *args: calls.append(1))
    out = ForwardRunner.run(model, adapter, inputs, SideRequest(hidden=(1,)), grad=True, module=module)[0]
    handle.remove()
    assert calls == [1]
    out.resolve("hidden[1]").register_hook(lambda g: g * 0)
    out.last_hidden.square().sum().backward()
    assert model.transformer.h[0].attn.c_attn.weight.grad.abs().sum() == 0
    assert model.transformer.h[1].attn.c_attn.weight.grad.abs().sum() > 0


def test_hooks_with_grad_checkpointing():
    model, adapter, inputs = make("gpt2")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    out = ForwardRunner.run(model, adapter, inputs, SideRequest(hidden=(1,)), grad=True)[0]
    before = out.hidden(1).detach().clone()
    out.hidden(1).sum().backward()
    torch.testing.assert_close(out.hidden(1), before)


def test_forward_through_engine_and_bundle(monkeypatch):
    """CPU double checks the T05 wrapper contract, without claiming DeepSpeed parity."""
    model, adapter, inputs = make("gpt2")
    calls = []
    original = adapter.forward_hidden

    def checked(*args, **kwargs):
        assert calls == ["engine", "bundle"]
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "forward_hidden", checked)

    class Bundle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.student_module = HiddenForward(model, adapter)

        def forward(self, **kwargs):
            calls.append("bundle")
            return self.student_module(**kwargs)

    class EngineWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = Bundle()

        def forward(self, **kwargs):
            calls.append("engine")
            return self.module(**kwargs)

    out = ForwardRunner.run(
        model, adapter, inputs, SideRequest(hidden=(1,), attention_last_row=(1,)), grad=True, module=EngineWrapper()
    )[0]
    out.resolve("hidden[1]").register_hook(lambda g: g * 0)
    out.last_hidden.square().sum().backward()
    assert calls == ["engine", "bundle"]
    assert model.transformer.h[0].attn.c_attn.weight.grad.abs().sum() == 0
    assert model.transformer.h[1].attn.c_attn.weight.grad.abs().sum() > 0
    assert not out.attention_last_row(1).requires_grad


@pytest.mark.parametrize("kind", ["gpt2", "llama", "qwen2", "qwen3"])
def test_attention_last_row(kind, monkeypatch):
    model, adapter, inputs = make(kind)
    ref = adapter.forward_hidden(model, **inputs, output_attentions=True)
    original = adapter.forward_hidden

    def checked(*args, **kwargs):
        assert not kwargs["output_attentions"]
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "forward_hidden", checked)
    req = SideRequest(attention_last_row=(1, -1))
    assert req.needs_forward
    assert req.merge(SideRequest(attention_last_row=(2,))).attention_last_row == (-1, 1, 2)
    out = ForwardRunner.run(model, adapter, inputs, req, grad=True)[0]
    mask = inputs["attention_mask"]
    last_idx = (mask.cumsum(-1) * mask).argmax(-1)
    for i in (1, 2):
        full = ref.attentions[i - 1].sum(1)
        expected = torch.stack([full[b, last_idx[b]] for b in range(len(last_idx))])
        torch.testing.assert_close(out.attention_last_row(i), expected, atol=1e-5, rtol=0.0)
        assert not out.attention_last_row(i).requires_grad
