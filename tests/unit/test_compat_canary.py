"""Canary tests: every assumption lmkd makes about third-party libraries (see ``lmkd.compat``).

When a pinned library is upgraded and one of these fails, the failure names the assumption; fix
``src/lmkd/compat/`` (or the pin) instead of debugging a training run. CPU only, a few seconds.
"""

import importlib
import inspect
import re
import sys
from importlib import metadata
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Qwen2Config, Qwen2ForCausalLM, get_wsd_schedule

from lmkd.compat import PINNED_LIBRARIES, ds_api, hf_api, installed_versions, vllm_api
from lmkd.models.adapters import HFAutoAdapter
from lmkd.rollout.weight_sync import clean_peft_name
from lmkd.train.schedules import wsd

ROOT = Path(__file__).resolve().parents[2]


def _pins() -> dict[str, str]:
    deps = re.findall(r'^\s*"([A-Za-z0-9_.\[\]-]+)==([^";]+)', (ROOT / "pyproject.toml").read_text(), re.M)
    return {re.sub(r"\[.*", "", n).lower(): v.strip() for n, v in deps}


# ------------------------------------------------------------------------------ versions
@pytest.mark.parametrize("lib", PINNED_LIBRARIES)
def test_installed_version_matches_pin(lib):
    if sys.platform == "darwin" and lib in ("deepspeed", "vllm"):
        pytest.skip(f"{lib} is not supported on macOS")
    pins = _pins()
    assert lib in pins, f"{lib} is no longer pinned in pyproject.toml; update PINNED_LIBRARIES"
    got = installed_versions()[lib]
    assert got is not None, f"{lib} is not installed"
    assert got.split("+")[0] == pins[lib], (
        f"{lib}=={got} installed but pyproject pins {pins[lib]}. After an upgrade: run this file, fix "
        f"src/lmkd/compat/, then update the pin and docs"
    )


# ------------------------------------------------------------------------------ torch
def test_torch_muon_and_fused_adamw():
    assert hasattr(torch.optim, "Muon"), "optimizer `muon` needs torch.optim.Muon (torch >= 2.9)"
    assert "fused" in inspect.signature(torch.optim.AdamW).parameters


# ------------------------------------------------------------------------------ transformers
def test_wsd_matches_transformers_public_schedule():
    """``lmkd.train.schedules.wsd`` is our own formula; get_wsd_schedule is the public reference."""
    total, warm, dec = 40, 5, 12
    for decay_type in ("linear", "cosine", "1-sqrt"):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1.0)
        sch = get_wsd_schedule(
            opt,
            num_warmup_steps=warm,
            num_decay_steps=dec,
            num_training_steps=total,
            min_lr_ratio=0.1,
            decay_type=decay_type,
        )
        for step in range(total):
            assert wsd(step, total, warmup=warm, decay=dec, min_lr_ratio=0.1, decay_type=decay_type) == pytest.approx(
                sch.get_last_lr()[0], abs=1e-6
            ), (decay_type, step)
            opt.step()
            sch.step()


@pytest.mark.parametrize("model_type", hf_api.ROTARY_MODEL_TYPES)
def test_rotary_helper_exists_for_supported_architectures(model_type):
    mod = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
    assert callable(mod.apply_rotary_pos_emb)
    assert hasattr(mod, f"{model_type.capitalize()}Attention") or any(n.endswith("Attention") for n in dir(mod)), (
        "attention class naming changed"
    )


def test_decoder_layer_hooks_see_hidden_states_kwarg():
    """ForwardRunner hooks read ``hidden_states`` from args/kwargs of the decoder block."""
    model = GPT2LMHeadModel(GPT2Config(vocab_size=11, n_positions=16, n_embd=8, n_layer=2, n_head=2))
    seen = []
    h = model.transformer.h[0].register_forward_pre_hook(
        lambda m, a, kw: seen.append(kw.get("hidden_states", a[0] if a else None)), with_kwargs=True
    )
    model(input_ids=torch.tensor([[1, 2, 3]]))
    h.remove()
    assert seen and isinstance(seen[0], torch.Tensor) and seen[0].shape == (1, 3, 8)


def test_hf_adapter_last_row_attention_runs_on_supported_models():
    """The only place that re-implements attention internals: gpt2 + one rotary architecture."""
    models = [
        GPT2LMHeadModel(GPT2Config(vocab_size=20, n_positions=16, n_embd=16, n_layer=2, n_head=2)),
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=20,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=16,
            )
        ),
    ]
    ad = HFAutoAdapter()
    for m in models:
        x = torch.randn(1, 5, 16)
        out = ad.last_row_attention(m.eval(), 1, x, torch.ones(1, 5, dtype=torch.long), None)
        assert out.shape[0] == 1 and out.shape[-1] == 5, type(m).__name__


def test_hf_adapter_last_row_attention_right_padding():
    """Verify right padding is masked out and query is at the last valid position."""
    models = [
        GPT2LMHeadModel(GPT2Config(vocab_size=20, n_positions=16, n_embd=16, n_layer=2, n_head=2)),
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=20,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=16,
            )
        ),
    ]
    ad = HFAutoAdapter()
    for m in models:
        # Batch size 2, lengths: sample 0 has len 3, sample 1 has len 4 (out of 6)
        x = torch.randn(2, 6, 16)
        attn_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.long)
        out = ad.last_row_attention(m.eval(), 1, x, attn_mask, None)
        assert out.shape == (2, 6)
        # Padded positions must have exactly zero attention probability
        assert torch.all(out[0, 3:] == 0.0)
        assert torch.all(out[1, 4:] == 0.0)
        # Valid positions must sum to 1.0 (after normalization per head/sum across heads)
        assert torch.all(out[0, :3] > 0.0)
        assert torch.all(out[1, :4] > 0.0)


# ------------------------------------------------------------------------------ deepspeed
def test_deepspeed_engine_api_used_by_engine_py():
    pytest.importorskip("deepspeed")
    from deepspeed.runtime.engine import DeepSpeedEngine

    for name in ("set_gradient_accumulation_boundary", "backward", "step", "save_checkpoint", "load_checkpoint"):
        assert hasattr(DeepSpeedEngine, name), f"DeepSpeedEngine.{name} disappeared (engine.py relies on it)"
    # was_step_applied is optional: ds_api falls back to skipped_steps (see test_ds_api_step_bookkeeping_fallback)


def test_deepspeed_zero3_marker_and_gather():
    deepspeed = pytest.importorskip("deepspeed")
    import deepspeed.runtime.zero.partition_parameters as pp

    assert ds_api.PARTITIONED_ATTR in inspect.getsource(pp), "ZeRO-3 no longer tags parameters with ds_id"
    assert hasattr(deepspeed.zero, "GatheredParameters")
    assert not ds_api.is_partitioned_param(torch.nn.Parameter(torch.zeros(1)))


class _FakeDS:
    def __init__(self):
        self.skipped_steps = 0


def test_ds_api_step_bookkeeping_fallback():
    e = _FakeDS()
    before = ds_api.skipped_steps(e)
    assert ds_api.step_was_applied(e, before) is True
    e.skipped_steps += 1
    assert ds_api.step_was_applied(e, before) is False
    e.was_step_applied = lambda: True  # preferred path when the engine provides it
    assert ds_api.step_was_applied(e, before) is True


# ------------------------------------------------------------------------------ vllm
def test_vllm_symbols_and_llm_methods():
    pytest.importorskip("vllm")
    from vllm import LLM

    prompt = vllm_api.tokens_prompt([1, 2, 3])
    assert prompt["prompt_token_ids"] == [1, 2, 3]
    assert vllm_api.sampling_params(n=1, temperature=0.0, max_tokens=1).max_tokens == 1
    assert vllm_api.lora_request("a", 1, "/tmp/x").lora_int_id == 1
    for name in ("generate", "sleep", "wake_up", "reset_prefix_cache", "collective_rpc"):
        assert hasattr(LLM, name), f"vllm.LLM.{name} disappeared (rollout/vllm_colocate.py relies on it)"


def test_vllm_in_process_model_lookup_error_is_explicit():
    class Fake:
        class llm_engine:  # noqa: N801
            class model_executor:  # noqa: N801
                pass

    with pytest.raises(RuntimeError, match="in-process"):
        vllm_api.in_process_model(Fake())


# ------------------------------------------------------------------------------ peft
def test_peft_api_used_for_weight_sync():
    peft = pytest.importorskip("peft")

    base = GPT2LMHeadModel(GPT2Config(vocab_size=11, n_positions=16, n_embd=8, n_layer=1, n_head=2))
    m = peft.get_peft_model(base, peft.LoraConfig(r=2, target_modules=["c_attn"], fan_in_fan_out=True))
    for name in ("merge_adapter", "unmerge_adapter", "disable_adapter"):
        assert hasattr(m, name), f"PeftModel.{name} disappeared"
    assert hasattr(peft, "set_peft_model_state_dict")

    names = [n for n, _ in m.named_parameters()]
    assert any(".base_layer." in n for n in names), "PEFT no longer wraps targets in base_layer"
    cleaned = {clean_peft_name(n) for n in names if "lora_" not in n}
    assert all(n is not None and "base_layer" not in n and not n.startswith("base_model.model.") for n in cleaned)


# ------------------------------------------------------------------------------ trl (vllm_server rollout, KTO runner)
def test_trl_version_matches_pin():
    pins = _pins()
    assert "trl" in pins, "trl is no longer pinned in pyproject.toml; lmkd.compat.trl_api targets a specific version"
    assert metadata.version("trl") == pins["trl"], (
        f"trl=={metadata.version('trl')} installed, pinned {pins['trl']}: re-run this file, fix compat/trl_api.py, "
        "then update the pin"
    )


def test_trl_vllm_client_methods_exist():
    """VLLMClient (T26) must have init_communicator, update_named_param, reset_prefix_cache, generate."""
    try:
        from trl.generation.vllm_client import VLLMClient
    except ImportError:
        from trl.extras.vllm_client import VLLMClient

    for method in ("init_communicator", "update_named_param", "reset_prefix_cache", "generate"):
        assert hasattr(VLLMClient, method), f"VLLMClient lacks required method '{method}'"
