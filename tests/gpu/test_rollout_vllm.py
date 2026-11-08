"""T08 GPU tests: vLLM colocate rollout, weight sync, vllm_scorer. Use tiny random models (no download).

Run on a free GPU:  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q -m gpu tests/gpu/test_rollout_vllm.py
"""

from __future__ import annotations

import multiprocessing as mp
import traceback

import pytest
import torch

pytest.importorskip("vllm")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")]

from lmkd.core.config import recipe_from_dict  # noqa: E402
from lmkd.core.registry import TEACHERS  # noqa: E402
from lmkd.models.logits_view import LogitsView  # noqa: E402
from lmkd.rollout.base import SamplingSpec, rollouts_to_samples  # noqa: E402
from lmkd.teachers.signals import TeacherSignalSpec  # noqa: E402
from lmkd.train.builder import build_runtime  # noqa: E402

VLLM = {
    "type": "vllm_colocate",
    "gpu_memory_utilization": 0.1,
    "dtype": "float32",
    "enforce_eager": True,
    "max_model_len": 128,
}


def _child(fn, recipe, q):
    try:
        fn(recipe)
        q.put(None)
    except BaseException:  # noqa: BLE001
        q.put(traceback.format_exc())


def _isolated(fn, recipe, attempts: int = 3):
    """vLLM sleep mode allows one instance per process, so each scenario runs in a fresh process.
    Retries when vLLM's memory profiling is disturbed by another process sharing the GPU."""
    ctx = mp.get_context("spawn")
    for _ in range(attempts):
        q = ctx.Queue()
        p = ctx.Process(target=_child, args=(fn, recipe, q))
        p.start()
        err = q.get(timeout=900)
        p.join()
        if err is None or "Error in memory profiling" not in err:
            break
    assert err is None, err


@pytest.fixture
def cuda_recipe(base_recipe):
    free, total = torch.cuda.mem_get_info()
    if free < 0.1 * total + (1 << 30):  # vLLM refuses to start without its share free
        pytest.skip("not enough free GPU memory for a vLLM engine")
    base_recipe["engine"] = {"type": "local", "device": "cuda", "mixed_precision": "no"}
    return base_recipe


def _trainer_logp(rt, rollouts):
    b = rt.collator(rollouts_to_samples(rollouts))
    b = b.to(rt.engine.device)
    rt.student_model.eval()
    with torch.no_grad():
        h = rt.student_model(
            input_ids=b["student.input_ids"], attention_mask=b["student.attention_mask"], output_hidden_states=True
        ).hidden_states[-1]
    lp = LogitsView(h, rt.student_adapter.lm_head(rt.student_model)).target_logp(b["student.labels"])
    return lp, b["student.labels"] != -100, b["rollout.rollout_logp"]


def _body_generate_and_logp(recipe):
    rt = build_runtime(recipe_from_dict(recipe))
    outs = rt.rollout.generate([[5, 6, 7], [8, 9]], SamplingSpec(n=3, max_new_tokens=8, temperature=1.0, seed=0))
    assert len(outs) == 2 and all(len(rs) == 3 for rs in outs)
    rollouts = [r for rs in outs for r in rs]
    assert all(r.prompt_ids in ([5, 6, 7], [8, 9]) and r.finish_reason in ("stop", "length") for r in rollouts)
    lp, m, rl = _trainer_logp(rt, rollouts)
    assert (lp - rl.to(lp.device)).abs()[m].mean() < 1e-2
    # second call goes through sleep/wake again
    assert len(rt.rollout.generate([[5, 6]], SamplingSpec(max_new_tokens=3))) == 1


def test_vllm_generate_and_logp(cuda_recipe):
    cuda_recipe["rollout"] = dict(VLLM)
    _isolated(_body_generate_and_logp, cuda_recipe)


def _body_weight_sync(recipe):
    rt = build_runtime(recipe_from_dict(recipe))
    greedy = SamplingSpec(n=1, temperature=0.0, max_new_tokens=4, return_logprobs=False)
    first = rt.rollout.generate([[5, 6, 7]], greedy)[0][0]
    with torch.no_grad():  # GPT-2 ties lm_head <-> wte: logits all 0
        rt.student_model.lm_head.weight.zero_()
    rt.state.step += 1  # auto stale detection
    out = rt.rollout.generate([[5, 6, 7]], greedy)[0][0]
    assert out.response_ids[0] == 0


def test_vllm_weight_sync_changes_output(cuda_recipe):
    cuda_recipe["rollout"] = dict(VLLM)
    _isolated(_body_weight_sync, cuda_recipe)


def _body_lora_merge(recipe):
    rt = build_runtime(recipe_from_dict(recipe))
    for n, p in rt.student_model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.2)
    outs = rt.rollout.generate([[5, 6, 7]], SamplingSpec(n=4, max_new_tokens=6, temperature=1.0, seed=1))
    lp, m, rl = _trainer_logp(rt, outs[0])
    assert (lp - rl.to(lp.device)).abs()[m].mean() < 1e-2  # vLLM saw the merged weights
    assert not rt.student_model.base_model.model.transformer.h[0].attn.c_attn.merged


def test_vllm_lora_merge_matches_trainer(cuda_recipe):
    pytest.importorskip("peft")
    cuda_recipe["models"]["student"]["lora"] = {
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["c_attn"],
        "fan_in_fan_out": True,
    }
    cuda_recipe["rollout"] = {**VLLM, "lora_mode": "merge"}
    _isolated(_body_lora_merge, cuda_recipe)


def _seq_logps(rt, b):
    got = rt.teacher.signals(b, TeacherSignalSpec(kind="seq_logp"), None, None).seq_logp
    ref = TEACHERS.build({"type": "hf_frozen"})
    ref.setup(rt)
    return got, ref.signals(b, TeacherSignalSpec(kind="seq_logp"), None, None).seq_logp


def _body_scorer(recipe):
    rt = build_runtime(recipe_from_dict(recipe))
    b = rt.collator([rt.train_dataset[i] for i in range(3)]).to(rt.engine.device)
    m = b["student.labels"] != -100
    got = rt.teacher.signals(b, TeacherSignalSpec(kind="topk", k=5, tail="bucket"), None, None)
    hf = TEACHERS.build({"type": "hf_frozen"})
    hf.setup(rt)
    ref_lp = hf.signals(b, TeacherSignalSpec(kind="sampled_logp"), None, None)
    ref_k = hf.signals(b, TeacherSignalSpec(kind="topk", k=5, tail="bucket"), None, None)
    assert (got.token_logp - ref_lp.token_logp).abs()[m].max() < 2e-2
    assert (got.topk_logp[m].exp().sum(-1) - ref_k.topk_logp[m].exp().sum(-1)).abs().max() < 2e-2
    assert (got.topk_ids[m] == ref_k.topk_ids[m]).float().mean() > 0.9
    assert (got.token_logp[~m] == 0).all()  # same convention as hf_frozen: 0 off-label
    sq = _seq_logps(rt, b)
    assert torch.allclose(sq[0], sq[1], atol=2e-2)
    assert got.topk_ids.shape == (3, b["student.input_ids"].shape[1], 5)
    with pytest.raises(ValueError, match="arbitrary"):
        rt.teacher.check(TeacherSignalSpec(kind="topk", k=5, support="student"), None)
    with pytest.raises(ValueError, match="temperature"):
        rt.teacher.check(TeacherSignalSpec(kind="topk", k=5, temperature=2.0), None)


def test_vllm_scorer_matches_hf(cuda_recipe, tiny_models):
    cuda_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    cuda_recipe["teacher"] = {
        "provider": {"type": "vllm_scorer", "k": 5, "gpu_memory_utilization": 0.1, "max_model_len": 128},
        "input": "same",
    }
    _isolated(_body_scorer, cuda_recipe)


def _body_lora_hot_load(recipe):
    rt = build_runtime(recipe_from_dict(recipe))
    sp = SamplingSpec(n=4, max_new_tokens=6, temperature=1.0, seed=1)
    for n, p in rt.student_model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.3)
    outs = rt.rollout.generate([[5, 6, 7]], sp)
    lp, m, rl = _trainer_logp(rt, outs[0])
    assert (lp - rl.to(lp.device)).abs()[m].mean() < 5e-2
    first_id = rt.rollout._lora_id
    for n, p in rt.student_model.named_parameters():  # adapter changes -> re-sync
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.3)
    rt.state.step += 1
    outs = rt.rollout.generate([[5, 6, 7]], sp)
    assert rt.rollout._lora_id == first_id + 1
    lp, m, rl = _trainer_logp(rt, outs[0])
    assert (lp - rl.to(lp.device)).abs()[m].mean() < 5e-2


def test_vllm_lora_hot_load_matches_trainer(cuda_recipe, tiny_models, tmp_path):
    pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from conftest import _make_tokenizer

    path = tmp_path / "qwen"
    tok = _make_tokenizer(path)
    torch.manual_seed(0)
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tok),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            eos_token_id=1,
            bos_token_id=1,
            tie_word_embeddings=False,
        )
    ).save_pretrained(path)
    cuda_recipe["models"]["student"].update(
        path=str(path), dtype="bf16", lora={"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}
    )
    cuda_recipe["rollout"] = {**VLLM, "dtype": "bfloat16", "lora_mode": "hot_load"}  # vLLM LoRA kernels need 16-bit
    _isolated(_body_lora_hot_load, cuda_recipe)


def _body_tp2(recipe):
    rt = build_runtime(recipe_from_dict(recipe))
    greedy = SamplingSpec(n=1, temperature=0.0, max_new_tokens=4, return_logprobs=False)
    outs = rt.rollout.generate([[5, 6, 7]], SamplingSpec(n=2, max_new_tokens=6, seed=0))
    lp, m, rl = _trainer_logp(rt, outs[0])
    assert (lp - rl.to(lp.device)).abs()[m].mean() < 1e-2  # weights synced to both TP shards
    with torch.no_grad():
        rt.student_model.lm_head.weight.zero_()
    rt.state.step += 1
    assert rt.rollout.generate([[5, 6, 7]], greedy)[0][0].response_ids[0] == 0


def test_vllm_tensor_parallel_weight_sync(cuda_recipe):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    free = min(torch.cuda.mem_get_info(i)[0] for i in range(2))
    if free < 0.1 * torch.cuda.mem_get_info(0)[1] + (1 << 30):
        pytest.skip("not enough free GPU memory on both GPUs")
    cuda_recipe["rollout"] = {**VLLM, "tensor_parallel_size": 2}
    _isolated(_body_tp2, cuda_recipe)
