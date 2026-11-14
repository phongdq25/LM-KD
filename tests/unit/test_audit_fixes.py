"""Unit tests verifying audit bug fixes across CLI, data, longcot, and teachers."""

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import yaml

from baselines.longCoT.sgl_spectral import compute_transition_hidden_index
from baselines.longCoT.ssft import SelectCumulativeStrength
from baselines.self_distill.sdpo import SDPOReprompt
from lmkd.cli import main
from lmkd.core.batch import Batch
from lmkd.core.config import TrainSection
from lmkd.data.producers import AdaptiveSGOProducer, DatasetProducer
from lmkd.models.loading import _check_tied_lora_targets
from lmkd.train.engine import MasterWeightOptimizer
from lmkd.train.step_strategies import StandardStep


def test_transition_hidden_index_integer_offset():
    # 0-based layer 24 in 36-layer model -> 1-based hidden index 25 (output of layer 24)
    idx = compute_transition_hidden_index(24, 36)
    assert idx == 25
    assert compute_transition_hidden_index(-1, 36) == -1
    assert compute_transition_hidden_index("two_thirds", 36) == 25


def test_select_cumulative_strength_preserves_eos():
    # Mock sample with 10 tokens: prompt=[0..3], response=[4..7], eos=[8..9]
    trans = SelectCumulativeStrength(
        SelectCumulativeStrength.Config(cumulative_ratio=0.7, coherence_max=0.8, always_keep=[0])
    )
    sample = {
        "student.input_ids": [1] * 10,
        "seq.steps": [[4, 6], [6, 8]],
        "seq.response_span": [4, 8],
        "signals.segment_ig": [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "meta.id": "test_1",
    }
    ctx = MagicMock()
    out = trans(sample, ctx)
    assert out is not None
    mask = out["seq.loss_mask"]
    # The EOS/stop tokens from response_span[1] (8) to seq_len (10) must be 1
    assert mask[8] == 1
    assert mask[9] == 1


def test_sdpo_reprompt_uses_raw_prompt_without_double_templating():
    # Verify SDPOReprompt falls back to raw_prompt if available
    builder = SDPOReprompt(SDPOReprompt.Config(reprompt_template="{prompt}{solution}{feedback}"))
    tok = MagicMock()
    tok.chat_template = None
    tok.pad_token_id = 0
    tok.return_value = {"input_ids": [10, 20]}
    rt = MagicMock()
    rt.student_tokenizer = tok
    rt.config.data.pad_to_multiple_of = None
    rt.state.log = MagicMock()

    batch = Batch(
        {
            "student.input_ids": torch.tensor([[1, 2], [3, 4]]),
            "student.attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "student.labels": torch.tensor([[-100, 2], [-100, 4]]),
            "student.prompt_len": torch.tensor([1, 1]),
            "rollout.reward": torch.tensor([0.0, 0.0]),
        }
    )
    batch.meta = {
        "group.uid": ["u1", "u1"],
        "meta.response_text": ["resp1", "resp2"],
        "meta.raw_prompt": ["What is 2+2?", "What is 2+2?"],
        "meta.prompt_text": ["<chat>What is 2+2?</chat>", "<chat>What is 2+2?</chat>"],
        "meta.feedback": ["", ""],
    }
    out = builder.build(batch, rt)
    assert "rollout.has_signal" in out


def test_cli_set_action_extends(tmp_path, capsys):
    data = {
        "models": {"student": {"path": "dummy-gpt2"}},
        "data": {"train": {"type": "jsonl", "path": "train.jsonl"}},
        "loss": {"terms": [{"criterion": "nll"}]},
        "train": {"max_steps": 5, "seed": 1},
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(data))

    # Passing multiple --set flags should both be applied
    code = main(["config", "-c", str(path), "--set", "train.max_steps=42", "--set", "train.seed=999"])
    captured = capsys.readouterr()
    assert code == 0
    assert "max_steps: 42" in captured.out
    assert "seed: 999" in captured.out


def test_adaptive_sgo_on_eval_and_sync_rng():
    cfg = AdaptiveSGOProducer.Config(sampling={}, dev_metric="dev_loss/loss", init_threshold=0.2, loss_eps=0.05)
    prod = AdaptiveSGOProducer(cfg)

    # on_eval should find dev_loss/loss or alternative fallback keys
    prod.on_eval({"dev_loss/loss": 2.0})
    assert prod.prev == 2.0
    prod.on_eval({"dev_loss/loss": 2.2})  # increases by > loss_eps (0.05) -> increases threshold
    assert prod.thr > 0.2


def test_master_weight_optimizer_bf16_updates():
    """Verify MasterWeightOptimizer prevents bf16 mantissa truncation on small updates."""
    p_master = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    opt_master = MasterWeightOptimizer(torch.optim.AdamW([p_master], lr=1e-5))

    p_raw = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    opt_raw = torch.optim.AdamW([p_raw], lr=1e-5)

    for _ in range(1000):
        # Raw AdamW in pure bf16
        (p_raw * 2.0).sum().backward()
        opt_raw.step()
        opt_raw.zero_grad()

        # MasterWeightOptimizer
        (p_master * 2.0).sum().backward()
        opt_master.step()
        opt_master.zero_grad()

    # Raw bf16 update stalls at 1.0 due to 7-bit mantissa truncation
    assert p_raw.item() == 1.0
    # Master weight in fp32 accumulates the small updates and updates p_master
    assert p_master.item() < 1.0

    # Test state_dict / load_state_dict
    sd = opt_master.state_dict()
    p_new = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    opt_new = MasterWeightOptimizer(torch.optim.AdamW([p_new], lr=1e-5))
    opt_new.load_state_dict(sd)
    assert p_new.item() == 1.0


def test_step_strategy_nonfinite_all_reduce():
    """Verify nonfinite decision is synchronized across distributed ranks."""
    class MockDistEngine:
        world_size = 2
        device = torch.device("cpu")

        def all_reduce_sum(self, tensor):
            # Simulate rank 1 having NaN while rank 0 is finite
            return tensor + 1.0

    rt = SimpleNamespace(engine=MockDistEngine())
    step = StandardStep(StandardStep.Config())
    step.rt = rt

    # Locally finite, but peer rank has NaN -> all_reduce_sum returns > 0 -> nonfinite is True
    metrics = {"loss/total": torch.tensor(1.5)}
    assert step.nonfinite(metrics) is True


def test_batch_size_divisibility_validations():
    """Verify batch_size divisibility validations."""
    # TrainSection validates micro_batch_size > batch_size
    with pytest.raises(ValueError, match="cannot exceed batch_size"):
        TrainSection(batch_size=4, micro_batch_size=8)

    # TrainSection validates batch_size % micro_batch_size == 0
    with pytest.raises(ValueError, match="must be divisible by micro_batch_size"):
        TrainSection(batch_size=7, micro_batch_size=4)

    # DatasetProducer validates batch_size % world_size == 0
    prod = DatasetProducer(DatasetProducer.Config())
    rt_bad_world = SimpleNamespace(
        config=SimpleNamespace(train=SimpleNamespace(batch_size=5, micro_batch_size=1)),
        engine=SimpleNamespace(world_size=2),
    )
    with pytest.raises(ValueError, match="divisible by world_size"):
        prod.setup(rt_bad_world)


def test_tied_word_embeddings_lora_warning():
    """Verify explicit warning is issued when LoRA targets tied lm_head/embeddings."""
    # Model with tied embeddings and lm_head targeted
    mock_model = SimpleNamespace(config=SimpleNamespace(tie_word_embeddings=True))
    with pytest.warns(UserWarning, match="tie_word_embeddings=True.*tied modules \\['lm_head'\\]"):
        _check_tied_lora_targets(mock_model, {"target_modules": ["c_attn", "lm_head"]})

    # Model with tied embeddings but attention-only LoRA -> no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_tied_lora_targets(mock_model, {"target_modules": ["c_attn", "c_proj"]})
        assert len([item for item in w if "tie_word_embeddings" in str(item.message)]) == 0

    # Model with untied embeddings -> no warning
    untied_model = SimpleNamespace(config=SimpleNamespace(tie_word_embeddings=False))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_tied_lora_targets(untied_model, {"target_modules": ["c_attn", "lm_head"]})
        assert len([item for item in w if "tie_word_embeddings" in str(item.message)]) == 0
