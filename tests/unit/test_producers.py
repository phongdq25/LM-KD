"""Unit tests for on-policy producers, replay buffer, and scheduling decisions (T09)."""

from __future__ import annotations

import pytest
import torch

from lmkd.core.config import recipe_from_dict
from lmkd.core.registry import PRODUCERS
from lmkd.core.state import TrainState
from lmkd.data.collator import Collator
from lmkd.data.producers import (
    AdaptiveSGOProducer,
    DatasetProducer,
    decide,
    make_on_policy_batch,
    sampling_threshold,
)
from lmkd.data.replay import ReplayBuffer, batch_to_samples
from lmkd.rollout.base import Rollout, SamplingSpec
from lmkd.train.builder import build_runtime


def test_sampling_threshold():
    """Verify decreasing, constant, and increasing schedule formulas, and invalid schedule error."""
    assert sampling_threshold(0.8, "decreasing", 25, 100) == pytest.approx(0.6)
    assert sampling_threshold(0.8, "constant", 25, 100) == pytest.approx(0.4)
    assert sampling_threshold(0.8, "increasing", 25, 100) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        sampling_threshold(0.8, "unknown", 25, 100)


def test_decide_rule_table():
    """Verify decision logic against exact rules from DistiLLM."""
    assert decide(0.1, 0.5, 0.2, 0, 10) == "generate"
    assert decide(0.3, 0.5, 0.2, 10, 10) == "replay"
    assert decide(0.3, 0.5, 0.2, 3, 10) == "generate"
    assert decide(0.7, 0.5, 0.2, 10, 10) == "data"


def test_on_eval_threshold_update():
    """Verify threshold increases when dev loss increases >= loss_eps, capped at max_threshold."""
    cfg = AdaptiveSGOProducer.Config(
        init_threshold=0.0,
        threshold_step=0.1,
        max_threshold=1.0,
        loss_eps=0.1,
        dev_metric="eval/dev_loss",
    )
    prod = AdaptiveSGOProducer(cfg)
    assert prod.prev is None
    assert prod.thr == pytest.approx(0.0)

    # First eval initializes prev
    prod.on_eval({"eval/dev_loss": 2.0})
    assert prod.prev == pytest.approx(2.0)
    assert prod.thr == pytest.approx(0.0)

    # Increase less than loss_eps (0.1) -> no change
    prod.on_eval({"eval/dev_loss": 2.05})
    assert prod.prev == pytest.approx(2.0)
    assert prod.thr == pytest.approx(0.0)

    # Increase >= loss_eps -> thr += 0.1, prev updated
    prod.on_eval({"eval/dev_loss": 2.2})
    assert prod.prev == pytest.approx(2.2)
    assert prod.thr == pytest.approx(0.1)

    # Repeat until max_threshold
    for i in range(15):
        prod.on_eval({"eval/dev_loss": 2.2 + (i + 1) * 0.2})
    assert prod.thr == pytest.approx(1.0)


def test_replay_buffer_fifo_and_seed():
    """Buffer keeps last capacity items (FIFO) and seed gives deterministic sample."""
    buf = ReplayBuffer(capacity=3, seed=42)
    samples = [{"id": i} for i in range(5)]
    buf.add_samples(samples)
    assert len(buf) == 3
    # FIFO: oldest dropped, last 3 kept
    assert [s["id"] for s in buf.buffer] == [2, 3, 4]

    buf1 = ReplayBuffer(capacity=10, seed=123)
    buf2 = ReplayBuffer(capacity=10, seed=123)
    items = [{"v": i} for i in range(8)]
    buf1.add_samples(items)
    buf2.add_samples(items)
    s1 = buf1.sample(3)
    s2 = buf2.sample(3)
    assert s1 == s2


def test_batch_to_samples_roundtrip():
    """Collating samples, unpacking with batch_to_samples, and re-collating yields identical tensors."""
    collator = Collator(pad_token_ids={"student": 0}, pad_to_multiple_of=8)
    s0 = {
        "student.input_ids": [10, 11, 12],
        "student.attention_mask": [1, 1, 1],
        "student.labels": [-100, 11, 12],
        "student.prompt_len": 1,
        "rollout.rollout_logp": [0.0, -0.5, -0.2],
        "meta.id": "id0",
    }
    s1 = {
        "student.input_ids": [20, 21, 22, 23, 24],
        "student.attention_mask": [1, 1, 1, 1, 1],
        "student.labels": [-100, -100, 22, 23, 24],
        "student.prompt_len": 2,
        "rollout.rollout_logp": [0.0, 0.0, -0.1, -0.2, -0.3],
        "meta.id": "id1",
    }
    s2 = {
        "student.input_ids": [30, 31],
        "student.attention_mask": [1, 1],
        "student.labels": [-100, 31],
        "student.prompt_len": 1,
        "rollout.rollout_logp": [0.0, -0.4],
        "meta.id": "id2",
    }
    batch1 = collator([s0, s1, s2])
    unpacked = batch_to_samples(batch1)
    assert len(unpacked) == 3
    batch2 = collator(unpacked)

    for k in batch1:
        t1, t2 = batch1[k], batch2[k]
        if isinstance(t1, torch.Tensor):
            assert torch.equal(t1, t2), f"Mismatch for tensor key {k}"
    assert batch1.meta == batch2.meta


def test_mixed_lmbda_zero_is_dataset(base_recipe):
    """When lmbda=0, MixedProducer produces identical batches to DatasetProducer."""
    base_recipe["rollout"] = {"type": "hf"}
    base_recipe["producer"] = {"type": "mixed", "lmbda": 0.0}
    cfg = recipe_from_dict(base_recipe)
    rt = build_runtime(cfg)

    ds_prod = DatasetProducer(DatasetProducer.Config())
    ds_prod.setup(rt)

    state = TrainState()
    for _ in range(3):
        b_mixed = rt.producer.next(state)
        b_ds = ds_prod.next(state)
        for k in b_mixed:
            if isinstance(b_mixed[k], torch.Tensor):
                assert torch.equal(b_mixed[k], b_ds[k])


def test_on_policy_batch_labels(base_recipe):
    """Target-aligned labels and preserved metadata order in make_on_policy_batch."""
    base_recipe["rollout"] = {"type": "hf"}
    cfg = recipe_from_dict(base_recipe)
    rt = build_runtime(cfg)

    # Fetch a source batch
    src = rt.producer.next(TrainState())
    spec = SamplingSpec(max_new_tokens=4, temperature=0.0)
    ob = make_on_policy_batch(rt, src, spec)

    assert ob.batch_size == src.batch_size
    assert ob.meta.get("meta.id") == src.meta.get("meta.id")

    input_ids = ob["student.input_ids"]
    labels = ob["student.labels"]
    prompt_lens = ob["student.prompt_len"]
    mask = ob["student.attention_mask"]

    for i in range(ob.batch_size):
        plen = int(prompt_lens[i].item())
        real_len = int(mask[i].sum().item())
        # Prompt positions are masked with -100
        assert (labels[i, :plen] == -100).all()
        # Generated positions match input_ids
        if real_len > plen:
            assert torch.equal(labels[i, plen:real_len], input_ids[i, plen:real_len])


class _FakeRollout:
    def generate(self, prompts, spec, lora=None):
        return [[Rollout(prompt_ids=p, response_ids=[7, 8, 1])] for p in prompts]


def test_adaptive_state_dict_roundtrip(base_recipe):
    """AdaptiveSGOProducer state_dict/load_state_dict reproduces decision sequence."""
    base_recipe["rollout"] = {"type": "hf"}
    base_recipe["producer"] = {
        "type": "adaptive_sgo",
        "init_threshold": 0.5,
        "schedule": "constant",
        "capacity": 10,
        "save_buffer": True,
        "seed_offset": 42,
    }
    cfg = recipe_from_dict(base_recipe)
    rt = build_runtime(cfg)
    rt.rollout = _FakeRollout()

    prod1 = rt.producer
    state = TrainState(total_steps=50)

    # Run 5 steps to populate state and buffer
    for step in range(5):
        state.step = step
        prod1.next(state)

    ckpt = prod1.state_dict()

    # Create new producer and load state
    prod2 = PRODUCERS.build(cfg.producer)
    prod2.setup(rt)
    prod2.load_state_dict(ckpt)

    # Next 15 steps should produce identical decisions and buffer sizes
    for step in range(5, 20):
        state.step = step
        # Call step on both with same state
        s1 = prod1.rng.getstate()
        s2 = prod2.rng.getstate()
        assert s1 == s2
        b1 = prod1.next(state)
        b2 = prod2.next(state)
        assert len(prod1.buffer) == len(prod2.buffer)
        assert torch.equal(b1["student.input_ids"], b2["student.input_ids"])


def test_replay_producer(base_recipe):
    """ReplayProducer wraps another producer and returns replay batches when buffer fills."""
    base_recipe["producer"] = {
        "type": "replay",
        "inner": "dataset",
        "replay_prob": 1.0,
        "capacity": 50,
        "min_fill": 16,
        "save_buffer": True,
    }
    cfg = recipe_from_dict(base_recipe)
    rt = build_runtime(cfg)

    prod = rt.producer
    assert prod.steps_per_epoch() == prod.inner.steps_per_epoch()

    state = TrainState()
    # Step 1: inner produces 8 rows, buffer has 8 rows (< min_fill 16), returns inner batch
    b1 = prod.next(state)
    assert b1.batch_size == 8
    assert len(prod.buffer) == 8

    # Step 2: inner produces 8 rows, buffer reaches 16 rows (>= min_fill 16), returns replay batch
    b2 = prod.next(state)
    assert b2.batch_size == 8
    assert len(prod.buffer) == 16

    # Test state_dict roundtrip
    ckpt = prod.state_dict()
    prod2 = PRODUCERS.build(cfg.producer)
    prod2.setup(rt)
    prod2.load_state_dict(ckpt)
    assert len(prod2.buffer) == 16
