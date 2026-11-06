"""Packing layout, sampler determinism, and packed training contracts."""

from __future__ import annotations

import random

import pytest
import torch

from lmkd.core.batch import FIELDS, Batch, FieldSpec
from lmkd.core.config import recipe_from_dict
from lmkd.data.collator import Collator
from lmkd.data.labels import make_labels
from lmkd.data.packing import LengthGroupedSampler, MultipackBatchSampler, ffd_bins, pack_samples
from lmkd.data.producers import EpochSampler
from lmkd.models.forward import side_inputs
from lmkd.train.builder import build_runtime
from lmkd.train.step_strategies import count_supervised


def _sample(ids: list[int], name: str) -> dict:
    return {
        "student.input_ids": ids,
        "student.attention_mask": [1] * len(ids),
        "student.labels": make_labels(ids, 1).tolist(),
        "student.prompt_len": 1,
        "seq.loss_mask": [0.0] + [1.0] * (len(ids) - 1),
        "meta.id": name,
    }


def test_pack_samples_layout():
    samples = [_sample([5, 6, 7], "a"), _sample([8, 9], "b")]
    packed = pack_samples(samples, max_tokens=5)
    assert packed["student.input_ids"] == [5, 6, 7, 8, 9]
    assert packed["student.position_ids"] == [0, 1, 2, 0, 1]
    assert packed["student.attention_mask"] == [1, 1, 1, 1, 1]
    assert packed["pack.seq_id"] == [0, 0, 0, 1, 1]
    assert packed["student.labels"] == [-100, 6, 7, -100, 9]
    assert packed["seq.loss_mask"] == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert packed["pack.num_seqs"] == 2
    assert packed["meta.id"] == ["a", "b"]
    assert "student.prompt_len" not in packed
    assert samples[1]["student.input_ids"] == [8, 9]  # input remains unchanged

    collator = Collator({"student": 0}, pad_to_multiple_of=None)
    unpacked_batch = collator(samples)
    packed_batch = collator([packed])
    unpacked_tok = count_supervised(unpacked_batch)[0]
    packed_tok, packed_seq = count_supervised(packed_batch)
    assert packed_tok == unpacked_tok == 3  # same global Norms.z_tok
    assert packed_seq == count_supervised(unpacked_batch)[1] == 2


def test_pack_too_long_raises():
    with pytest.raises(ValueError, match="max_tokens"):
        pack_samples([_sample([5, 6, 7], "a"), _sample([8, 9], "b")], max_tokens=4)


def test_pack_token_k_and_reject_ragged():
    FIELDS.register(FieldSpec("seq.test_token_k", "token_k", torch.long, -1, "student"))
    FIELDS.register(FieldSpec("align.test_ragged", "ragged", torch.long))
    samples = [_sample([5, 6], "a"), _sample([7, 8, 9], "b")]
    samples[0]["seq.test_token_k"] = [[1, 2], [3, 4]]
    samples[1]["seq.test_token_k"] = [[5, 6], [7, 8], [9, 10]]
    packed = pack_samples(samples, max_tokens=5)
    assert packed["seq.test_token_k"] == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    assert Collator({"student": 0}, pad_to_multiple_of=None)([packed])["seq.test_token_k"].shape == (1, 5, 2)
    for sample in samples:
        sample["align.test_ragged"] = [[1], [2]]
    with pytest.raises(ValueError, match="ragged"):
        pack_samples(samples, max_tokens=5)


def test_ffd_bins():
    lengths = [5, 4, 3, 3, 1]
    bins = ffd_bins(lengths, capacity=8)
    assert len(bins) == 2
    assert sorted(i for bin_ in bins for i in bin_) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in bin_) <= 8 for bin_ in bins)
    with pytest.raises(ValueError, match="capacity"):
        ffd_bins([9], capacity=8)


def test_multipack_deterministic_and_sharded():
    lengths = [2, 5, 3, 4, 6, 2, 4, 3] * 4
    kw = dict(capacity=8, packs_per_step=2, seed=17, lookahead=8)
    one = MultipackBatchSampler(lengths, **kw)
    two = MultipackBatchSampler(lengths, **kw)
    one.set_epoch(2)
    two.set_epoch(2)
    assert list(one) == list(two)
    assert len(one) > 0
    assert len(list(one)) == len(list(two))

    rank0 = MultipackBatchSampler(lengths, rank=0, world_size=2, **kw)
    rank1 = MultipackBatchSampler(lengths, rank=1, world_size=2, **kw)
    rank0.set_epoch(2)
    rank1.set_epoch(2)
    steps0, steps1 = list(rank0), list(rank1)
    assert len(steps0) == len(steps1)
    assert {i for step in steps0 for bin_ in step for i in bin_}.isdisjoint(
        {i for step in steps1 for bin_ in step for i in bin_}
    )
    rank0.set_start(1)
    assert list(rank0) == steps0[1:]


def test_packed_producer_resume(base_recipe):
    base_recipe["data"]["packing"] = True
    base_recipe["compat"] = {"allow_unsafe_packing": True}
    base_recipe["producer"] = {"type": "packed_dataset", "max_tokens": 32, "lookahead": 12}
    base_recipe["train"].update(batch_size=2, micro_batch_size=1)
    cfg = recipe_from_dict(base_recipe)
    rt_a = build_runtime(cfg)
    first = rt_a.producer.next(rt_a.state)
    assert rt_a.state.extra["metrics"]["pack/efficiency"] == pytest.approx(
        float(first["student.attention_mask"].sum()) / (32 * first.batch_size)
    )
    saved = rt_a.producer.state_dict()
    expected = [rt_a.producer.next(rt_a.state)["student.input_ids"] for _ in range(2)]
    assert "pack/efficiency" in rt_a.state.extra["metrics"]
    rt_b = build_runtime(cfg)
    rt_b.producer.load_state_dict(saved)
    actual = [rt_b.producer.next(rt_b.state)["student.input_ids"] for _ in range(2)]
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert rt_b.producer.state_dict() == rt_a.producer.state_dict()
    assert rt_b.state.epoch == pytest.approx(rt_a.state.epoch)


def test_packed_producer_resume_at_epoch_boundary(base_recipe):
    base_recipe["data"]["packing"] = True
    base_recipe["compat"] = {"allow_unsafe_packing": True}
    base_recipe["producer"] = {"type": "packed_dataset", "max_tokens": 32, "lookahead": 12}
    base_recipe["train"].update(batch_size=2, micro_batch_size=1)
    cfg = recipe_from_dict(base_recipe)
    rt_a = build_runtime(cfg)
    for _ in range(rt_a.producer.steps_per_epoch()):
        rt_a.producer.next(rt_a.state)
    saved = rt_a.producer.state_dict()
    expected = rt_a.producer.next(rt_a.state)["student.input_ids"]
    rt_b = build_runtime(cfg)
    rt_b.producer.load_state_dict(saved)
    actual = rt_b.producer.next(rt_b.state)["student.input_ids"]
    assert torch.equal(expected, actual)
    assert rt_b.producer.state_dict() == rt_a.producer.state_dict()


def test_packed_requires_packing_flag(base_recipe):
    base_recipe["producer"] = {"type": "packed_dataset", "max_tokens": 32}
    with pytest.raises(ValueError, match="data.packing: true"):
        build_runtime(recipe_from_dict(base_recipe))


def test_packed_requires_safe_attention(base_recipe):
    base_recipe["data"]["packing"] = True
    base_recipe["producer"] = {"type": "packed_dataset", "max_tokens": 32}
    with pytest.raises(ValueError, match="flash_attention_2"):
        build_runtime(recipe_from_dict(base_recipe))


def test_packed_flash_rejects_multi_row_micro_batch(base_recipe):
    # T02 report: HF flash-attn varlen only isolates samples when the forward
    # has a single row, so safe packing must use micro_batch_size == 1 (see "Future work" in the report).
    from types import SimpleNamespace

    from lmkd.data.producers import PackedDatasetProducer

    base_recipe["data"]["packing"] = True
    base_recipe["models"]["student"]["attn_implementation"] = "flash_attention_2"
    base_recipe["train"]["batch_size"] = 4
    base_recipe["train"]["micro_batch_size"] = 2
    base_recipe["producer"] = {"type": "packed_dataset", "max_tokens": 32}
    rt = SimpleNamespace(config=recipe_from_dict(base_recipe))
    producer = PackedDatasetProducer(PackedDatasetProducer.Config(max_tokens=32))
    with pytest.raises(ValueError, match="micro_batch_size"):
        producer.setup(rt)


def test_packed_rejects_separate_teacher_inputs(base_recipe, tiny_models):
    base_recipe["data"]["packing"] = True
    base_recipe["compat"] = {"allow_unsafe_packing": True}
    base_recipe["producer"] = {"type": "packed_dataset", "max_tokens": 32}
    base_recipe["models"]["teacher"] = {"path": tiny_models["teacher"], "dtype": "fp32"}
    base_recipe["teacher"] = {"provider": "hf_frozen", "input": "pretokenized"}
    with pytest.raises(ValueError, match="packing with separate teacher inputs not supported"):
        build_runtime(recipe_from_dict(base_recipe))


def test_length_grouped_resume_and_ranks():
    lengths = [((i * 17) % 29) + 1 for i in range(64)]
    kw = dict(batch_size=4, seed=23, length_bucket_mult=4)
    rank0 = LengthGroupedSampler(lengths, rank=0, world_size=2, **kw)
    rank1 = LengthGroupedSampler(lengths, rank=1, world_size=2, **kw)
    rank0.set_epoch(3)
    rank1.set_epoch(3)
    indices0, indices1 = list(rank0), list(rank1)
    assert len(indices0) == len(indices1) == rank0.per_rank()
    assert set(indices0).isdisjoint(indices1)
    rank0.set_start(5)
    assert list(rank0) == indices0[5:]
    assert len(rank0) == len(indices0) - 5


def test_length_grouped_reduces_padding():
    lengths = [random.Random(i).randint(1, 100) for i in range(64)]
    bs = 4
    ordinary = list(EpochSampler(len(lengths), shuffle=True, seed=7))
    grouped = list(LengthGroupedSampler(lengths, batch_size=bs, seed=7, length_bucket_mult=8))

    def padding(indices):
        return sum(
            bs * max(lengths[i] for i in indices[j : j + bs]) - sum(lengths[i] for i in indices[j : j + bs])
            for j in range(0, len(indices), bs)
        )

    assert padding(grouped) < padding(ordinary)


def test_dataset_length_grouped_producer_resume(base_recipe):
    base_recipe["producer"] = {"type": "dataset", "group_by_length": True, "length_bucket_mult": 4}
    base_recipe["train"].update(batch_size=4, micro_batch_size=2)
    cfg = recipe_from_dict(base_recipe)
    rt_a = build_runtime(cfg)
    assert isinstance(rt_a.producer.sampler, LengthGroupedSampler)
    rt_a.producer.next(rt_a.state)
    saved = rt_a.producer.state_dict()
    expected = [rt_a.producer.next(rt_a.state)["student.input_ids"] for _ in range(2)]
    rt_b = build_runtime(cfg)
    rt_b.producer.load_state_dict(saved)
    actual = [rt_b.producer.next(rt_b.state)["student.input_ids"] for _ in range(2)]
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_count_supervised_packed():
    b = Batch(
        {
            "student.labels": torch.tensor([[-100, 2, 3], [-100, 4, -100]]),
            "pack.num_seqs": torch.tensor([3, 2]),
        }
    )
    assert count_supervised(b) == (3.0, 5.0)


def test_side_inputs_drops_mask_when_packed():
    b = Batch(
        {
            "student.input_ids": torch.tensor([[5, 6, 7]]),
            "student.attention_mask": torch.ones((1, 3), dtype=torch.long),
            "student.position_ids": torch.tensor([[0, 1, 0]]),
            "pack.seq_id": torch.tensor([[0, 0, 1]]),
        }
    )
    inputs = side_inputs(b, "student")
    assert "attention_mask" not in inputs
    assert torch.equal(inputs["position_ids"], b["student.position_ids"])
