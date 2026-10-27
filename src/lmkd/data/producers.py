"""BatchProducer (Axis B, DESIGN §7.2): returns the batch for ONE optimizer step.

The batch has ``train.batch_size / world_size`` rows; the StepStrategy splits it into
micro-batches. Producers that generate data use ``rt.rollout`` (task T08/T09).
"""
from __future__ import annotations

import random
import warnings
from dataclasses import replace
from typing import Any

import torch
from pydantic import Field

from lmkd.core.batch import FIELDS, Batch, FieldSpec
from lmkd.core.plugin import BaseConfig, Plugin
from lmkd.core.registry import PRODUCERS, REWARDS, Registry
from lmkd.core.state import TrainState
from lmkd.data.packing import LengthGroupedSampler, MultipackBatchSampler, PackedCollator
from lmkd.data.replay import ReplayBuffer, batch_to_samples
from lmkd.rollout.base import SamplingSpec, prompts_from_batch, rollouts_to_samples, sampling_from_cfg

for _key in ("reward", "acc", "truncated"):
    FIELDS.register(FieldSpec(f"rollout.{_key}", "scalar", torch.float32))
FIELDS.register(FieldSpec("meta.feedback", "meta"))


class BatchProducer(Plugin):
    """Source of the global batch of each optimizer step (axis B). Must support ``state_dict`` for exact resume."""
    def setup(self, rt: Any) -> None:
        self.rt = rt

    def next(self, state: TrainState) -> Batch:
        raise NotImplementedError

    def on_eval(self, metrics: dict) -> None:
        return None

    def steps_per_epoch(self) -> int | None:
        return None


class EpochSampler(torch.utils.data.Sampler):
    """Deterministic, resumable, rank-aware sampler (K3).

    Order of epoch e = randperm(n, seed + e) (or arange); rank r takes ``perm[r::world]`` truncated
    to equal length. ``set_start(k)`` skips the first k indices of this rank (resume mid-epoch).
    """

    def __init__(self, n: int, *, shuffle: bool, seed: int, rank: int = 0, world_size: int = 1):
        self.n, self.shuffle, self.seed, self.rank, self.world = n, shuffle, seed, rank, world_size
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start(self, k: int) -> None:
        self.start = k

    def _indices(self) -> list[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(self.n, generator=g).tolist()
        else:
            perm = list(range(self.n))
        per_rank = self.n // self.world
        return perm[self.rank::self.world][:per_rank]

    def per_rank(self) -> int:
        return self.n // self.world

    def __iter__(self):
        return iter(self._indices()[self.start:])

    def __len__(self) -> int:
        return self.per_rank() - self.start


@PRODUCERS.register("dataset")
class DatasetProducer(BatchProducer):
    """Plain off-policy DataLoader over the prepared train dataset; loops over epochs.
    Deterministic and resumable: ``state_dict`` = {epoch, batches_done}."""

    class Config(BaseConfig):
        shuffle: bool = True
        drop_last: bool = True
        group_by_length: bool = False
        length_bucket_mult: int = 50

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        if rt.config.train.batch_size % rt.engine.world_size != 0:
            raise ValueError(f"train.batch_size ({rt.config.train.batch_size}) must be divisible by world_size ({rt.engine.world_size})")
        self.bs = rt.config.train.batch_size // rt.engine.world_size
        if self.bs % rt.config.train.micro_batch_size != 0:
            raise ValueError(f"per-rank batch size ({self.bs}) must be divisible by micro_batch_size ({rt.config.train.micro_batch_size})")
        self.sampler: LengthGroupedSampler | EpochSampler
        if self.cfg.group_by_length:
            lengths = [len(sample["student.input_ids"]) for sample in rt.train_dataset.samples]
            self.sampler = LengthGroupedSampler(
                lengths, batch_size=self.bs, seed=rt.config.train.seed,
                length_bucket_mult=self.cfg.length_bucket_mult, shuffle=self.cfg.shuffle,
                rank=rt.engine.rank, world_size=rt.engine.world_size,
            )
        else:
            self.sampler = EpochSampler(len(rt.train_dataset), shuffle=self.cfg.shuffle, seed=rt.config.train.seed,
                                        rank=rt.engine.rank, world_size=rt.engine.world_size)
        data_cfg = getattr(rt.config, "data", None)
        pin = getattr(data_cfg, "pin_memory", True) if data_cfg is not None else True
        is_cuda = getattr(getattr(rt.engine, "device", None), "type", None) == "cuda"
        pin = pin and is_cuda
        nw = getattr(data_cfg, "num_workers", 0) if data_cfg is not None else 0
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.bs,
            "sampler": self.sampler,
            "collate_fn": rt.collator,
            "drop_last": self.cfg.drop_last,
            "num_workers": nw,
            "pin_memory": pin,
            "generator": torch.Generator().manual_seed(rt.config.train.seed + rt.engine.rank),
        }
        if nw > 0:
            loader_kwargs["persistent_workers"] = True
        self.loader = torch.utils.data.DataLoader(rt.train_dataset, **loader_kwargs)
        per_rank = self.sampler.per_rank()
        self._spe = per_rank // self.bs if self.cfg.drop_last else -(-per_rank // self.bs)
        if self._spe == 0:
            raise ValueError(
                f"train dataset ({len(rt.train_dataset)} samples) smaller than per-device batch size {self.bs}"
            )
        self._it: Any = None
        self.epoch = 0
        self.batches_done = 0   # batches consumed in the current epoch

    def steps_per_epoch(self) -> int:
        return self._spe

    def next(self, state: TrainState) -> Batch:
        if self._it is None:
            self.sampler.set_epoch(self.epoch)
            self.sampler.set_start(self.batches_done * self.bs)
            self._it = iter(self.loader)
        try:
            batch = next(self._it)
        except StopIteration:
            self.epoch += 1
            self.batches_done = 0
            self._it = None
            return self.next(state)
        self.batches_done += 1
        state.epoch = self.epoch + self.batches_done / self._spe
        return batch

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "batches_done": self.batches_done}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state["epoch"])
        self.batches_done = int(state["batches_done"])
        self._it = None


@PRODUCERS.register("packed_dataset", source="docs/DESIGN.md:§11.5")
class PackedDatasetProducer(BatchProducer):
    """FFD multipacking with reset position IDs and FlashAttention varlen isolation.

    ``train.batch_size`` counts packed rows per optimizer step globally, and
    ``train.micro_batch_size`` counts packed rows per forward on one rank.
    With ``flash_attention_2`` only ``micro_batch_size: 1`` is accepted (see T02 report, "Future work").
    Packing efficiency = real tokens / (max_tokens × packed rows). Source: T02 §5.3–5.6.
    """

    class Config(BaseConfig):
        max_tokens: int = 8192
        lookahead: int = 10000
        shuffle: bool = True

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        if not rt.config.data.packing:
            raise ValueError("set data.packing: true to use packed_dataset")
        if (rt.config.models.student.attn_implementation != "flash_attention_2"
                and not rt.config.compat.get("allow_unsafe_packing", False)):
            raise ValueError("packed_dataset needs flash_attention_2; set models.student.attn_implementation")
        if (rt.config.models.student.attn_implementation == "flash_attention_2"
                and rt.config.train.micro_batch_size > 1):
            # HF only derives varlen from position_ids when the forward has a single row, so
            # several packed rows would attend across sample boundaries (see T02 report, "Future work").
            raise ValueError("packed_dataset with flash_attention_2 needs train.micro_batch_size: 1")
        teacher_name, _ = Registry.split_spec(rt.config.teacher.provider)
        input_name, _ = Registry.split_spec(rt.config.teacher.input)
        if teacher_name != "none" and input_name != "same":
            raise ValueError("packing with separate teacher inputs not supported")

        samples = rt.train_dataset.samples
        for sample in samples:
            for key in sample:
                if key.startswith("teacher."):
                    raise ValueError("packing with separate teacher inputs not supported")
                if not key.startswith("meta.") and FIELDS.get(key).kind == "ragged":
                    raise ValueError(f"packing does not support ragged field {key!r}")
        lengths = [len(sample["student.input_ids"]) for sample in samples]
        world_size = rt.engine.world_size
        if rt.config.train.batch_size % world_size:
            raise ValueError("train.batch_size must be divisible by world_size for packed_dataset")
        packs_per_step = rt.config.train.batch_size // world_size
        self.sampler = MultipackBatchSampler(
            lengths, capacity=self.cfg.max_tokens, packs_per_step=packs_per_step,
            seed=rt.config.train.seed, lookahead=self.cfg.lookahead, shuffle=self.cfg.shuffle,
            rank=rt.engine.rank, world_size=world_size,
        )
        self._spe = len(self.sampler)
        if self._spe == 0:
            raise ValueError("packed dataset is smaller than train.batch_size packed rows per step")
        self.collator = PackedCollator(rt.collator, self.cfg.max_tokens)
        self._it: Any = None
        self.epoch = 0
        self.steps_done = 0

    def steps_per_epoch(self) -> int:
        return self._spe

    def next(self, state: TrainState) -> Batch:
        while True:
            if self._it is None:
                self.sampler.set_epoch(self.epoch)
                self.sampler.set_start(self.steps_done)
                self._it = iter(self.sampler)
            try:
                bins = next(self._it)
                break
            except StopIteration:
                self.epoch += 1
                self.steps_done = 0
                self._it = None
        batch = self.collator([[self.rt.train_dataset[i] for i in bin_] for bin_ in bins])
        real_tokens = sum(len(self.rt.train_dataset.samples[i]["student.input_ids"])
                          for bin_ in bins for i in bin_)
        state.log("pack/efficiency", real_tokens / (self.cfg.max_tokens * len(bins)))
        self.steps_done += 1
        state.epoch = self.epoch + self.steps_done / self._spe
        return batch

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "steps_done": self.steps_done}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state["epoch"])
        self.steps_done = int(state["steps_done"])
        self.sampler.set_epoch(self.epoch)
        self.sampler.set_start(self.steps_done)
        self._it = None


def _truncate(sample: dict[str, Any], max_length: int) -> dict[str, Any]:
    """Truncate sample token fields to max_length while preserving full prompts longer than max_length - 1."""
    ids = sample.get("student.input_ids")
    if ids is None or len(ids) <= max_length:
        return sample
    plen = int(sample.get("student.prompt_len", 0))
    cutoff = max(max_length, plen)
    if len(ids) <= cutoff:
        return sample
    sample["student.input_ids"] = ids[:cutoff]
    if "student.attention_mask" in sample:
        sample["student.attention_mask"] = sample["student.attention_mask"][:cutoff]
    if "student.labels" in sample:
        sample["student.labels"] = sample["student.labels"][:cutoff]
    if "rollout.rollout_logp" in sample:
        sample["rollout.rollout_logp"] = sample["rollout.rollout_logp"][:cutoff]
    return sample


def make_on_policy_batch(rt: Any, src: Batch, spec: SamplingSpec) -> Batch:
    """Student generates 1 response per prompt of `src`; returns a collated Batch with the same
    meta.* keys (one row per prompt, order preserved)."""
    if getattr(rt, "rollout", None) is None:
        raise ValueError("producer needs 'rollout:' in recipe (T08)")
    prompts = prompts_from_batch(src, "student")
    rolls = rt.rollout.generate(prompts, replace(spec, n=1))
    meta = [{k: v[i] for k, v in src.meta.items() if k.startswith("meta.")} for i in range(len(prompts))]
    samples = rollouts_to_samples([r[0] for r in rolls], copy_meta=meta)
    max_len = getattr(getattr(rt.config, "data", None), "max_length", None)
    if max_len is not None:
        samples = [_truncate(s, max_len) for s in samples]
    return rt.collator(samples)


def sampling_threshold(thr: float, schedule: str, step: int, total: int) -> float:
    """Compute adaptive sampling threshold given schedule, global step, and total steps."""
    p = step / max(1, total)
    if schedule == "decreasing":
        return thr * (1.0 - p)
    if schedule == "constant":
        return thr * 0.5
    if schedule == "increasing":
        return thr * p
    raise ValueError(f"unknown schedule: {schedule!r}")


def decide(r: float, thr: float, samp_thr: float, buf_len: int, capacity: int) -> str:
    """Decide next batch source: 'generate' | 'replay' | 'data' (DistiLLM finetune.py:310-322)."""
    if r < samp_thr or (r < thr and buf_len < capacity):
        return "generate"
    if r < thr:
        return "replay"
    return "data"


@PRODUCERS.register("mixed", paper="GKD (Agarwal et al. 2024)")
class MixedProducer(DatasetProducer):
    """With prob ``lmbda`` replace the dataset batch by student rollouts (GKD; OPSD with lmbda=1)."""

    class Config(DatasetProducer.Config):
        lmbda: float = 0.5
        sampling: dict = {}
        seed_offset: int = 1000

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        # Global synchronized seed across ranks to avoid distributed deadlocks in next()
        seed = rt.config.train.seed + self.cfg.seed_offset
        self.rng = random.Random(seed)
        self.spec = sampling_from_cfg(self.cfg.sampling)
        self._has_max_new_tokens = "max_new_tokens" in self.cfg.sampling
        self._src: str = "data"

    def next(self, state: TrainState) -> Batch:
        b = super().next(state)
        if self.rng.random() < self.cfg.lmbda:
            self._src = "on_policy"
            state.log("producer/on_policy", 1.0)
            spec = self.spec
            if not self._has_max_new_tokens:
                prompts = prompts_from_batch(b, "student")
                max_p = max((len(p) for p in prompts), default=0)
                max_new = max(1, self.rt.config.data.max_length - max_p)
                spec = replace(spec, max_new_tokens=max_new)
            return make_on_policy_batch(self.rt, b, spec)
        self._src = "data"
        state.log("producer/on_policy", 0.0)
        return b

    def state_dict(self) -> dict:
        return {**super().state_dict(), "rng": self.rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if "rng" in state:
            self.rng.setstate(state["rng"])


@PRODUCERS.register("adaptive_sgo", source="white-box-distillation/same-tokenizer/finetune.py:266-322,413-419")
class AdaptiveSGOProducer(DatasetProducer):
    """DistiLLM adaptive student-generated output with replay buffer.

    Formula / algorithm:
      samp_threshold = thr * (1 - step/total_steps)  [decreasing]
      r ~ U(0, 1):
        if r < samp_threshold or (r < thr and len(buffer) < capacity): generate on-policy, add to buffer
        elif r < thr: replay from buffer
        else: use dataset batch

    Deviations from original white-box-distillation code:
      1. prev evaluation: original evaluates dev set once before training; here prev is initialized
         on the first on_eval call if not set initially.
      2. global_step offset: original had off-by-one in schedule step; here state.step (0-indexed completed steps) is used.
      3. label alignment: target-aligned convention avoids 1-token shift bug of original code.
      4. pad filtering: safe handling of pad==eos via rollout lengths rather than token value filtering.
    """

    class Config(DatasetProducer.Config):
        init_threshold: float = 0.0
        schedule: str = "decreasing"   # decreasing | constant | increasing
        threshold_step: float = 0.1
        max_threshold: float = 1.0
        loss_eps: float = 0.1
        capacity: int = 1000
        sampling: dict = {}
        dev_metric: str = "dev_loss/loss"
        save_buffer: bool = False
        seed_offset: int = 2000
        eval_step_zero: bool = True

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.thr: float = float(self.cfg.init_threshold)
        self.prev: float | None = None
        self._warned_dev_metric = False

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        self.thr = float(self.cfg.init_threshold)
        self.prev = None
        # Synchronized seed across ranks to prevent distributed deadlocks
        seed = rt.config.train.seed + self.cfg.seed_offset
        self.rng = random.Random(seed)
        self.buffer = ReplayBuffer(self.cfg.capacity, seed=seed + 1)
        self.spec = sampling_from_cfg(self.cfg.sampling)
        self._has_max_new_tokens = "max_new_tokens" in self.cfg.sampling
        self._warned_dev_metric = False

        if getattr(self.cfg, "eval_step_zero", True) and hasattr(rt, "evaluators"):
            for ev in getattr(rt, "evaluators", []):
                if getattr(ev, "name_prefix", "") == "dev_loss" or getattr(ev, "alias", "") == "dev_loss":
                    try:
                        metrics = ev.evaluate(rt)
                        self.on_eval(metrics)
                    except Exception:
                        pass
                    break

    def on_eval(self, metrics: dict) -> None:
        v = metrics.get(self.cfg.dev_metric)
        if v is None:
            for alt in ("dev_loss/loss", "eval/dev_loss", "eval/dev_loss/loss"):
                if alt in metrics:
                    v = metrics[alt]
                    break
        if v is None:
            if not self._warned_dev_metric:
                warnings.warn(
                    f"metric {self.cfg.dev_metric!r} not found in eval metrics: {sorted(metrics)}",
                    stacklevel=2,
                )
                self._warned_dev_metric = True
            return
        curr = float(v)
        if self.prev is None:
            self.prev = curr
            return
        if curr >= self.prev + self.cfg.loss_eps:
            self.thr = min(self.thr + self.cfg.threshold_step, self.cfg.max_threshold)
            self.prev = curr

    def next(self, state: TrainState) -> Batch:
        b = super().next(state)
        samp = sampling_threshold(self.thr, self.cfg.schedule, state.step, state.total_steps)
        r = self.rng.random()
        d = decide(r, self.thr, samp, len(self.buffer), self.cfg.capacity)

        if d == "generate":
            spec = self.spec
            if not self._has_max_new_tokens:
                prompts = prompts_from_batch(b, "student")
                max_p = max((len(p) for p in prompts), default=0)
                max_new = max(1, self.rt.config.data.max_length - max_p)
                spec = replace(spec, max_new_tokens=max_new)
            res_batch = make_on_policy_batch(self.rt, b, spec)
            self.buffer.add_samples(batch_to_samples(res_batch))
            source_code = 1.0
        elif d == "replay":
            if len(self.buffer) < self.bs:
                res_batch = b
                source_code = 0.0
                state.log("producer/replay_underflow", 1.0)
            else:
                res_batch = self.rt.collator(self.buffer.sample(self.bs))
                source_code = 2.0
        else:
            res_batch = b
            source_code = 0.0

        state.log("producer/source", source_code)
        state.log("producer/threshold", self.thr)
        state.log("producer/samp_threshold", samp)
        state.log("producer/buffer", float(len(self.buffer)))
        return res_batch

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            "thr": self.thr,
            "prev": self.prev,
            "rng": self.rng.getstate(),
            "buffer": self.buffer.state_dict(include_data=self.cfg.save_buffer),
        }

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.thr = float(state.get("thr", self.thr))
        self.prev = float(state["prev"]) if state.get("prev") is not None else None
        if "rng" in state:
            self.rng.setstate(state["rng"])
        if "buffer" in state and isinstance(state["buffer"], dict):
            self.buffer.load_state_dict(state["buffer"])


@PRODUCERS.register("replay")
class ReplayProducer(BatchProducer):
    """Wraps another producer: with prob `replay_prob` returns a batch sampled from a buffer of past
    batches produced by the inner producer (all rows of every inner batch are added)."""

    class Config(BaseConfig):
        inner: dict | str = "dataset"
        replay_prob: float = 0.5
        capacity: int = 4096
        min_fill: int = 0
        save_buffer: bool = False
        seed_offset: int = 3000

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        self.inner = PRODUCERS.build(self.cfg.inner)
        self.inner.setup(rt)
        self.bs = getattr(self.inner, "bs", rt.config.train.batch_size // rt.engine.world_size)
        rank = rt.engine.rank
        seed = rt.config.train.seed + self.cfg.seed_offset + rank
        self.rng = random.Random(seed)
        self.buffer = ReplayBuffer(self.cfg.capacity, seed=seed + 1)

    def steps_per_epoch(self) -> int | None:
        return self.inner.steps_per_epoch()

    def on_eval(self, metrics: dict) -> None:
        self.inner.on_eval(metrics)

    def next(self, state: TrainState) -> Batch:
        b = self.inner.next(state)
        self.buffer.add_samples(batch_to_samples(b))
        threshold = max(self.cfg.min_fill, self.bs)
        if len(self.buffer) >= threshold and self.rng.random() < self.cfg.replay_prob:
            state.log("producer/replay", 1.0)
            return self.rt.collator(self.buffer.sample(self.bs))
        state.log("producer/replay", 0.0)
        return b

    def state_dict(self) -> dict:
        return {
            "inner": self.inner.state_dict(),
            "rng": self.rng.getstate(),
            "buffer": self.buffer.state_dict(include_data=self.cfg.save_buffer),
        }

    def load_state_dict(self, state: dict) -> None:
        if "inner" in state and hasattr(self.inner, "load_state_dict"):
            self.inner.load_state_dict(state["inner"])
        if "rng" in state:
            self.rng.setstate(state["rng"])
        if "buffer" in state and isinstance(state["buffer"], dict):
            self.buffer.load_state_dict(state["buffer"])


@PRODUCERS.register("rollout_group", source="Self-Distill/SDPO-main/verl/trainer/ppo/ray_trainer.py")
class RolloutGroupProducer(DatasetProducer):
    """Generate contiguous groups of n responses, then verify the complete rollout batch once."""
    requires = ("student.input_ids", "student.prompt_len", "student.attention_mask")
    provides = ("student.labels", "group.uid", "group.index", "rollout.reward", "rollout.acc",
                "rollout.truncated", "meta.feedback", "meta.response_text", "rollout.rollout_logp")

    class Config(DatasetProducer.Config):
        n: int = Field(default=8, ge=1)
        sampling: dict = {}
        verifier: dict | str = "math_verify"
        drop_all_same: bool = False

    def setup(self, rt: Any) -> None:
        BatchProducer.setup(self, rt)
        total = rt.config.train.batch_size
        world = rt.engine.world_size
        if total % (world * self.cfg.n):
            raise ValueError("train.batch_size must be divisible by world_size * rollout_group.n")
        if rt.rollout is None:
            raise ValueError("rollout_group requires a rollout engine")
        if self.cfg.group_by_length:
            raise ValueError("rollout_group does not support group_by_length")
        self.prompts_per_step = self.bs = total // world // self.cfg.n
        self.sampler = EpochSampler(len(rt.train_dataset), shuffle=self.cfg.shuffle, seed=rt.config.train.seed,
                                    rank=rt.engine.rank, world_size=world)
        data_cfg = getattr(rt.config, "data", None)
        pin = getattr(data_cfg, "pin_memory", True) if data_cfg is not None else True
        is_cuda = getattr(getattr(rt.engine, "device", None), "type", None) == "cuda"
        pin = pin and is_cuda
        nw = getattr(data_cfg, "num_workers", 0) if data_cfg is not None else 0
        loader_kwargs = {
            "batch_size": self.bs,
            "sampler": self.sampler,
            "collate_fn": rt.collator,
            "drop_last": self.cfg.drop_last,
            "num_workers": nw,
            "pin_memory": pin,
            "generator": torch.Generator().manual_seed(rt.config.train.seed + rt.engine.rank),
        }
        if nw > 0:
            loader_kwargs["persistent_workers"] = True
        self.loader = torch.utils.data.DataLoader(rt.train_dataset, **loader_kwargs)
        count = self.sampler.per_rank()
        self._spe = count // self.bs if self.cfg.drop_last else -(-count // self.bs)
        if self._spe == 0:
            raise ValueError("train dataset is smaller than the prompt batch")
        self._it = None
        self.epoch = self.batches_done = 0
        self.verifier = REWARDS.build(self.cfg.verifier)
        self.verifier.setup(rt)

    def next(self, state: TrainState) -> Batch:
        prompts = DatasetProducer.next(self, state)
        spec = replace(sampling_from_cfg(self.cfg.sampling), n=self.cfg.n)
        rolls = self.rt.rollout.generate(prompts_from_batch(prompts), spec)
        if len(rolls) != prompts.batch_size or any(len(group) != self.cfg.n for group in rolls):
            raise ValueError("rollout engine returned an incorrect group size")
        samples, metas, texts = [], [], []
        for i, group in enumerate(rolls):
            meta = {k: v[i] for k, v in prompts.meta.items()}
            uid = f"{meta.get('meta.id', i)}@{state.step}"
            for j, roll in enumerate(group):
                sample = rollouts_to_samples([roll], copy_meta=[meta])[0]
                sample.update({"group.uid": uid, "group.index": j, "meta.response_text": roll.text,
                               "rollout.truncated": float(roll.finish_reason == "length")})
                row = {k.removeprefix("meta."): v for k, v in meta.items()}
                row["truncated"] = bool(sample["rollout.truncated"])
                samples.append(sample)
                metas.append(row)
                texts.append(roll.text)
        results = self.verifier([m.get("prompt_text", "") for m in metas], texts, metas)
        if len(results) != len(samples):
            raise ValueError("verifier must return one result per rollout")
        for sample, result in zip(samples, results):
            sample.update({"rollout.reward": float(result["score"]), "rollout.acc": float(result["acc"]),
                           "meta.feedback": str(result["feedback"])})
        batch = self.rt.collator(samples)
        for key in ("reward", "acc", "truncated"):
            metric = "truncated_frac" if key == "truncated" else f"{key}_mean"
            state.log(f"rl/{metric}", batch[f"rollout.{key}"].mean())
        state.log("rl/response_len_mean", (batch["student.attention_mask"].sum(-1) - batch["student.prompt_len"]).float().mean())
        if self.cfg.drop_all_same:
            # Keep row count and collective calls aligned across ranks; excluded groups contribute zero.
            rewards = batch["rollout.reward"].reshape(-1, self.cfg.n)
            active = rewards.max(-1).values.ne(rewards.min(-1).values).repeat_interleave(self.cfg.n)
            batch["seq.loss_mask"] = active.float()[:, None].expand_as(batch["student.labels"]).clone()
        return batch
