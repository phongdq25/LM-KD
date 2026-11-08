"""Epoch-level student rollout producer for DistiLLM-2.

Paper: "DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs" (ICML 2025 Oral).
As proposed in the paper, the student model generates completions for training prompts
at the beginning of each epoch. During the epoch, training proceeds efficiently on paired
demonstrations (chosen teacher response y^+ and rejected student completion y^-).
At each epoch boundary, the updated student model regenerates completions for the next epoch.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader, Dataset

from lmkd.core.batch import Batch
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import PRODUCERS
from lmkd.core.state import TrainState
from lmkd.data.producers import (
    BatchProducer,
    DatasetProducer,
    EpochSampler,
    rollouts_to_samples,
    sampling_from_cfg,
)


class _ListDataset(Dataset):
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Any:
        return self.items[idx]


@PRODUCERS.register("distillm2", source="baselines.same_tokenizer.distillm2:DistiLLM2Producer")
class DistiLLM2Producer(BatchProducer):
    """Epoch-level student generation producer for DistiLLM-2."""

    provides: ClassVar[tuple[str, ...]] = (
        *DatasetProducer.provides,
        "group.size",
        "group.index",
    )

    class Config(BaseConfig):
        sampling: dict[str, Any] = {}
        drop_last: bool = True

    def setup(self, rt: Any) -> None:
        super().setup(rt)
        if getattr(rt, "rollout", None) is None:
            raise ValueError("DistiLLM-2 producer requires a 'rollout:' engine in recipe")

        total_bs = rt.config.train.batch_size
        world = rt.engine.world_size
        if (total_bs // world) % 2 != 0:
            raise ValueError("train.batch_size per worker must be even for pairwise training")

        self.bs = total_bs // world
        self.pairs_per_batch = self.bs // 2
        self.rt = rt
        self.spec = sampling_from_cfg({**self.cfg.sampling, "n": 1})
        self.epoch = 0
        self.batches_done = 0
        self._it: Any = None
        self._loader: DataLoader | None = None

        max_len = getattr(getattr(self.rt.config, "data", None), "max_length", None)
        raw_samples = rt.train_dataset.samples if hasattr(rt.train_dataset, "samples") else list(rt.train_dataset)

        self.prompts: list[list[int]] = []
        self.metas: list[dict[str, Any]] = []
        self.chosen_samples: list[dict[str, Any]] = []

        for s in raw_samples:
            c = copy.deepcopy(s)
            plen = c.get("student.prompt_len", len(c.get("student.input_ids", [])))
            self.prompts.append(list(c["student.input_ids"][:plen]))
            self.metas.append({k: v for k, v in c.items() if k.startswith("meta.")})

            if max_len is not None:
                for k in ("student.input_ids", "student.labels", "student.attention_mask"):
                    if k in c and len(c[k]) > max_len:
                        c[k] = c[k][:max_len]
            c["group.size"] = 2
            c["group.index"] = 0
            self.chosen_samples.append(c)

        num_pairs = len(self.chosen_samples)
        per_rank_pairs = num_pairs // world
        self._spe = (
            per_rank_pairs // self.pairs_per_batch if self.cfg.drop_last else -(-per_rank_pairs // self.pairs_per_batch)
        )
        self._spe = max(1, self._spe)

    def _sync_weights_if_needed(self) -> None:
        if hasattr(self.rt, "rollout") and self.rt.rollout is not None:
            mark = getattr(self.rt.rollout, "mark_stale", None)
            if callable(mark):
                mark()

    def _generate_epoch_dataset(self) -> None:
        self._sync_weights_if_needed()

        rolls = self.rt.rollout.generate(self.prompts, self.spec)
        rejected = rollouts_to_samples([r[0] for r in rolls], copy_meta=self.metas)

        max_len = getattr(getattr(self.rt.config, "data", None), "max_length", None)
        for r in rejected:
            if max_len is not None:
                for k in ("student.input_ids", "student.labels", "student.attention_mask"):
                    if k in r and len(r[k]) > max_len:
                        r[k] = r[k][:max_len]
            r["group.size"] = 2
            r["group.index"] = 1

        pairs = list(zip(self.chosen_samples, rejected))
        sampler = EpochSampler(
            len(pairs),
            shuffle=True,
            seed=self.rt.config.train.seed,
            rank=self.rt.engine.rank,
            world_size=self.rt.engine.world_size,
        )
        sampler.set_epoch(self.epoch)
        sampler.set_start(self.batches_done * self.pairs_per_batch)
        self._sampler = sampler

        def _collate_pairs(batch_pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> Batch:
            flattened = []
            for c, r in batch_pairs:
                flattened.extend((c, r))
            return self.rt.collator(flattened)

        data_cfg = getattr(self.rt.config, "data", None)
        pin = getattr(data_cfg, "pin_memory", True) if data_cfg is not None else True
        is_cuda = getattr(getattr(self.rt.engine, "device", None), "type", None) == "cuda"
        nw = getattr(data_cfg, "num_workers", 0) if data_cfg is not None else 0

        self._loader = DataLoader(
            _ListDataset(pairs),
            batch_size=self.pairs_per_batch,
            sampler=sampler,
            collate_fn=_collate_pairs,
            drop_last=self.cfg.drop_last,
            num_workers=nw,
            pin_memory=pin and is_cuda,
            generator=torch.Generator().manual_seed(self.rt.config.train.seed + self.rt.engine.rank + self.epoch),
        )
        self._it = iter(self._loader)

    def steps_per_epoch(self) -> int:
        return self._spe

    def next(self, state: TrainState) -> Batch:
        if self._it is None:
            self._generate_epoch_dataset()

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

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "batches_done": self.batches_done}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.batches_done = int(state.get("batches_done", 0))
        self._it = None


DistiLLM2EpochProducer = DistiLLM2Producer
