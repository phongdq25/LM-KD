"""Prompt-group producer that flattens listwise members into contiguous rows."""
from __future__ import annotations

from typing import Any

import torch

from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import PRODUCERS
from lmkd.data.producers import DatasetProducer, EpochSampler
from lmkd.data.transforms.listwise import tokenize_fingerprint


def _zero_like(value: Any, *, key: str) -> Any:
    if isinstance(value, list):
        return [_zero_like(item, key=key) for item in value]
    if key == "cache.adv_ids":
        return -1
    return 0


@PRODUCERS.register("listwise", source="integrations/preference_distillation.py:307-363")
class ListwiseProducer(DatasetProducer):
    """Flatten each prompt's N responses and preserve whole groups across micro-batches.

    Source: human_alignment/integrations/preference_distillation.py:307-363.
    """

    provides = ("student.input_ids", "student.attention_mask", "student.labels", "student.prompt_len",
                "group.index", "group.size", "group.valid", "group.uid", "pref.teacher_score")

    class Config(BaseConfig):
        shuffle: bool = True
        pad_groups_to: int | None = None

    def _flatten(self, samples: list[dict]) -> Any:
        rows = []
        for sample in samples:
            members = sample["group.members"]
            if len(members) > self.n:
                raise ValueError("response group exceeds configured pad_groups_to")
            for i, member in enumerate(members):
                rows.append({**member, "group.index": i, "group.size": self.n})
            for i in range(len(members), self.n):
                row = dict(members[0])
                row["group.index"] = i
                row["group.size"] = self.n
                row["group.valid"] = False
                row["student.labels"] = [-100] * len(row["student.input_ids"])
                row["pref.teacher_score"] = float("nan")
                row["meta.id"] = f"{sample['meta.id']}:{i}:pad"
                if "meta.cache_tokenize_fp" in row:
                    row["meta.cache_tokenize_fp"] = tokenize_fingerprint(
                        row["student.input_ids"], row["student.labels"],
                    )
                for key in row:
                    if key.startswith("cache."):
                        row[key] = _zero_like(row[key], key=key)
                rows.append(row)
        return self.rt.collator(rows)

    def setup(self, rt: Any) -> None:
        self.rt = rt
        samples = rt.train_dataset.samples
        if not samples:
            raise ValueError("listwise producer needs nonempty training data")
        sizes = [len(sample["group.members"]) for sample in samples]
        if any(size < 2 for size in sizes):
            raise ValueError("each listwise prompt needs at least two responses")
        self.n = self.cfg.pad_groups_to or max(sizes)
        if self.n < max(sizes):
            raise ValueError("pad_groups_to is smaller than a response group")
        world = rt.engine.world_size
        global_bs = rt.config.train.batch_size
        micro = rt.config.train.micro_batch_size
        if global_bs % (self.n * world):
            raise ValueError(f"train.batch_size must be a multiple of group size {self.n} x world_size {world}")
        if micro % self.n:
            raise ValueError(f"train.micro_batch_size must be a multiple of group size {self.n}")
        self.bs = global_bs // (self.n * world)
        self.sampler = EpochSampler(len(samples), shuffle=self.cfg.shuffle, seed=rt.config.train.seed,
                                    rank=rt.engine.rank, world_size=world)
        data_cfg = getattr(rt.config, "data", None)
        pin = getattr(data_cfg, "pin_memory", True) if data_cfg is not None else True
        is_cuda = getattr(getattr(rt.engine, "device", None), "type", None) == "cuda"
        pin = pin and is_cuda
        nw = getattr(data_cfg, "num_workers", 0) if data_cfg is not None else 0
        loader_kwargs = {
            "batch_size": self.bs,
            "sampler": self.sampler,
            "collate_fn": self._flatten,
            "drop_last": True,
            "num_workers": nw,
            "pin_memory": pin,
            "generator": torch.Generator().manual_seed(rt.config.train.seed + rt.engine.rank),
        }
        if nw > 0:
            loader_kwargs["persistent_workers"] = True
        self.loader = torch.utils.data.DataLoader(rt.train_dataset, **loader_kwargs)
        self._spe = self.sampler.per_rank() // self.bs
        if self._spe == 0:
            raise ValueError("train dataset is smaller than one listwise step")
        self._it = None
        self.epoch = 0
        self.batches_done = 0
