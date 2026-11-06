"""LUFFY: Learning to Reason Under Off-Policy Guidance - Rollout Group Producer.

Paper: LUFFY: Learning to Reason Under Off-Policy Guidance (NeurIPS 2025)
Source: LUFFY/luffy/verl/verl/mix_src/mix_vllm_rollout.py:200-290 & mix_trainer.py:490-538
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import ClassVar

import torch
from pydantic import Field

from lmkd.core.batch import FIELDS, Batch, FieldSpec
from lmkd.core.registry import PRODUCERS
from lmkd.core.state import TrainState
from lmkd.data.producers import (
    DatasetProducer,
    RolloutGroupProducer,
    prompts_from_batch,
    rollouts_to_samples,
    sampling_from_cfg,
)
from lmkd.rollout.base import Rollout

FIELDS.register(
    FieldSpec(
        "rollout.is_off_policy",
        "scalar",
        torch.float32,
        0.0,
        doc="1.0 for off-policy demonstrations, 0.0 for on-policy rollouts",
    )
)
FIELDS.register(
    FieldSpec(
        "rollout.prefix_mask",
        "token",
        torch.bool,
        False,
        "student",
        doc="1 for off-policy prefix tokens, 0 for on-policy tokens",
    )
)


@PRODUCERS.register("luffy_group", source="LUFFY/luffy/verl/verl/mix_src/mix_vllm_rollout.py:200-290", force=True)
@PRODUCERS.register("luffy_rollout", source="LUFFY/luffy/verl/verl/mix_src/mix_vllm_rollout.py:200-290", force=True)
class LUFFYGroupProducer(RolloutGroupProducer):
    """Produces mixed on-policy and off-policy rollout groups for LUFFY training.

    For each prompt, generates `n` responses. When off-policy demonstration traces
    are present in the dataset (e.g. `target` or `solution`), `n_prefix` responses
    are designated as off-policy demonstrations, setting `rollout.prefix_mask` accordingly.
    The remaining `n - n_prefix` samples are standard on-policy rollouts.
    """

    provides: ClassVar[tuple[str, ...]] = (
        *RolloutGroupProducer.provides,
        "rollout.is_off_policy",
        "rollout.prefix_mask",
    )

    class Config(RolloutGroupProducer.Config):
        n_prefix: int = Field(default=1, ge=0)
        min_prefix_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
        max_prefix_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
        target_field: str = "target"
        prefix_strategy: str = "random"

    def _encode_target(self, target_text: str, prompt_len: int, max_new_tokens: int, fallback: list[int]) -> list[int]:
        tok = getattr(self.rt, "student_tokenizer", None)
        if tok is not None:
            raw_ids = (
                tok.encode(target_text, add_special_tokens=False)
                if hasattr(tok, "encode")
                else tok(target_text, add_special_tokens=False)["input_ids"]
            )
            target_ids = list(raw_ids)
            eos_id = getattr(tok, "eos_token_id", None)
            if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                target_ids.append(int(eos_id))
        else:
            target_ids = list(fallback)

        max_len = getattr(getattr(self.rt.config, "data", None), "max_length", None)
        max_resp = max_new_tokens
        if max_len is not None:
            max_resp = min(max_resp, max(1, max_len - prompt_len))
        target_ids = target_ids[:max_resp]
        return target_ids if target_ids else list(fallback)

    def next(self, state: TrainState) -> Batch:
        prompts = DatasetProducer.next(self, state)
        spec = replace(sampling_from_cfg(self.cfg.sampling), n=self.cfg.n)
        rolls = self.rt.rollout.generate(prompts_from_batch(prompts), spec)
        if len(rolls) != prompts.batch_size or any(len(group) != self.cfg.n for group in rolls):
            raise ValueError("rollout engine returned an incorrect group size")

        tok = getattr(self.rt, "student_tokenizer", None)
        base_seed = int(getattr(getattr(self.rt.config, "train", None), "seed", 42))
        samples, metas, texts = [], [], []
        for i, group in enumerate(rolls):
            meta = {k: v[i] for k, v in prompts.meta.items()}
            uid = f"{meta.get('meta.id', i)}@{state.step}"
            raw_target: str | None = None
            for cand in (
                self.cfg.target_field,
                f"meta.{self.cfg.target_field}",
                "solution",
                "meta.solution",
                "response",
                "meta.response",
            ):
                if cand in meta and meta[cand]:
                    raw_target = str(meta[cand])
                    break
            has_target = raw_target is not None and bool(raw_target)
            target_text = raw_target or ""

            for j, roll in enumerate(group):
                is_off_policy = (j < self.cfg.n_prefix) and has_target
                if is_off_policy:
                    if self.cfg.min_prefix_ratio >= self.cfg.max_prefix_ratio:
                        ratio = float(self.cfg.min_prefix_ratio)
                    else:
                        rng = random.Random(base_seed + state.step * 1000 + i * 100 + j)
                        ratio = rng.uniform(self.cfg.min_prefix_ratio, self.cfg.max_prefix_ratio)
                    if ratio <= 0.0:
                        is_off_policy = False

                if is_off_policy:
                    plen = len(roll.prompt_ids)
                    target_ids = self._encode_target(target_text, plen, spec.max_new_tokens, list(roll.response_ids))
                    if ratio >= 1.0:
                        resp_ids = target_ids
                        resp_text = target_text
                        resp_logp = [0.0] * len(resp_ids) if roll.response_logp is not None else None
                        resp_roll = Rollout(
                            prompt_ids=list(roll.prompt_ids),
                            response_ids=resp_ids,
                            response_logp=resp_logp,
                            finish_reason="stop",
                            text=resp_text,
                        )
                        sample_prefix_mask = [False] * plen + [True] * len(resp_ids)
                    else:
                        prefix_len = max(1, int(len(target_ids) * ratio))
                        prefix_ids = target_ids[:prefix_len]
                        max_len = getattr(getattr(self.rt.config, "data", None), "max_length", None)
                        max_resp = spec.max_new_tokens
                        if max_len is not None:
                            max_resp = min(max_resp, max(1, max_len - plen))
                        rem_budget = max(0, max_resp - len(prefix_ids))
                        cont_ids = list(roll.response_ids)[:rem_budget]
                        resp_ids = prefix_ids + cont_ids
                        if roll.response_logp is not None:
                            resp_logp = [0.0] * len(prefix_ids) + list(roll.response_logp)[:rem_budget]
                        else:
                            resp_logp = None
                        resp_text = (
                            tok.decode(resp_ids, skip_special_tokens=True)
                            if tok is not None and hasattr(tok, "decode")
                            else target_text
                        )
                        resp_roll = Rollout(
                            prompt_ids=list(roll.prompt_ids),
                            response_ids=resp_ids,
                            response_logp=resp_logp,
                            finish_reason=roll.finish_reason,
                            text=resp_text,
                        )
                        sample_prefix_mask = [False] * plen + [True] * len(prefix_ids) + [False] * len(cont_ids)
                else:
                    resp_roll = roll
                    resp_text = roll.text
                    sample_prefix_mask = [False] * (len(roll.prompt_ids) + len(roll.response_ids))

                sample = rollouts_to_samples([resp_roll], copy_meta=[meta])[0]
                sample.update(
                    {
                        "group.uid": uid,
                        "group.index": j,
                        "meta.response_text": resp_text,
                        "rollout.truncated": float(resp_roll.finish_reason == "length"),
                        "rollout.is_off_policy": float(is_off_policy),
                        "rollout.prefix_mask": sample_prefix_mask,
                    }
                )
                row = {k.removeprefix("meta."): v for k, v in meta.items()}
                row["truncated"] = bool(sample["rollout.truncated"])
                samples.append(sample)
                metas.append(row)
                texts.append(resp_text)

        results = self.verifier([m.get("prompt_text", "") for m in metas], texts, metas)
        if len(results) != len(samples):
            raise ValueError("verifier must return one result per rollout")

        for sample, result in zip(samples, results):
            sample.update(
                {
                    "rollout.reward": float(result["score"]),
                    "rollout.acc": float(result["acc"]),
                    "meta.feedback": str(result["feedback"]),
                }
            )

        batch = self.rt.collator(samples)

        for key in ("reward", "acc", "truncated"):
            metric = "truncated_frac" if key == "truncated" else f"{key}_mean"
            state.log(f"rl/{metric}", batch[f"rollout.{key}"].mean())

        state.log(
            "rl/response_len_mean",
            (batch["student.attention_mask"].sum(-1) - batch["student.prompt_len"]).float().mean(),
        )

        if self.cfg.drop_all_same:
            rewards = batch["rollout.reward"].reshape(-1, self.cfg.n)
            active = rewards.max(-1).values.ne(rewards.min(-1).values).repeat_interleave(self.cfg.n)
            batch["seq.loss_mask"] = active.float()[:, None].expand_as(batch["student.labels"]).clone()

        return batch


__all__ = ["LUFFYGroupProducer"]
