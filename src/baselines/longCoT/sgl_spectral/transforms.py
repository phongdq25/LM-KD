"""SGL-Spectral data transforms and weight allocation algorithms."""

from __future__ import annotations

import bisect
import math
import re
import sys
from typing import Any

import torch

from lmkd.core.batch import FIELDS, FieldSpec
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import CorpusTransform, Transform

# -----------------------------------------------------------------------------
# SGL Step Transitions & Provenance Fields
# -----------------------------------------------------------------------------
FIELDS.register(FieldSpec("seq.step_id", "token", torch.long, -1, "student", doc="step id per token for L_trans"))
FIELDS.register(FieldSpec("seq.step_end", "ragged", torch.long, doc="[K'] token indices closing each step"))
FIELDS.register(FieldSpec("seq.pair_src", "ragged", torch.long, doc="[P] source step indices of kept transitions"))


def select_steps_by_energy(strengths: list[float], energy_threshold: float) -> list[int]:
    """Minimal step set whose cumulative spectral strength >= p * total (SGL selection/energy.py:4-29)."""
    if not strengths:
        return []
    total = sum(strengths)
    order = sorted(range(len(strengths)), key=lambda i: (-strengths[i], i))
    if total <= 0:
        return list(range(len(strengths)))

    selected, accumulated = [], 0.0
    for index in order:
        selected.append(index)
        accumulated += strengths[index]
        if accumulated >= energy_threshold * total:
            break
    return sorted(selected)


def _stable_entropy_weights(
    entropies: list[float],
    lengths: list[int],
    temperature: float = 1.0,
    interpolation: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
) -> list[float]:
    """Entropy weights with variance scaling and mass preservation."""
    if not 0.0 <= interpolation <= 1.0:
        raise ValueError("interpolation must be in [0, 1]")
    if clip < 0:
        raise ValueError("clip must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not entropies:
        return []

    mean = sum(entropies) / len(entropies)
    variance = sum((entropy - mean) ** 2 for entropy in entropies) / len(entropies)
    std = math.sqrt(variance)
    z_scores = [0.0 if std < epsilon else (entropy - mean) / (std + epsilon) for entropy in entropies]
    scaled = [max(-clip, min(clip, z_score)) / temperature for z_score in z_scores]
    maximum = max(scaled)
    raw = [max(math.exp(value - maximum), sys.float_info.min) for value in scaled]
    token_mass = sum(lengths)
    weighted_mass = sum(length * weight for length, weight in zip(lengths, raw))
    normalized = [weight * token_mass / weighted_mass for weight in raw]
    return [(1.0 - interpolation) + interpolation * weight for weight in normalized]


PREFIX, CONTINUATION, JUNCTION = "P", "C", "J"


def step_regions(step_spans: list[tuple[int, int]], boundary: int) -> list[str]:
    """Region of each step given boundary (SGL allocation/region.py:30-40)."""
    regions = []
    for start, end in step_spans:
        if end <= boundary:
            regions.append(PREFIX)
        elif start >= boundary:
            regions.append(CONTINUATION)
        else:
            regions.append(JUNCTION)
    return regions


def region_stable_step_weights(
    entropies: list[float],
    lengths: list[int],
    regions: list[str],
    temperature: float = 1.0,
    interpolation: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
) -> list[float]:
    """Stable step weights within each region (SGL allocation/region.py:43-67)."""
    if not len(entropies) == len(lengths) == len(regions):
        raise ValueError("entropies, lengths and regions must have the same number of steps")
    weights = [1.0] * len(entropies)
    for region in (PREFIX, CONTINUATION):
        indices = [index for index, value in enumerate(regions) if value == region]
        if not indices:
            continue
        region_weights = _stable_entropy_weights(
            [entropies[index] for index in indices],
            [lengths[index] for index in indices],
            temperature,
            interpolation,
            clip,
            epsilon,
        )
        for index, weight in zip(indices, region_weights):
            weights[index] = weight
    return weights


def region_gain_ratio(
    gains: list[list[float]],
    lengths: list[list[int]],
    regions: list[list[str]],
    gamma: float = 1.0,
    tail_fraction: float = 0.0,
) -> tuple[float, float, float]:
    """Corpus answer gain ratio (SGL allocation/region.py:69-98)."""
    if not 0.0 <= tail_fraction < 1.0:
        raise ValueError("tail_fraction must be in [0, 1)")
    totals = {PREFIX: [0.0, 0], CONTINUATION: [0.0, 0]}
    for record_gains, record_lengths, record_regions in zip(gains, lengths, regions):
        keep = len(record_gains) - math.ceil(tail_fraction * len(record_gains))
        for gain, length, region in zip(record_gains[:keep], record_lengths[:keep], record_regions[:keep]):
            if region in totals:
                totals[region][0] += gain
                totals[region][1] += length
    if not totals[PREFIX][1] or not totals[CONTINUATION][1]:
        raise ValueError("region ratio needs both prefix and continuation tokens")
    gain_prefix = totals[PREFIX][0] / totals[PREFIX][1]
    gain_cont = totals[CONTINUATION][0] / totals[CONTINUATION][1]
    if gain_prefix <= 0 or gain_cont <= 0:
        raise ValueError(
            f"answer gain per token must be positive in both regions (prefix {gain_prefix:.6g}, continuation {gain_cont:.6g})"
        )
    return gain_prefix, gain_cont, (gain_prefix / gain_cont) ** gamma


def region_budget(lengths: list[int], regions: list[str], ratio: float) -> list[float]:
    """Per-step budget coefficient b_r (SGL allocation/region.py:100-111)."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    prefix_mass = sum(length for length, region in zip(lengths, regions) if region == PREFIX)
    cont_mass = sum(length for length, region in zip(lengths, regions) if region == CONTINUATION)
    if not prefix_mass or not cont_mass:
        return [1.0] * len(lengths)
    cont_coefficient = (prefix_mass + cont_mass) / (ratio * prefix_mass + cont_mass)
    by_region = {PREFIX: ratio * cont_coefficient, CONTINUATION: cont_coefficient, JUNCTION: 1.0}
    return [by_region[region] for region in regions]


def boost_tail(
    loss_mask: list[int],
    loss_weights: list[float] | None,
    tail_tokens: int,
    boost: float,
) -> list[float]:
    """Tail boost with sample mass preserved (SGL transforms/tail_boost.py:19-32)."""
    if tail_tokens < 0 or boost <= 0:
        raise ValueError("tail_tokens must be >= 0 and boost > 0")
    weights = [float(w) for w in loss_weights] if loss_weights is not None else [float(m) for m in loss_mask]
    supervised = [index for index, keep in enumerate(loss_mask) if keep]
    mass = sum(weights[index] for index in supervised)
    for index in supervised[-tail_tokens:] if tail_tokens else []:
        weights[index] *= boost
    boosted = sum(weights[index] for index in supervised)
    if boosted > 0:
        scale = mass / boosted
        weights = [weight * scale if loss_mask[index] else 0.0 for index, weight in enumerate(weights)]
    return weights


@TRANSFORMS.register("select_energy_top_p", source="SpectralGuidedLearning/src/sgl/selection/energy.py:4-29")
class SelectEnergyTopP(Transform):
    """Energy-based step selection with stop token preservation.

    Source: SGL selection/energy.py:4-47, selection/build_masks.py:19-51.
    """

    requires = ("seq.steps", "signals.step_strength")
    provides = ("seq.loss_mask", "meta.selected_steps")

    class Config(BaseConfig):
        p: float = 0.95
        require_signal: bool = True

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        step_spans = sample["seq.steps"]
        strengths = sample.get("signals.step_strength")
        if strengths is None:
            if self.cfg.require_signal:
                raise KeyError("sample missing 'signals.step_strength'")
            selected = list(range(len(step_spans)))
        else:
            if len(strengths) != len(step_spans):
                raise ValueError(f"record {sample.get('meta.id')}: steps={len(step_spans)}, strengths={len(strengths)}")
            selected = select_steps_by_energy(strengths, self.cfg.p)

        seq_len = len(sample["student.input_ids"])
        mask = [0] * seq_len
        for idx in selected:
            start, end = step_spans[idx]
            for pos in range(start, min(end, seq_len)):
                mask[pos] = 1

        resp_end = sample["seq.response_span"][1]
        for pos in range(resp_end, seq_len):
            mask[pos] = 1

        sample["seq.loss_mask"] = mask
        sample["meta.selected_steps"] = selected
        return sample


def _paint_step_weights(
    sample: dict[str, Any],
    selected: list[int],
    step_weights: list[float],
    step_spans: list[tuple[int, int]],
) -> list[float]:
    """Per-token weights: each selected step's tokens get its weight, tokens after the response
    (stop tokens) get 1.0, everything else 0.0."""
    n = len(sample["seq.loss_mask"])
    weights = [0.0] * n
    for step_idx, weight in zip(selected, step_weights):
        start, end = step_spans[step_idx]
        weights[start:end] = [weight] * (end - start)
    for pos in range(sample["seq.response_span"][1], n):
        weights[pos] = 1.0
    return weights


@TRANSFORMS.register("allocate_region", source="SpectralGuidedLearning/src/sgl/allocation/region.py:30-111")
class AllocateRegion(CorpusTransform):
    """Student-aligned region weighting (SARW) across prefix and continuation.

    Source: SGL allocation/region.py:30-111, allocation/build.py:96-118.
    """

    requires = ("seq.loss_mask", "meta.selected_steps", "meta.prefix_boundary")
    provides = ("seq.token_weights",)

    class Config(BaseConfig):
        variant: str = "sarw"
        gamma: float = 1.0
        tail_fraction: float = 0.1
        tau: float = 2.0
        lam: float = 0.5
        clip: float = 2.0
        eps: float = 1e-8
        signal: str = "step_entropy"
        gain_signal: str = "step_gain"

    ratio: float | None = None  # corpus prefix/continuation gain ratio, set by fit()

    def _needs_ratio(self) -> bool:
        return self.cfg.variant in ("region-gain", "sarw")

    def fit(self, samples: list[dict[str, Any]], ctx: DataContext) -> None:
        """Corpus statistic (SGL allocation/build.py:96-118): needs the WHOLE corpus, never a chunk."""
        if not self._needs_ratio():
            self.ratio = 1.0
            return
        gain_key = (
            self.cfg.gain_signal if self.cfg.gain_signal.startswith("signals.") else f"signals.{self.cfg.gain_signal}"
        )
        all_gains, all_lengths, all_regions = [], [], []
        for sample in samples:
            step_spans = sample["seq.steps"]
            gains = sample.get(gain_key)
            if gains is None:
                raise ValueError(f"record {sample.get('meta.id')}: variant '{self.cfg.variant}' needs {gain_key!r}")
            if len(gains) != len(step_spans):
                raise ValueError(f"record {sample.get('meta.id')}: steps={len(step_spans)}, gains={len(gains)}")
            all_gains.append(gains)
            all_lengths.append([end - start for start, end in step_spans])
            all_regions.append(step_regions(step_spans, sample["meta.prefix_boundary"]))
        if not all_gains:
            raise ValueError("allocate_region.fit: no samples")
        _, _, self.ratio = region_gain_ratio(
            all_gains, all_lengths, all_regions, self.cfg.gamma, self.cfg.tail_fraction
        )

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        step_spans = sample["seq.steps"]
        boundary = sample["meta.prefix_boundary"]
        regions = step_regions(step_spans, boundary)
        selected = sample["meta.selected_steps"]

        if self._needs_ratio() and self.ratio is None:
            raise RuntimeError(
                "allocate_region: fit() was not called (needs the whole corpus; run it through "
                "run_transforms / prepare_dataset, not per sample or per chunk)"
            )
        if self.cfg.variant == "region-cont":
            selected = [i for i in selected if regions[i] != PREFIX]
            sample["meta.selected_steps"] = selected
            step_weights = [1.0] * len(selected)
            new_mask = [0] * len(sample["seq.loss_mask"])
            for i in selected:
                start, end = step_spans[i]
                for pos in range(start, min(end, len(new_mask))):
                    new_mask[pos] = 1
            resp_end = sample["seq.response_span"][1]
            for pos in range(resp_end, len(new_mask)):
                new_mask[pos] = 1
            sample["seq.loss_mask"] = new_mask
        else:
            sig_key = f"signals.{self.cfg.signal}" if not self.cfg.signal.startswith("signals.") else self.cfg.signal
            entropies = sample.get(sig_key, sample.get("signals.step_entropy"))
            if entropies is None:
                raise KeyError(f"sample missing {sig_key!r}")
            if len(entropies) != len(step_spans):
                raise ValueError(f"record {sample.get('meta.id')}: steps={len(step_spans)}, signal={len(entropies)}")
            sel_entropies = [entropies[i] for i in selected]
            sel_lengths = [step_spans[i][1] - step_spans[i][0] for i in selected]
            sel_regions = [regions[i] for i in selected]

            within = (
                region_stable_step_weights(
                    sel_entropies,
                    sel_lengths,
                    sel_regions,
                    self.cfg.tau,
                    self.cfg.lam,
                    self.cfg.clip,
                    self.cfg.eps,
                )
                if self.cfg.variant not in ("region-gain", "region-budget")
                else [1.0] * len(selected)
            )
            budget = (
                region_budget(sel_lengths, sel_regions, (self.ratio if self.ratio is not None else 1.0))
                if self.cfg.variant != "region-stable"
                else [1.0] * len(selected)
            )
            step_weights = [a * b for a, b in zip(within, budget)]

        sample["seq.token_weights"] = _paint_step_weights(sample, selected, step_weights, step_spans)
        return sample


# -----------------------------------------------------------------------------
# SGL Step Transitions & Provenance Helpers & Transforms
# -----------------------------------------------------------------------------
_PARAGRAPH_BREAK = re.compile(r"\n{2,}")


def cot_char_span(response: str) -> tuple[int, int]:
    """Character span of CoT inside response (SGL transforms/step_transitions.py:19-33)."""
    close = response.find("</think>")
    if close < 0:
        return 0, len(response)
    open_tag = response.find("<think>")
    start = open_tag + len("<think>") if 0 <= open_tag < close else 0
    return start, close


def split_cot_char_cuts(response: str) -> list[int]:
    """Paragraph breaks inside CoT (SGL transforms/step_transitions.py:35-47)."""
    start, end = cot_char_span(response)
    cot = response[start:end]
    if not cot.strip():
        return []
    cuts = [start]
    for match in _PARAGRAPH_BREAK.finditer(cot):
        if match.end() < len(cot):
            cuts.append(start + match.end())
    cuts.append(end)
    return cuts


def merge_short_steps(spans: list[tuple[int, int]], min_step_tokens: int) -> list[tuple[int, int]]:
    """Merge steps shorter than min_step_tokens (SGL transforms/step_transitions.py:50-60)."""
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and (end - start < min_step_tokens or merged[-1][1] - merged[-1][0] < min_step_tokens):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def step_token_spans_from_cuts(cuts: list[int], token_starts: list[int], offset: int = 0) -> list[tuple[int, int]]:
    """Map character cuts to absolute token spans (SGL transforms/step_transitions.py:63-72)."""
    if len(cuts) < 2:
        return []
    bounds = [bisect.bisect_left(token_starts, cut) for cut in cuts]
    return [(offset + a, offset + b) for a, b in zip(bounds, bounds[1:]) if b > a]


def build_transition_fields(
    response: str,
    token_starts: list[int],
    response_offset: int,
    sequence_length: int,
    min_step_tokens: int = 8,
) -> dict:
    """Next-step representation objective fields (SGL transforms/step_transitions.py:74-105)."""
    spans = step_token_spans_from_cuts(split_cot_char_cuts(response), token_starts, response_offset)
    spans = merge_short_steps(spans, min_step_tokens)

    step_id = [-1] * sequence_length
    for index, (start, end) in enumerate(spans):
        for position in range(start, min(end, sequence_length)):
            step_id[position] = index
    step_end = [end - 1 for _, end in spans]
    pair_src = [i for i in range(len(spans) - 1) if spans[i + 1][1] - spans[i + 1][0] >= min_step_tokens]
    return {"step_id": step_id, "step_end": step_end, "num_steps": len(spans), "pair_src": pair_src}


@TRANSFORMS.register("tail_boost", source="SpectralGuidedLearning/src/sgl/transforms/tail_boost.py:19-32")
class TailBoost(Transform):
    """Boost loss on concluding supervised tokens while preserving sample mass.

    Source: SGL transforms/tail_boost.py:19-32.
    """

    requires = ("seq.loss_mask",)
    provides = ("seq.token_weights",)

    class Config(BaseConfig):
        tail_tokens: int = 128
        boost: float = 3.0

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        loss_mask = sample["seq.loss_mask"]
        existing_weights = sample.get("seq.token_weights")
        labels = sample.get("student.labels")

        supervised_mask = [1 if (m and (labels is None or labels[i] != -100)) else 0 for i, m in enumerate(loss_mask)]
        new_weights = boost_tail(supervised_mask, existing_weights, self.cfg.tail_tokens, self.cfg.boost)
        sample["seq.token_weights"] = new_weights
        return sample


@TRANSFORMS.register("provenance", source="SpectralGuidedLearning/src/sgl/transforms/provenance.py:29-82")
class Provenance(Transform):
    """Tag per-token origin (prompt=0, prefix=1, continuation=2, junction=3).

    Source: SGL transforms/provenance.py:29-82.
    """

    requires = ("seq.response_span", "meta.response_text", "meta.token_starts")
    provides = ("seq.token_source", "meta.prefix_boundary")

    class Config(BaseConfig):
        junction_tokens: int = 0
        marker: str = "<End_of_Prefix>"

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        response = sample["meta.response_text"]
        starts = sample["meta.token_starts"]
        start, _ = sample["seq.response_span"]

        cut = response.find(self.cfg.marker)
        if cut < 0:
            n_prefix = 0
        else:
            boundary_char = cut + len(self.cfg.marker)
            n_prefix = sum(1 for s in starts if s < boundary_char)

        token_source = [0] * len(sample["student.input_ids"])
        loss_mask = sample.get("seq.loss_mask", [1] * len(token_source))
        labels = sample.get("student.labels")

        for pos in range(len(token_source)):
            if loss_mask[pos] and (labels is None or labels[pos] != -100):
                if pos < start + n_prefix:
                    token_source[pos] = 1
                elif n_prefix > 0 and pos < start + n_prefix + self.cfg.junction_tokens:
                    token_source[pos] = 3
                else:
                    token_source[pos] = 2

        sample["seq.token_source"] = token_source
        sample["meta.prefix_boundary"] = start + n_prefix
        return sample


@TRANSFORMS.register("step_transitions", source="SpectralGuidedLearning/src/sgl/transforms/step_transitions.py:19-105")
class StepTransitions(Transform):
    """Construct step representations and transition pairs for L_trans.

    Source: SGL transforms/step_transitions.py:19-105.
    """

    requires = ("seq.response_span", "meta.response_text", "meta.token_starts")
    provides = ("seq.step_id", "seq.step_end", "seq.pair_src")

    class Config(BaseConfig):
        min_step_tokens: int = 8

    def __call__(self, sample: dict[str, Any], ctx: DataContext) -> dict[str, Any] | None:
        response = sample["meta.response_text"]
        starts = sample["meta.token_starts"]
        resp_offset = sample["seq.response_span"][0]
        seq_len = len(sample["student.input_ids"])

        fields = build_transition_fields(response, starts, resp_offset, seq_len, self.cfg.min_step_tokens)
        sample["seq.step_id"] = fields["step_id"]
        sample["seq.step_end"] = fields["step_end"]
        sample["seq.pair_src"] = fields["pair_src"]
        return sample


__all__ = [
    "select_steps_by_energy",
    "step_regions",
    "region_stable_step_weights",
    "region_gain_ratio",
    "region_budget",
    "boost_tail",
    "cot_char_span",
    "split_cot_char_cuts",
    "merge_short_steps",
    "step_token_spans_from_cuts",
    "build_transition_fields",
    "SelectEnergyTopP",
    "AllocateRegion",
    "TailBoost",
    "Provenance",
    "StepTransitions",
]
