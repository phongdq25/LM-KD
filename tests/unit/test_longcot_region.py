"""T18 review fixes: allocate_region (fit/ratio, region-cont mask), selection length checks, corpus fit in prepare."""

from __future__ import annotations

import pytest

from baselines.longCoT.sgl_spectral.transforms import (
    AllocateRegion,
    SelectEnergyTopP,
    region_gain_ratio,
    step_regions,
)
from baselines.longCoT.ssft.transforms import SelectCumulativeStrength
from lmkd.core.plugin import BaseConfig
from lmkd.core.registry import TRANSFORMS
from lmkd.data.context import DataContext
from lmkd.data.transforms.base import CorpusTransform
from lmkd.plugins import load_builtin_plugins  # noqa: E402

load_builtin_plugins()

CTX = DataContext(student_tokenizer=None, max_length=64)


def _sample(gains=(1.0, 2.0, 2.0), boundary=4, selected=(0, 1, 2)):
    """prompt = tokens 0-1; steps (2,4) (4,6) (6,8); stop tokens 8,9. boundary=4 -> step 0 is prefix."""
    n = 10
    mask = [0] * n
    for i in selected:
        a, b = [(2, 4), (4, 6), (6, 8)][i]
        mask[a:b] = [1] * (b - a)
    mask[8:] = [1, 1]
    return {
        "student.input_ids": list(range(n)),
        "seq.response_span": [2, 8],
        "seq.steps": [(2, 4), (4, 6), (6, 8)],
        "seq.loss_mask": mask,
        "meta.selected_steps": list(selected),
        "meta.prefix_boundary": boundary,
        "signals.step_entropy": [0.5, 1.5, 0.2],
        "signals.step_gain": list(gains),
    }


def test_region_cont_drops_prefix_from_mask_and_weights():
    t = AllocateRegion(AllocateRegion.Config(variant="region-cont"))
    t.fit([_sample()], CTX)
    out = t(_sample(), CTX)
    assert out["meta.selected_steps"] == [1, 2]
    assert out["seq.loss_mask"] == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # prefix step 0 not supervised; stop is
    assert out["seq.token_weights"] == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert all(w == 0 for w, m in zip(out["seq.token_weights"], out["seq.loss_mask"]) if not m)


def test_region_sarw_fit_uses_whole_corpus_ratio():
    samples = [_sample(gains=(1.0, 2.0, 2.0)), _sample(gains=(3.0, 1.0, 1.0))]
    t = AllocateRegion(AllocateRegion.Config(variant="sarw", tail_fraction=0.0))
    t.fit(samples, CTX)
    lens = [[2, 2, 2]] * 2
    regs = [step_regions([(2, 4), (4, 6), (6, 8)], 4)] * 2
    _, _, expected = region_gain_ratio([s["signals.step_gain"] for s in samples], lens, regs, 1.0, 0.0)
    assert t.ratio == pytest.approx(expected)
    out = t(samples[0], CTX)
    assert out["seq.token_weights"][8:] == [1.0, 1.0]


def test_region_requires_fit_and_gain_signal():
    t = AllocateRegion(AllocateRegion.Config(variant="sarw"))
    with pytest.raises(RuntimeError, match="fit"):
        t(_sample(), CTX)
    s = _sample()
    del s["signals.step_gain"]
    with pytest.raises(ValueError, match="step_gain"):
        t.fit([s], CTX)
    s["signals.step_gain"] = [1.0]
    with pytest.raises(ValueError, match="steps"):
        t.fit([s], CTX)


def test_energy_select_checks_signal_length():
    s = _sample()
    s["signals.step_strength"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="steps=3"):
        SelectEnergyTopP(SelectEnergyTopP.Config())(s, CTX)


def test_ssft_select_checks_signal_length():
    s = _sample()
    s["signals.segment_ig"] = [[4, 0.1, 0.1]]
    with pytest.raises(ValueError, match="steps=3"):
        SelectCumulativeStrength(SelectCumulativeStrength.Config())(s, CTX)


def test_prepare_dataset_fits_corpus_transform_on_whole_corpus(tmp_path):
    """num_proc>1 must not fit a CorpusTransform per chunk."""
    import json
    from types import SimpleNamespace

    from lmkd.pipeline.stage import StageContext
    from lmkd.pipeline.stages.prepare import PrepareDataset

    @TRANSFORMS.register("_t18_count_corpus", force=True)
    class _Count(CorpusTransform):
        class Config(BaseConfig):
            pass

        def fit(self, samples, ctx):
            self.n = len(samples)

        def __call__(self, sample, ctx):
            sample["meta.n_fit"] = self.n
            return sample

    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps({"id": i, "prompt": "p", "response": "r"}) for i in range(8)))
    stage = PrepareDataset(
        PrepareDataset.Config(
            source={"type": "jsonl", "path": str(path)}, transforms=[{"type": "_t18_count_corpus"}], num_proc=2
        )
    )
    out = tmp_path / "out"
    out.mkdir()
    stage.setup(SimpleNamespace())
    stage.run(StageContext(rt=SimpleNamespace(), output=SimpleNamespace(path=out), inputs={}))
    from lmkd.pipeline.io import load_samples_dir

    assert {s["meta.n_fit"] for s in load_samples_dir(out)} == {8}
