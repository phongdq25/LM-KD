"""Layer selection tests against reference implementation."""

from types import SimpleNamespace

import pytest
import torch
from tests.parity._ref import impact_ref as ref

from baselines.cross_tokenizer.impact import MapperInputs, compute_bi_scores, compute_ppl_scores
from lmkd.core.registry import LAYER_MAPPERS
from lmkd.core.state import TrainState
from lmkd.losses.cross_tokenizer._common import view_from_target_hidden


def _hidden() -> list[torch.Tensor]:
    torch.manual_seed(15)
    return [torch.randn(2, 5, 4) for _ in range(13)]


def _inputs(state: TrainState | None = None) -> MapperInputs:
    hidden = _hidden()
    return MapperInputs(
        6,
        12,
        state or TrainState(),
        teacher_hidden=lambda l: hidden[l],
        rt=SimpleNamespace(config=SimpleNamespace(train=SimpleNamespace(seed=17))),
    )


def test_bi_scores_match_ref():
    inp = _inputs()
    torch.testing.assert_close(compute_bi_scores(inp), ref.compute_bi_scores(_hidden()), atol=1e-6, rtol=0)


def test_ppl_scores_match_ref():
    torch.manual_seed(18)
    hidden = [torch.randn(2, 5, 4) for _ in range(4)]
    head = torch.nn.Linear(4, 9, bias=False)
    labels = torch.tensor([[-100, 1, 2, 3, -100], [-100, 2, 3, 4, 5]])
    inp = MapperInputs(
        2,
        3,
        TrainState(),
        teacher_labels=labels,
        teacher_view_at_layer=lambda l: view_from_target_hidden(hidden[l], head),
    )
    model = SimpleNamespace(get_output_embeddings=lambda: head)
    torch.testing.assert_close(
        compute_ppl_scores(inp), ref.compute_ppl_layer_scores(hidden, model, labels), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize(
    "name,choose_align,choose_layer",
    [
        ("topk_bi", "BI", "BI"),
        ("first", "BI", "FIRST"),
        ("last", "BI", "LAST"),
        ("random", "BI", "RANDOM"),
        ("ratio_1to1", "1-1", "BI"),
    ],
)
def test_pairs_match_ref(name, choose_align, choose_layer):
    inp = _inputs()
    scores = torch.linspace(0.1, 1.2, 12)
    cfg = {"type": name, "k": 4, "tau": 0.7}
    if name == "random":
        cfg["seed"] = 17
    mapper = LAYER_MAPPERS.build(cfg)
    if name == "topk_bi":
        mapper.prepare(SimpleNamespace(extra={"impact_layer_scores": scores}))
    got = mapper.pairs(inp)
    expected, _, _ = ref.build_impact_alignment_pairs(
        choose_align,
        choose_layer,
        _hidden(),
        4,
        12,
        6,
        bi_tau=0.7,
        random_seed=17,
        precomputed_layer_scores=scores if name == "topk_bi" else None,
    )
    assert [(l, m) for l, m, _ in got] == [(l, m) for l, m, _ in expected]
    torch.testing.assert_close(
        torch.tensor([w for _, _, w in got]), torch.tensor([w for _, _, w in expected]), atol=1e-6, rtol=0
    )


def test_ppl_selects_largest_nll():
    values = torch.tensor([0.2, 1.2, 0.1, 0.5, 3.0, 0.6, 1.0, 0.9, 0.8, 0.7, 0.4, 0.3])
    inp = _inputs()
    inp.teacher_labels = torch.tensor([[-100, 1, 2], [-100, 2, 1]])
    inp.teacher_view_at_layer = lambda l: SimpleNamespace(target_logp=lambda labels: -values[l - 1].expand_as(labels))
    mapper = LAYER_MAPPERS.build({"type": "topk_ppl", "k": 3, "calibration": "online"})
    mapper.observe(inp)
    assert [l for l, _, _ in mapper.pairs(inp)] == [2, 5, 7]


def test_railkd_schedule():
    mapper = LAYER_MAPPERS.build({"type": "railkd", "seed": 7, "resample_epochs": 1})
    a = MapperInputs(4, 12, TrainState(epoch=0.0, step=0))
    first = mapper.pairs(a)
    assert mapper.pairs(MapperInputs(4, 12, TrainState(epoch=0.8, step=2))) == first
    b = MapperInputs(4, 12, TrainState(epoch=1.0, step=3))
    second = mapper.pairs(b)
    sampled = ref.sample_rail_kd_teacher_layers(12, 4, seed=7 + 10007 + 3)
    assert [l for l, _, _ in second] == sampled[:3]
    assert second != first
    restored = LAYER_MAPPERS.build({"type": "railkd", "seed": 7})
    restored.load_state_dict(mapper.state_dict())
    assert restored.pairs(b) == second


def test_accumulate_freeze(monkeypatch):
    import baselines.cross_tokenizer.impact.layer_mappers as lm

    score = torch.tensor([0.1, 0.2, 0.3])
    monkeypatch.setattr(lm, "compute_bi_scores", lambda inp: score.clone())
    mapper = LAYER_MAPPERS.build({"type": "topk_bi", "k": 2, "calibration": "accumulate", "calib_samples": 4})
    inp = MapperInputs(2, 3, TrainState(), teacher_hidden=lambda l: torch.zeros(2, 3, 4))
    mapper.observe(inp)
    score[:] = torch.tensor([0.9, 0.8, 0.7])
    mapper.observe(inp)
    frozen = mapper.state_dict()
    score[:] = torch.tensor([9.0, 8.0, 7.0])
    mapper.observe(inp)
    assert mapper.state_dict()["count"] == 4
    restored = LAYER_MAPPERS.build({"type": "topk_bi", "k": 2, "calibration": "accumulate"})
    restored.load_state_dict(frozen)
    assert restored.pairs(inp) == mapper.pairs(inp)


def test_alpkd_weights_sum_one():
    torch.manual_seed(16)
    student = [torch.randn(2, 5, 3) for _ in range(4)]
    teacher = [torch.randn(2, 6, 4) for _ in range(6)]
    W = torch.randn(4, 3)
    inp = MapperInputs(
        3, 5, TrainState(), teacher_hidden=lambda l: teacher[l], student_hidden=lambda l: student[l], W=W
    )
    mapper = LAYER_MAPPERS.build({"type": "alpkd"})
    got = mapper.fuse_weights(inp)
    expected = ref.compute_alp_kd_attention_weights(student, teacher, W, 5, 3)
    assert got is not None and got.shape == (2, 3, 5)
    torch.testing.assert_close(got.sum(-1), torch.ones(2, 3))
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=0)


def test_bi_ratio_extension():
    mapper = LAYER_MAPPERS.build({"type": "bi_ratio", "layers": [2, 7, 12]})
    assert mapper.pairs(_inputs()) == [(2, 1, 1 / 3), (7, 4, 1 / 3), (12, 6, 1 / 3)]
