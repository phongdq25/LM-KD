import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tests.conftest import _make_gpt2, _make_tokenizer

from lmkd.core.batch import Batch
from lmkd.pipeline.artifacts import Artifact
from lmkd.pipeline.io import save_samples
from lmkd.pipeline.stage import StageContext
from lmkd.pipeline.stages.embedding import EmbedTeacher
from lmkd.teachers.providers import EmbeddingStoreTeacher


def test_lookup_by_id(tmp_path):
    emb_file = tmp_path / "embeddings.safetensors"
    ids_file = tmp_path / "ids.json"

    N, D = 10, 8
    tensors = torch.randn(N, D)
    save_file({"emb": tensors}, str(emb_file))
    ids = [f"doc_{i}" for i in range(N)]
    ids_file.write_text(json.dumps(ids))

    teacher = EmbeddingStoreTeacher(EmbeddingStoreTeacher.Config(path=str(emb_file)))
    teacher.setup(SimpleNamespace(extra={}))

    # Shuffled lookup
    shuffled_ids = ["doc_7", "doc_2", "doc_0"]
    batch = Batch({}, meta={"meta.id": shuffled_ids})
    out = teacher.signals(batch, None, None, None)

    assert out is not None
    assert out.embedding is not None
    assert out.embedding.shape == (3, D)
    assert torch.equal(out.embedding[0], tensors[7])
    assert torch.equal(out.embedding[1], tensors[2])
    assert torch.equal(out.embedding[2], tensors[0])

    # Unknown ID -> KeyError containing the ID
    bad_batch = Batch({}, meta={"meta.id": ["doc_unknown"]})
    with pytest.raises(KeyError, match="doc_unknown"):
        teacher.signals(bad_batch, None, None, None)


def test_by_position_requires_flag(tmp_path):
    pt_file = tmp_path / "embeddings.pt"
    tensors = torch.randn(5, 8)
    torch.save(tensors, pt_file)

    # Missing ids.json and by_position=False -> error
    teacher = EmbeddingStoreTeacher(EmbeddingStoreTeacher.Config(path=str(pt_file), by_position=False))
    with pytest.raises(ValueError, match="ids.json not found"):
        teacher.setup(SimpleNamespace(extra={}))

    # by_position=True -> succeeds
    teacher_pos = EmbeddingStoreTeacher(EmbeddingStoreTeacher.Config(path=str(pt_file), by_position=True))
    teacher_pos.setup(SimpleNamespace(extra={}))

    batch = Batch({}, meta={"meta.id": ["3", "1"]})
    out = teacher_pos.signals(batch, None, None, None)
    assert out is not None
    assert torch.equal(out.embedding[0], tensors[3])
    assert torch.equal(out.embedding[1], tensors[1])


def test_embed_teacher_stage(tmp_path):
    # Create tiny model and tokenizer
    model_dir = tmp_path / "teacher_model"
    tok = _make_tokenizer(model_dir)
    _make_gpt2(model_dir, 32, 2, len(tok), 0)

    # Create input data artifact
    data_dir = tmp_path / "data_art"
    samples = [{"meta.id": f"s_{i}", "text": f"w{i} w{i + 1}"} for i in range(4)]
    save_samples(data_dir, samples)
    data_art = Artifact("data", "data", data_dir)

    out_dir = tmp_path / "output_art"
    out_art = Artifact("embed_teacher", "emb", out_dir)

    stage = EmbedTeacher(
        EmbedTeacher.Config(
            model=str(model_dir),
            pooling="cls",
            max_len=32,
            batch_size=2,
        )
    )
    ctx = StageContext(
        rt=SimpleNamespace(),
        inputs={"data": data_art},
        output=out_art,
    )
    manifest = stage.run(ctx)

    assert manifest["n"] == 4
    assert manifest["dim"] == 32
    assert (out_dir / "embeddings.safetensors").exists()
    assert (out_dir / "ids.json").exists()
    saved_ids = json.loads((out_dir / "ids.json").read_text())
    assert saved_ids == ["s_0", "s_1", "s_2", "s_3"]
