import torch
from transformers import BertConfig, BertModel

from lmkd.core.batch import Batch
from lmkd.data.context import DataContext
from lmkd.data.transforms.tokenize import TokenizeText
from lmkd.models.adapters import HFAutoAdapter
from lmkd.models.forward import SideOutput
from lmkd.train.step_strategies import count_supervised


def test_bert_adapter_backbone():
    cfg = BertConfig(
        vocab_size=67,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    model = BertModel(cfg)
    adapter = HFAutoAdapter()

    # 1. lm_head is None for pure encoder
    assert adapter.lm_head(model) is None

    # 2. backbone is the model itself
    assert adapter.backbone(model) is model

    # 3. layers find encoder.layer and has 3 layers
    layers = adapter.layers(model)
    assert len(layers) == 3
    assert adapter.num_layers(model) == 3
    assert adapter.hidden_size(model) == 32
    assert adapter.vocab_size(model) == 67

    # 4. forward_hidden and SideOutput
    input_ids = torch.randint(0, 67, (2, 8))
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]])
    hf_out = adapter.forward_hidden(model, input_ids, attention_mask, output_hidden_states=True)
    side_out = SideOutput(model, adapter, hf_out, attention_mask)

    assert side_out.logits is None
    assert torch.equal(side_out.hidden(3), hf_out.last_hidden_state)
    assert side_out.hidden(0).shape == (2, 8, 32)
    assert side_out.hidden(-1).shape == (2, 8, 32)
    assert torch.equal(side_out.hidden(-1), hf_out.last_hidden_state)

    # 5. count_supervised without student.labels -> z_tok = 0
    batch = Batch(
        {"student.input_ids": input_ids, "student.attention_mask": attention_mask},
        meta={"meta.id": ["0", "1"]},
    )
    z_tok, z_seq = count_supervised(batch)
    assert z_tok == 0.0
    assert z_seq == 2.0


def test_tokenize_text_keep_text_truncation(tmp_path):
    from tests.conftest import _make_tokenizer

    tok = _make_tokenizer(tmp_path / "tok")
    ctx = DataContext(student_tokenizer=tok, max_length=4)

    t = TokenizeText(TokenizeText.Config(keep_text=True, text_field="text"))
    sample = {"text": "w0 w1 w2 w3 w4 w5", "meta.id": "sample_0"}
    res = t(sample, ctx)

    assert res is not None
    assert "student.input_ids" in res
    assert "student.attention_mask" in res
    assert "student.labels" not in res
    assert res["meta.text"] == "w0 w1 w2 w3 w4 w5"
    assert res["meta.id"] == "sample_0"
    assert len(res["student.input_ids"]) <= 4
    assert len(res["student.attention_mask"]) == len(res["student.input_ids"])
