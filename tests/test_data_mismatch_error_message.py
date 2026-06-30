from pathlib import Path

import pytest

import magicmindnet as ai


def test_train_rejects_classification_with_data_mismatch_error(tmp_path: Path):
    cls_path = tmp_path / "cls.json"
    cls_path.write_text('[{"text": "hi", "tag": "A"}]', encoding="utf-8")
    ds = ai.DatasetClassification(str(cls_path), "text", "tag")
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16)
    cfg = ai.TrainConfig(epochs=1, batch_size=2, cuda=False)
    with pytest.raises(ai.DataMismatchError) as exc:
        ai.Train(bot, ds, cfg)
    assert "DatasetQA or DatasetCorpus" in str(exc.value)
