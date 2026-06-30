import json

import pytest

import magicmindnet as ai


def test_chatbot_compute_mean_loss_rejects_classification_dataset(tmp_path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps([{"text": "hi", "tag": "A"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=1)
    with pytest.raises(ai.DataMismatchError):
        bot.compute_mean_loss(ds)
