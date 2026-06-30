"""Classifier checkpoint export/import."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_classifier_export_import_roundtrip(tmp_path: Path):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            [
                {"text": "yay", "tag": "Happy"},
                {"text": "boo", "tag": "Sad"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=32)
    ckpt = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(ckpt))
    loaded = ai.import_classifier("safetensors", [str(ckpt)])
    assert loaded.labels == clf.labels
    before = clf.predict("yay")
    after = loaded.predict("yay")
    for label in clf.labels:
        assert abs(before[label] - after[label]) < 1e-4


def test_import_classifier_rejects_chatbot_checkpoint(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32)
    path = tmp_path / "chatbot.mmn"
    ai.export(bot, "safetensors", str(path))
    with pytest.raises(RuntimeError, match="mmn-classifier-v1"):
        ai.import_classifier("safetensors", [str(path)])
