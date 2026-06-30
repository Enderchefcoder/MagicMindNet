"""Checkpoint import must reject wrong format wrappers."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_import_model_rejects_classifier_checkpoint(tmp_path: Path):
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps([{"text": "a", "tag": "X"}, {"text": "b", "tag": "Y"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(labels), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=16, seed=1)
    path = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(path))
    with pytest.raises(RuntimeError, match="mmn-safetensors-v1"):
        ai.import_model("safetensors", [str(path)])


def test_import_model_rejects_unknown_format_field(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32)
    path = tmp_path / "bad.mmn"
    ai.export(bot, "safetensors", str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["format"] = "not-a-real-format"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mmn-safetensors-v1"):
        ai.import_model("safetensors", [str(path)])
