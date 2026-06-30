"""Legacy `bin` export stores architecture meta only (no weights)."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_bin_export_import_restores_architecture(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, seed=5)
    path = tmp_path / "arch.bin"
    ai.export(bot, "bin", str(path))
    loaded = ai.import_model("bin", [str(path)])
    assert loaded.layer_size == bot.layer_size
    assert loaded.parameters == bot.parameters


def test_import_bin_rejects_safetensors_checkpoint(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    with pytest.raises(RuntimeError, match="mmn-bin-v1"):
        ai.import_model("bin", [str(path)])


def test_import_bin_rejects_classifier_checkpoint(tmp_path: Path):
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps([{"text": "a", "tag": "X"}, {"text": "b", "tag": "Y"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(labels), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=16, seed=1)
    path = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(path))
    with pytest.raises(RuntimeError, match="mmn-bin-v1"):
        ai.import_model("bin", [str(path)])


def test_import_bin_rejects_invalid_json_py(tmp_path: Path):
    path = tmp_path / "bad.bin"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        ai.import_model("bin", [str(path)])


def test_import_bin_rejects_empty_file_py(tmp_path: Path):
    path = tmp_path / "empty.bin"
    path.write_text("", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        ai.import_model("bin", [str(path)])
