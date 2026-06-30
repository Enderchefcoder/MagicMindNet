from pathlib import Path

import pytest

import magicmindnet as ai


def test_export_chatbot_rejects_unknown_format(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16)
    path = tmp_path / "out.mmn"
    with pytest.raises(ValueError, match="Unknown format"):
        ai.export(bot, "onnx", str(path))


def test_export_classifier_rejects_unknown_format(tmp_path: Path):
    clf = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=1)
    path = tmp_path / "out.mmn"
    with pytest.raises(ValueError, match="Unknown format"):
        ai.export_classifier(clf, "bin", str(path))


def test_import_model_rejects_unknown_format(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    with pytest.raises(ValueError, match="Unknown format"):
        ai.import_model("onnx", [str(path)])
