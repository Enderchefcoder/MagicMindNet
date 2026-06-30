import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_import_safetensors_rejects_missing_vocab_size_meta_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"].pop("vocab_size", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    assert "vocab_size" in str(exc.value).lower()


def test_import_safetensors_rejects_invalid_json_py(tmp_path: Path):
    path = tmp_path / "bad.mmn"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        ai.import_model("safetensors", [str(path)])


def test_import_safetensors_rejects_empty_file_py(tmp_path: Path):
    path = tmp_path / "empty.mmn"
    path.write_text("", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        ai.import_model("safetensors", [str(path)])


def test_import_classifier_rejects_missing_head_py(tmp_path: Path):
    clf = ai.Classifier.with_labels(["a", "b"], input_dim=16, seed=1)
    path = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"].pop("head", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_classifier("safetensors", [str(path)])
    assert "head" in str(exc.value).lower()
