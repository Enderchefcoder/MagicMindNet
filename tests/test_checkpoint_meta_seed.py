"""Checkpoint meta records optional init seed for reproducibility metadata."""

import json
from pathlib import Path

import magicmindnet as ai


def test_chatbot_export_meta_includes_seed(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=99)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
    assert meta.get("seed") == 99


def test_chatbot_import_restores_init_seed_getter(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=77)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.init_seed == 77


def test_chatbot_without_seed_omits_meta_seed(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32)
    assert bot.init_seed is None
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
    assert "seed" not in meta


def test_classifier_export_meta_includes_seed(tmp_path: Path):
    cls_path = tmp_path / "cls.json"
    cls_path.write_text(
        json.dumps([{"text": "a", "tag": "X"}, {"text": "b", "tag": "Y"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(cls_path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=16, seed=55)
    ckpt = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(ckpt))
    meta = json.loads(ckpt.read_text(encoding="utf-8"))["meta"]
    assert meta.get("seed") == 55
