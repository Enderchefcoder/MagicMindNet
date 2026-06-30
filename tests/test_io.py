"""Export, import, quantize, and bin IO."""

from pathlib import Path

import pytest

import magicmindnet as ai


def test_export_import_embed_roundtrip(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    path = tmp_path / "weights.mmn"
    ai.export(bot, "safetensors", str(path))
    assert path.exists() and path.stat().st_size > 0
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.parameters == bot.parameters
    assert loaded.layer_size == bot.layer_size


def test_import_preserves_layer_size(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    path = tmp_path / "w.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.layer_size == bot.layer_size


def test_quantize_modes():
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    ai.quantize(bot, "int8")
    with pytest.raises(RuntimeError, match="Unknown quant"):
        ai.quantize(bot, "fp16")
