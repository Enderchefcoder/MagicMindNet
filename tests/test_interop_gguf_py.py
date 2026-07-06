"""GGUF export/import — from-scratch container, no llama.cpp anywhere."""

from pathlib import Path

import pytest

import magicmindnet as ai
from conftest import checkpoint_tensor_first_f32


def make_bot(seed=7):
    return ai.Chatbot(vocab_size=64, n_layer=2, d_model=16, seed=seed)


def test_gguf_roundtrip_preserves_shape(tmp_path):
    path = str(tmp_path / "bot.gguf")
    bot = make_bot()
    bot.save(path, format="gguf")
    loaded = ai.Chatbot.load(path)
    assert loaded.vocab_size == bot.vocab_size
    assert loaded.n_layer == bot.n_layer
    assert loaded.d_model == bot.d_model


def test_gguf_magic_bytes(tmp_path):
    path = tmp_path / "bot.gguf"
    make_bot().save(str(path), format="gguf")
    assert path.read_bytes()[:4] == b"GGUF"


def test_gguf_universal_load_returns_chatbot(tmp_path):
    path = str(tmp_path / "bot.gguf")
    make_bot().save(path, format="gguf")
    loaded = ai.load(path)
    assert isinstance(loaded, ai.Chatbot)


def test_gguf_import_model_function(tmp_path):
    path = str(tmp_path / "bot.gguf")
    make_bot().save(path, format="gguf")
    loaded = ai.import_model("gguf", [path])
    assert loaded.vocab_size == 64


def test_gguf_loaded_model_generates(tmp_path):
    path = str(tmp_path / "bot.gguf")
    make_bot().save(path, format="gguf")
    loaded = ai.load(path)
    reply = loaded.chat("hello")
    assert isinstance(reply, str)


def test_gguf_q8_0_roundtrip_close(tmp_path):
    path = str(tmp_path / "bot_q8.gguf")
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=32, seed=3)
    bot.save(path, format="gguf-q8_0")
    loaded = ai.load(path)
    ref = str(tmp_path / "ref.mmn")
    quant = str(tmp_path / "quant.mmn")
    bot.save(ref)
    loaded.save(quant)
    a = checkpoint_tensor_first_f32(Path(ref), "embed")
    b = checkpoint_tensor_first_f32(Path(quant), "embed")
    assert abs(a - b) < 0.02


def test_gguf_roundtrip_preserves_loss(tmp_path):
    data = ai.DatasetQA(data=[{"input": "hi", "output": "hello"}])
    bot = make_bot(seed=11)
    before = bot.compute_mean_loss(data)
    path = str(tmp_path / "bot.gguf")
    bot.save(path, format="gguf")
    loaded = ai.load(path)
    after = loaded.compute_mean_loss(data)
    assert abs(before - after) < 1e-4


def test_gguf_vision_export_rejected(tmp_path):
    bot = ai.Chatbot(vision=True, vocab_size=64, n_layer=1, d_model=16)
    with pytest.raises((RuntimeError, ValueError), match="vision"):
        bot.save(str(tmp_path / "v.gguf"), format="gguf")


def test_export_unknown_format_lists_gguf(tmp_path):
    bot = make_bot()
    with pytest.raises(ValueError, match="gguf"):
        bot.save(str(tmp_path / "x.bad"), format="not-a-format")
