"""Hugging Face binary safetensors interchange."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_hf_safetensors_roundtrip(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, vision=True, seed=4)
    path = tmp_path / "model.safetensors"
    ai.export(bot, "hf-safetensors", str(path))
    raw = path.read_bytes()
    assert not raw.startswith(b"{")
    loaded = ai.import_model("hf-safetensors", [str(path)])
    assert loaded.has_vision == bot.has_vision
    assert loaded.compute_loss("a", "b") == pytest.approx(bot.compute_loss("a", "b"))


def test_import_safetensors_auto_detects_hf_binary(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=2)
    path = tmp_path / "hf_auto.safetensors"
    ai.export(bot, "hf-safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.compute_loss("x", "y") == pytest.approx(bot.compute_loss("x", "y"))


def test_json_safetensors_still_works(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16)
    path = tmp_path / "json.mmn"
    ai.export(bot, "safetensors", str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "mmn-safetensors-v1"
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.compute_loss("hi", "yo") == pytest.approx(bot.compute_loss("hi", "yo"))
