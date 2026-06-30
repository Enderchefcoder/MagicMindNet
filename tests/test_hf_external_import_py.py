"""External Hugging Face checkpoint layout adapters."""

from pathlib import Path

import pytest

import magicmindnet as ai


def test_hf_safetensors_preserves_custom_ffn_dim(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "custom_ffn.safetensors"
    ai.export(bot, "hf-safetensors", str(path))
    loaded = ai.import_model("hf-safetensors", [str(path)])
    assert loaded.d_model == bot.d_model
    assert loaded.compute_loss("a", "b") == pytest.approx(bot.compute_loss("a", "b"))
