"""Optional head_dim independent of d_model // n_heads (Qwen-style)."""

from pathlib import Path

import pytest

import magicmindnet as ai


def test_chatbot_custom_head_dim_getter_and_default():
    custom = ai.Chatbot(
        vocab_size=64,
        n_layer=1,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        seed=1,
    )
    assert custom.head_dim == 8
    assert custom.n_heads == 4
    assert custom.n_kv_heads == 2
    # Default head_dim = d_model // n_heads
    default = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, n_heads=4, seed=2)
    assert default.head_dim == 4


def test_chatbot_custom_head_dim_forward_and_loss():
    bot = ai.Chatbot(
        vocab_size=64,
        n_layer=1,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        use_rope=True,
        seed=3,
    )
    loss = bot.compute_loss("hi", "ho")
    assert loss == loss  # finite
    text = bot.generate("Hi", max_new_tokens=2)
    assert isinstance(text, str)


def test_custom_head_dim_safetensors_roundtrip(tmp_path: Path):
    bot = ai.Chatbot(
        vocab_size=64,
        n_layer=1,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        seed=5,
    )
    path = tmp_path / "qwen_style.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.head_dim == 16
    assert loaded.compute_loss("a", "b") == pytest.approx(bot.compute_loss("a", "b"))


def test_synthetic_hf_like_qwen_shapes_import(tmp_path: Path):
    """Import via export of a Qwen-shaped Chatbot (no Hub download)."""
    bot = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,  # != 32//4
        use_rope=True,
        seed=7,
    )
    path = tmp_path / "qwen_style.safetensors"
    ai.export(bot, "hf-safetensors", str(path))
    loaded = ai.import_model("hf-safetensors", [str(path)])
    assert loaded.d_model == 32
    assert loaded.n_heads == 4
    assert loaded.n_kv_heads == 2
    assert loaded.head_dim == 16
