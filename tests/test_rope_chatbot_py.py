"""Rotary position embedding (RoPE) on Chatbot attention."""

from pathlib import Path

import pytest

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_rope_chatbot_getters():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, use_rope=True, rope_theta=5000.0, seed=1)
    assert bot.use_rope is True
    assert bot.rope_theta == pytest.approx(5000.0)
    assert bot.use_learned_pos_embed is False


def test_rope_changes_loss_vs_sinusoidal():
    tokens_in, tokens_out = "hello", "world"
    plain = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=2)
    rope_bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, use_rope=True, seed=2)
    assert plain.compute_loss(tokens_in, tokens_out) != rope_bot.compute_loss(tokens_in, tokens_out)


def test_rope_checkpoint_roundtrip(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, use_rope=True, rope_theta=8000.0, seed=3)
    path = tmp_path / "rope.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.use_rope is True
    assert loaded.rope_theta == pytest.approx(8000.0)


def test_merge_rejects_rope_theta_mismatch():
    a = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, use_rope=True, rope_theta=10000.0, seed=1)
    b = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, use_rope=True, rope_theta=5000.0, seed=2)
    with pytest.raises(ai.ModelMismatchError):
        ai.merge(a, b)


def test_merge_rejects_rope_vs_sinusoidal():
    rope_bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, use_rope=True, seed=1)
    plain = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=2)
    with pytest.raises(ai.ModelMismatchError):
        ai.merge(rope_bot, plain)


def test_rope_trains_and_reduces_loss():
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, use_rope=True, seed=4)
    loss_before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    loss_after = bot.compute_mean_loss(ds)
    assert loss_after < loss_before
