"""Grouped-query attention (GQA) Chatbot Python API and IO."""

from pathlib import Path

import pytest

import magicmindnet as ai


def test_gqa_chatbot_exposes_head_getters():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, n_heads=4, n_kv_heads=2, seed=1)
    assert bot.n_heads == 4
    assert bot.n_kv_heads == 2


def test_gqa_default_kv_heads_equals_query_heads():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, n_heads=4, seed=1)
    assert bot.n_kv_heads == bot.n_heads


def test_gqa_has_fewer_parameters_than_mha():
    mha = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, n_heads=4, n_kv_heads=4, seed=1)
    gqa = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, n_heads=4, n_kv_heads=2, seed=1)
    assert gqa.parameters < mha.parameters


def test_gqa_hf_safetensors_roundtrip(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, n_heads=4, n_kv_heads=2, seed=3)
    path = tmp_path / "gqa.safetensors"
    ai.export(bot, "hf-safetensors", str(path))
    loaded = ai.import_model("hf-safetensors", [str(path)])
    assert loaded.n_heads == 4
    assert loaded.n_kv_heads == 2
    assert loaded.compute_loss("hi", "ho") == pytest.approx(bot.compute_loss("hi", "ho"))


def test_gqa_json_safetensors_roundtrip(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, n_heads=4, n_kv_heads=2, seed=5)
    path = tmp_path / "gqa.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.n_kv_heads == 2
    assert loaded.compute_loss("a", "b") == pytest.approx(bot.compute_loss("a", "b"))


def test_gqa_trains_and_reduces_loss():
    data = ai.DatasetQA(
        file="tests/fixtures/qa_valid.json",
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, n_heads=4, n_kv_heads=2, seed=7)
    before = bot.compute_mean_loss(data)
    ai.Train(bot, data, ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05))
    after = bot.compute_mean_loss(data)
    assert after < before
