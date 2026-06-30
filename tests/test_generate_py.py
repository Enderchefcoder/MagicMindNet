"""Chatbot autoregressive text generation."""

from pathlib import Path

import magicmindnet as ai


def test_generate_greedy_is_deterministic():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=42)
    a = bot.generate("hi", max_new_tokens=8, temperature=0.0)
    b = bot.generate("hi", max_new_tokens=8, temperature=0.0)
    assert a == b


def test_generate_returns_string():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=1)
    out = bot.generate("hello", max_new_tokens=4)
    assert isinstance(out, str)


def test_generate_after_train_produces_output(tmp_path: Path):
    data = ai.DatasetQA(
        file="tests/fixtures/qa_valid.json",
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=3)
    ai.Train(bot, data, ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05))
    out = bot.generate("What is", max_new_tokens=12, temperature=0.0)
    assert isinstance(out, str)


def test_generate_with_bpe_encoder():
    enc = ai.BytePairEncoder.train(["hello hello", "hello world"], vocab_size=256, num_merges=8)
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=5)
    out = bot.generate("hello", max_new_tokens=6, bpe_encoder=enc)
    assert isinstance(out, str)


def test_bpe_decode_roundtrip():
    enc = ai.BytePairEncoder.train(["hello hello"], vocab_size=512, num_merges=6)
    ids = enc.encode("hello")
    decoded = enc.decode(ids)
    assert decoded == "hello"
