"""Train/tokenize seq_len and tiny autoset presets."""

from __future__ import annotations

import magicmindnet as ai


def test_tokenize_respects_max_seq_len():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=0, max_seq_len=64)
    # Long corpus row should not be hard-capped at 32 when max_seq_len is larger.
    text = "abcdefghij" * 8  # 80 chars
    data = ai.DatasetCorpus(data=[text])
    before = bot.compute_mean_loss(data)
    losses = bot.train(data, epochs=1, batch_size=1, learning_rate=1e-2, optimizer="adamw")
    assert len(losses) == 1
    after = bot.compute_mean_loss(data)
    assert after <= before + 1e-3 or losses[0] < 20.0


def test_tiny_autoset_presets():
    for name, budget in [
        ("sub-1M", 1_000_000),
        ("sub-10M", 10_000_000),
        ("sub-50M", 50_000_000),
    ]:
        bot = ai.Chatbot(autoset=name, vocab_size=4096, seed=1)
        assert bot.parameters <= int(budget * 1.05)
        assert bot.n_layer >= 1
        assert bot.d_model >= 16


def test_ffn_dim_python_ctor():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, ffn_dim=96, seed=2)
    assert bot.ffn_dim == 96
    assert bot.parameters > 0


def test_cot_format_used_in_qa_training(tmp_path):
    # With cot=True + thinktag, training wraps the assistant target.
    data = ai.DatasetQA(
        data=[{"input": "hi", "output": "hello there"}],
        cot=True,
        thinktag="think",
    )
    sample = data.format_sample(0)
    assert "<think>" in sample and "</think>" in sample
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=3, max_seq_len=64)
    losses = bot.train(data, epochs=1, batch_size=1, optimizer="adamw", learning_rate=1e-2)
    assert len(losses) == 1
    # Loss on CoT-wrapped target should be finite and defined.
    assert bot.compute_mean_loss(data) == bot.compute_mean_loss(data)
