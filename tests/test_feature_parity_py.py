"""Feature parity: llama.cpp sampling, Ollama stream/embed, PyTorch schedules."""

from __future__ import annotations

import math

import pytest

import magicmindnet as ai


def test_generate_typical_p_and_mirostat_kwargs():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=1)
    a = bot.generate("hi", max_new_tokens=4, temperature=0.8, typical_p=0.9)
    b = bot.generate("hi", max_new_tokens=4, temperature=0.8, mirostat=2, mirostat_tau=5.0, mirostat_eta=0.1)
    assert isinstance(a, str) and isinstance(b, str)


def test_generate_stream_returns_token_chunks():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=2)
    chunks = bot.generate_stream("ab", max_new_tokens=5, temperature=0.0)
    assert isinstance(chunks, list)
    assert len(chunks) == 5
    assert all(isinstance(c, str) for c in chunks)
    joined = "".join(chunks)
    full = bot.generate("ab", max_new_tokens=5, temperature=0.0)
    assert joined == full


def test_chatbot_embed_mean_pool():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=32, seed=3)
    v = bot.embed("hello")
    assert isinstance(v, list)
    assert len(v) == 32
    assert all(isinstance(x, float) and math.isfinite(x) for x in v)
    batch = bot.embed(["hello", "world"])
    assert len(batch) == 2
    assert len(batch[0]) == 32
    # Same text → same embedding
    assert bot.embed("hello") == v


def test_train_config_weight_decay_and_cosine_schedule():
    cfg = ai.TrainConfig(
        epochs=2,
        batch_size=1,
        learning_rate=0.05,
        weight_decay=0.1,
        lr_schedule="cosine",
        warmup_steps=1,
        optimizer="adamw",
        cuda=False,
    )
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.lr_schedule == "cosine"
    assert cfg.warmup_steps == 1
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=4)
    ds = ai.DatasetQA(data=[{"input": "a", "output": "b"}] * 4)
    before = bot.compute_mean_loss(ds)
    ai.Train(bot, ds, cfg)
    after = bot.compute_mean_loss(ds)
    assert after == after
    assert after <= before + 2.0


def test_format_chat_messages_openai_style():
    from magicmindnet.chat import format_chat_messages

    text = format_chat_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
        ]
    )
    assert "system" in text.lower() or "You are helpful" in text
    assert "Hi" in text and "Bye" in text
    assert "Hello!" in text


def test_chatbot_chat_messages_api():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=5)
    out = bot.chat_messages(
        [{"role": "user", "content": "hi"}],
        max_new_tokens=4,
        temperature=0.0,
    )
    assert isinstance(out, str)
    assert len(out) > 0


def test_train_config_rejects_bad_lr_schedule():
    with pytest.raises(ValueError):
        ai.TrainConfig(lr_schedule="nope")


def test_parity_exports():
    assert "format_chat_messages" in ai.__all__ or hasattr(ai, "format_chat_messages")
    from magicmindnet import chat as chat_mod

    assert callable(chat_mod.format_chat_messages)
