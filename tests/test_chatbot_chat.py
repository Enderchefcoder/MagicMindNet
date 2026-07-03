"""`Chatbot.chat()` — the beginner-facing generation helper."""

import pytest

import magicmindnet as ai


def make_bot():
    return ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=8)


def test_chat_returns_string():
    reply = make_bot().chat("hello", max_new_tokens=6)
    assert isinstance(reply, str)


def test_chat_greedy_is_deterministic():
    bot = make_bot()
    a = bot.chat("hello", max_new_tokens=6, temperature=0.0)
    b = bot.chat("hello", max_new_tokens=6, temperature=0.0)
    assert a == b


def test_chat_respects_stop_strings():
    bot = make_bot()
    full = bot.chat("hello", max_new_tokens=12, temperature=0.0)
    assert len(full) > 1
    stopped = bot.chat(
        "hello", max_new_tokens=12, temperature=0.0, stop_strings=[full[1]]
    )
    assert len(stopped) <= len(full)


def test_chat_matches_generate_with_same_settings():
    bot = make_bot()
    chat_out = bot.chat(
        "hi", max_new_tokens=8, temperature=0.0, top_p=0.0, repetition_penalty=1.0
    )
    gen_out = bot.generate("hi", max_new_tokens=8, temperature=0.0)
    assert chat_out == gen_out


def test_chat_rejects_two_encoders():
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=16, seed=8)
    bpe = ai.BytePairEncoder.train(["aaa bbb"] * 4, vocab_size=512, num_merges=4)
    uni = ai.UnigramEncoder.train(["aaa bbb"] * 4, vocab_size=512)
    with pytest.raises(ai.DataMismatchError, match="one of"):
        bot.chat("hi", bpe_encoder=bpe, unigram_encoder=uni)
