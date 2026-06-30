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


def test_generate_tokens_returns_ids():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=2)
    ids = bot.generate_tokens("hi", max_new_tokens=5, temperature=0.0)
    assert isinstance(ids, list)
    assert len(ids) == 5


def test_generate_stop_token_ids():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=99)
    first = bot.generate_tokens("z", max_new_tokens=1)[0]
    stopped = bot.generate_tokens("z", max_new_tokens=8, stop_token_ids=[first])
    assert stopped == []


def test_generate_long_prompt_not_truncated_to_32():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=4)
    prompt = "a" * 48
    ids = bot.generate_tokens(prompt, max_new_tokens=2, temperature=0.0)
    assert len(ids) == 2


def test_sliding_window_past_learned_max_seq_len():
    bot = ai.Chatbot(
        vocab_size=64,
        n_layer=1,
        d_model=16,
        use_learned_pos_embed=True,
        max_seq_len=8,
        seed=5,
    )
    ids = bot.generate_tokens("ab", max_new_tokens=12, temperature=0.0)
    assert len(ids) == 12
    full = bot.generate_tokens(
        "ab", max_new_tokens=12, temperature=0.0, use_kv_cache=False
    )
    assert ids == full


def test_frequency_and_presence_penalty_kwargs():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=9)
    ids = bot.generate_tokens(
        "xy",
        max_new_tokens=4,
        temperature=0.0,
        frequency_penalty=0.5,
        presence_penalty=0.25,
    )
    assert len(ids) == 4

def test_rope_kv_slide_matches_full_forward():
    bot = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=16,
        use_rope=True,
        max_seq_len=8,
        seed=77,
    )
    kv = bot.generate_tokens("hi", max_new_tokens=10, temperature=0.0)
    full = bot.generate_tokens(
        "hi", max_new_tokens=10, temperature=0.0, use_kv_cache=False
    )
    assert kv == full
