"""KV-cache generation parity and GQA+RoPE benchmark smoke."""

import magicmindnet as ai


def test_kv_cache_matches_full_forward():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, use_rope=True, seed=21)
    a = bot.generate_tokens("hi", max_new_tokens=8, temperature=0.0, use_kv_cache=True)
    b = bot.generate_tokens("hi", max_new_tokens=8, temperature=0.0, use_kv_cache=False)
    assert a == b


def test_kv_cache_gqa_rope():
    bot = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        use_rope=True,
        seed=3,
    )
    out = bot.generate("hello", max_new_tokens=6, use_kv_cache=True)
    assert isinstance(out, str)
