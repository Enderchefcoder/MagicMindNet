"""Vision Chatbot KV-cache generation parity."""

import magicmindnet as ai


def test_vision_kv_cache_matches_full_forward():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, vision=True, seed=12)
    patch = ai.vision_rgb_patch_from_text("a photo")
    kv = bot.generate_tokens(
        "describe",
        max_new_tokens=6,
        temperature=0.0,
        use_kv_cache=True,
        image_patch=patch,
    )
    full = bot.generate_tokens(
        "describe",
        max_new_tokens=6,
        temperature=0.0,
        use_kv_cache=False,
        image_patch=patch,
    )
    assert kv == full


def test_vision_sliding_window_past_max_ctx():
    bot = ai.Chatbot(
        vocab_size=64,
        n_layer=1,
        d_model=16,
        vision=True,
        seed=44,
        use_rope=True,
        max_seq_len=4,
    )
    patch = ai.vision_rgb_patch_from_text("scene")
    kv = bot.generate_tokens(
        "abcd",
        max_new_tokens=10,
        temperature=0.0,
        use_kv_cache=True,
        image_patch=patch,
    )
    full = bot.generate_tokens(
        "abcd",
        max_new_tokens=10,
        temperature=0.0,
        use_kv_cache=False,
        image_patch=patch,
    )
    assert len(kv) == 10
    assert kv == full


def test_unigram_prune_reduces_piece_count():
    enc = ai.UnigramEncoder.train(["repeat " * 20], vocab_size=400)
    before = enc.piece_count
    enc.prune_pieces_below_logprob(-4.0)
    assert enc.piece_count <= before
    assert enc.piece_count >= 256
