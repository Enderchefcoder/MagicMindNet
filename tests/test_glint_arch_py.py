"""Glint-like Chatbot architecture: n_loops, RMSNorm, SwiGLU, tied embeddings."""

from __future__ import annotations

import magicmindnet as ai


def test_glint_like_tiny_construct_train_generate_roundtrip(tmp_path):
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        n_loops=4,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        ffn="swiglu",
        norm="rms",
        use_rope=True,
        tie_embeddings=True,
        loop_embed=True,
        seed=0,
        max_seq_len=64,
    )
    assert bot.n_loops == 4
    assert bot.norm == "rms"
    assert bot.ffn == "swiglu"
    assert bot.tie_embeddings is True
    assert bot.loop_embed is True
    assert bot.ffn_dim == 64
    assert bot.use_rope is True

    data = ai.DatasetCorpus(data=["hello world from glint tiny lm", "another short corpus line"])
    losses = bot.train(data, epochs=1, learning_rate=0.01, verbose=False)
    assert len(losses) == 1
    assert losses[0] == losses[0]  # finite

    text = bot.chat("hi", max_new_tokens=8)
    assert isinstance(text, str)
    assert len(text) > 0

    path = tmp_path / "glint_tiny.safetensors"
    bot.save(str(path))
    loaded = ai.load(str(path))
    assert loaded.n_loops == 4
    assert loaded.norm == "rms"
    assert loaded.ffn == "swiglu"
    assert loaded.tie_embeddings is True
    assert loaded.loop_embed is True
    assert loaded.ffn_dim == 64


def test_glint_defaults_match_classic_chatbot():
    classic = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=1)
    assert classic.n_loops == 1
    assert classic.norm == "layer"
    assert classic.ffn == "gelu"
    assert classic.tie_embeddings is False
    assert classic.loop_embed is False
    assert classic.final_norm is False
    assert classic.lora_rank == 0
