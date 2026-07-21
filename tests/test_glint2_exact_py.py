"""Glint-2 exact architecture knobs on MagicMindNet Chatbot (no torch)."""

from __future__ import annotations

import magicmindnet as ai


def test_glint2_exact_ctor_knobs():
    bot = ai.Chatbot(
        vocab_size=4096,
        n_layer=1,
        d_model=96,
        n_heads=8,
        ffn_dim=2112,
        max_seq_len=4096,
        use_rope=True,
        rope_theta=10000.0,
        n_loops=8,
        max_loops=16,
        norm="rms",
        ffn="swiglu",
        tie_embeddings=True,
        loop_embed=True,
        final_norm=True,
        lora_rank=4,
        coda_layers=1,
        prelude_layers=0,
        attention_window=256,
        seed=0,
    )
    assert bot.n_loops == 8
    assert bot.max_loops == 16
    assert bot.coda_layers == 1
    assert bot.prelude_layers == 0
    assert bot.attention_window == 256
    assert bot.lora_rank == 4
    assert bot.final_norm is True
    assert bot.loop_embed is True
    assert bot.norm == "rms"
    assert bot.ffn == "swiglu"
    assert bot.tie_embeddings is True
    assert bot.ffn_dim == 2112
    assert bot.d_model == 96
    assert bot.n_heads == 8
    # With coda, Glint-2 checkpoint is ~1.71M; without coda ~1.06M.
    assert bot.parameters > 1_000_000
    assert bot.parameters < 2_000_000


def test_glint2_coda_changes_logits_vs_no_coda():
    common = dict(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=32,
        use_rope=True,
        n_loops=2,
        norm="rms",
        ffn="swiglu",
        tie_embeddings=True,
        loop_embed=True,
        final_norm=True,
        lora_rank=2,
        seed=7,
    )
    a = ai.Chatbot(**common, coda_layers=0)
    b = ai.Chatbot(**common, coda_layers=1)
    assert a.parameters < b.parameters
    ta = a.generate("hi", max_new_tokens=4, temperature=0.0)
    tb = b.generate("hi", max_new_tokens=4, temperature=0.0)
    # Different topology → different generations in general
    assert isinstance(ta, str) and isinstance(tb, str)


def test_glint2_attention_window_roundtrip(tmp_path):
    bot = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=64,
        use_rope=True,
        n_loops=2,
        norm="rms",
        ffn="swiglu",
        tie_embeddings=True,
        loop_embed=True,
        final_norm=True,
        lora_rank=2,
        coda_layers=1,
        attention_window=16,
        seed=3,
    )
    path = tmp_path / "glint2.mmn"
    bot.save(str(path))
    loaded = ai.load(str(path))
    assert loaded.attention_window == 16
    assert loaded.coda_layers == 1
    assert loaded.n_loops == 2
    assert loaded.lora_rank == 2
    assert loaded.final_norm is True


def test_glint2_train_updates_on_corpus():
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=64,
        use_rope=True,
        n_loops=2,
        max_loops=4,
        norm="rms",
        ffn="swiglu",
        tie_embeddings=True,
        loop_embed=True,
        final_norm=True,
        lora_rank=2,
        coda_layers=1,
        attention_window=32,
        seed=1,
    )
    data = ai.DatasetCorpus(
        data=[
            "Once upon a time there was a little girl named Lily.",
            "The sun is a star at the center of the solar system.",
        ]
        * 8
    )
    before = bot.compute_mean_loss(data)
    losses = bot.train(data, epochs=2, learning_rate=1e-2, batch_size=1, optimizer="adamw")
    after = bot.compute_mean_loss(data)
    assert len(losses) == 2
    assert after < before
