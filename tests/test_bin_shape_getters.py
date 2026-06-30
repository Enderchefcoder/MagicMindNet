"""`bin` import restores architecture exposed via shape getters."""

from pathlib import Path

import magicmindnet as ai


def test_bin_roundtrip_preserves_shape_getters(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=200, n_layer=3, d_model=48, vision=True, seed=2)
    path = tmp_path / "arch.bin"
    ai.export(bot, "bin", str(path))
    loaded = ai.import_model("bin", [str(path)])
    assert loaded.vocab_size == 200
    assert loaded.n_layer == 3
    assert loaded.d_model == 48
    assert loaded.has_vision is True


def test_bin_roundtrip_preserves_learned_pos_embed_meta(tmp_path: Path):
    bot = ai.Chatbot(
        vocab_size=200,
        n_layer=2,
        d_model=32,
        seed=3,
        use_learned_pos_embed=True,
        max_seq_len=48,
    )
    path = tmp_path / "learned_arch.bin"
    ai.export(bot, "bin", str(path))
    loaded = ai.import_model("bin", [str(path)])
    assert loaded.use_learned_pos_embed is True
    assert loaded.max_seq_len == 48
    assert loaded.vocab_size == 200
    assert loaded.n_layer == 2
    assert loaded.d_model == 32


def test_bin_roundtrip_vision_and_learned_pos_embed_meta(tmp_path: Path):
    bot = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=16,
        vision=True,
        seed=4,
        use_learned_pos_embed=True,
        max_seq_len=32,
    )
    path = tmp_path / "vision_learned.bin"
    ai.export(bot, "bin", str(path))
    loaded = ai.import_model("bin", [str(path)])
    assert loaded.has_vision is True
    assert loaded.use_learned_pos_embed is True
    assert loaded.max_seq_len == 32
