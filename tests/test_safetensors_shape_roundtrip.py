from pathlib import Path

import magicmindnet as ai


def test_safetensors_roundtrip_preserves_shape_getters(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=150, n_layer=2, d_model=40, seed=11)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.vocab_size == 150
    assert loaded.n_layer == 2
    assert loaded.d_model == 40
