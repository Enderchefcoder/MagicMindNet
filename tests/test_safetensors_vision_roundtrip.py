from pathlib import Path

import magicmindnet as ai


def test_safetensors_roundtrip_preserves_has_vision(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, vision=True, seed=3)
    path = tmp_path / "vision.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.has_vision is True
