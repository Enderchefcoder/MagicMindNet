from pathlib import Path

import magicmindnet as ai


def test_export_chatbot_creates_output_file(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=1)
    out = tmp_path / "nested" / "bot.mmn"
    ai.export(bot, "safetensors", str(out))
    assert out.is_file()
    assert out.stat().st_size > 0
