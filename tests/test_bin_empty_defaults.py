from pathlib import Path

import magicmindnet as ai


def test_import_bin_empty_object_uses_documented_defaults(tmp_path: Path):
    path = tmp_path / "empty.bin"
    path.write_text("{}", encoding="utf-8")
    bot = ai.import_model("bin", [str(path)])
    assert bot.vocab_size == 32000
    assert bot.d_model == 128
    assert bot.n_layer == 4
    assert bot.has_vision is False
