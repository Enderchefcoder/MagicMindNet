import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_import_safetensors_rejects_missing_d_model_meta_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"].pop("d_model", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    assert "d_model" in str(exc.value).lower()
