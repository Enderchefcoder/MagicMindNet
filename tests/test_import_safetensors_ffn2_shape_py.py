import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_import_safetensors_rejects_ffn2_shape_mismatch_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"]["blocks.0.ffn2"]["shape"] = [32, 32]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert "blocks.0.ffn2" in msg and "shape" in msg
