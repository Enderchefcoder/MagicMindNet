import json
from pathlib import Path

import pytest

import magicmindnet as ai


def _export_chatbot(tmp_path: Path) -> Path:
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    return path


def test_import_safetensors_rejects_block_shape_mismatch_py(tmp_path: Path):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"]["blocks.0.attn.q"]["shape"] = [8, 32]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert "blocks.0.attn.q" in msg and "shape" in msg


def test_import_safetensors_rejects_n_layer_meta_mismatch_py(tmp_path: Path):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"]["n_layer"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    assert "blocks.1" in str(exc.value).lower()
