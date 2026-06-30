import json
from pathlib import Path

import pytest

import magicmindnet as ai


def _export_chatbot(tmp_path: Path) -> Path:
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    return path


def test_import_safetensors_rejects_incomplete_meta_py(tmp_path: Path):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"].pop("n_layer", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    assert "n_layer" in str(exc.value).lower()


def test_import_safetensors_rejects_missing_embed_py(tmp_path: Path):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"].pop("embed", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    assert "embed" in str(exc.value).lower()


def test_import_safetensors_rejects_tensor_length_mismatch_py(tmp_path: Path):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"]["embed"]["data"] = [0, 0, 0, 0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert "length" in msg or "mismatch" in msg


def test_import_safetensors_rejects_embed_shape_mismatch_py(tmp_path: Path):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"]["d_model"] = 32
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert "embed" in msg and "shape" in msg
