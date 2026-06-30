"""Multi-block (n_layer>1) chatbot safetensors IO strictness."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai
from conftest import load_checkpoint_tensors, tensor_entry_first_f32

BLOCK_SUFFIXES = [
    "attn.q",
    "attn.k",
    "attn.v",
    "attn.out",
    "ffn",
    "ffn2",
    "ln1.gamma",
    "ln1.beta",
    "ln2.gamma",
    "ln2.beta",
]

BLOCK1_TENSOR_KEYS = [f"blocks.1.{suffix}" for suffix in BLOCK_SUFFIXES]


def _export_two_layer(tmp_path: Path) -> Path:
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=16, seed=1)
    path = tmp_path / "bot2.mmn"
    ai.export(bot, "safetensors", str(path))
    return path


def test_export_two_layer_includes_block1_tensors(tmp_path: Path):
    path = _export_two_layer(tmp_path)
    tensors = json.loads(path.read_text(encoding="utf-8"))["tensors"]
    for key in BLOCK1_TENSOR_KEYS:
        assert key in tensors, f"missing exported key {key}"


@pytest.mark.parametrize("tensor_key", BLOCK1_TENSOR_KEYS)
def test_import_rejects_missing_block1_tensor_py(tmp_path: Path, tensor_key: str):
    path = _export_two_layer(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"].pop(tensor_key, None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    assert tensor_key.lower() in str(exc.value).lower()


@pytest.mark.parametrize(
    "tensor_key",
    [
        "blocks.1.attn.q",
        "blocks.1.ffn",
        "blocks.1.ln2.gamma",
    ],
)
def test_merge_two_layer_averages_block1_tensor_py(tmp_path: Path, tensor_key: str):
    a = ai.Chatbot(vocab_size=128, n_layer=2, d_model=16, seed=1)
    b = ai.Chatbot(vocab_size=128, n_layer=2, d_model=16, seed=2)
    path_a = tmp_path / "a.mmn"
    path_b = tmp_path / "b.mmn"
    path_m = tmp_path / "m.mmn"
    ai.export(a, "safetensors", str(path_a))
    ai.export(b, "safetensors", str(path_b))
    merged = ai.merge(a, b)
    ai.export(merged, "safetensors", str(path_m))
    tensors_a = load_checkpoint_tensors(path_a)
    tensors_b = load_checkpoint_tensors(path_b)
    tensors_m = load_checkpoint_tensors(path_m)
    wa = tensor_entry_first_f32(tensors_a[tensor_key])
    wb = tensor_entry_first_f32(tensors_b[tensor_key])
    wm = tensor_entry_first_f32(tensors_m[tensor_key])
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_import_two_layer_roundtrip_preserves_n_layer(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=16, seed=3)
    path = tmp_path / "rt.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.n_layer == 2
