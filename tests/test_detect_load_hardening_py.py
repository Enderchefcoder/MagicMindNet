"""Hardened ai.load / Chatbot.load format detection."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import magicmindnet as ai


MAGIC = {"ggml": 0x67676D6C, "ggmf": 0x67676D66, "ggjt": 0x67676A74}


def _build_legacy(container: str, version: int, tensors):
    out = bytearray(struct.pack("<I", MAGIC[container]))
    if container != "ggml":
        out += struct.pack("<I", version)
    # hparams: n_vocab, n_embd, n_mult, n_head, n_layer, n_rot, ftype
    vocab = 16
    d = 8
    out += struct.pack("<7i", vocab, d, 1, 2, 1, 4, 0)
    for i in range(vocab):
        token = f"tok{i}".encode()
        out += struct.pack("<I", len(token)) + token + struct.pack("<f", -float(i))
    for name, ne, values in tensors:
        out += struct.pack("<3I", len(ne), len(name), 0)
        for dim in ne:
            out += struct.pack("<I", dim)
        out += name.encode()
        out += b"\x00" * (-len(out) % 32)
        out += values.tobytes()
    return bytes(out), vocab, d


def test_chatbot_load_ggml_legacy(tmp_path: Path):
    vocab, d, ffn = 16, 8, 16
    tensors = [
        ("tok_embeddings.weight", [d, vocab], np.full(d * vocab, 0.5, np.float32)),
        ("output.weight", [d, vocab], np.full(d * vocab, 0.25, np.float32)),
        ("layers.0.attention.wq.weight", [d, d], np.full(d * d, 0.1, np.float32)),
        ("layers.0.attention.wk.weight", [d, d], np.full(d * d, 0.1, np.float32)),
        ("layers.0.attention.wv.weight", [d, d], np.full(d * d, 0.1, np.float32)),
        ("layers.0.attention.wo.weight", [d, d], np.full(d * d, 0.1, np.float32)),
        ("layers.0.feed_forward.w1.weight", [d, ffn], np.full(d * ffn, 0.2, np.float32)),
        ("layers.0.feed_forward.w2.weight", [ffn, d], np.full(d * ffn, 0.3, np.float32)),
        ("layers.0.feed_forward.w3.weight", [d, ffn], np.full(d * ffn, 0.4, np.float32)),
        ("layers.0.attention_norm.weight", [d], np.ones(d, np.float32)),
        ("layers.0.ffn_norm.weight", [d], np.ones(d, np.float32)),
    ]
    blob, vocab, d = _build_legacy("ggjt", 3, tensors)
    path = tmp_path / "tiny.ggjt"
    path.write_bytes(blob)
    bot = ai.Chatbot.load(str(path))
    assert bot.vocab_size == vocab
    assert bot.d_model == d
    assert isinstance(bot.chat("hi"), str)


def test_detect_rejects_random_binary_with_hint(tmp_path: Path):
    path = tmp_path / "noise.bin"
    path.write_bytes(b"\x00\x01\x02\x03not-a-checkpoint" + b"\xff" * 64)
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.load(str(path))
    msg = str(exc.value).lower()
    assert "safetensors" not in msg or "unrecognized" in msg or "hint" in msg or "supported" in msg
    assert "unrecognized" in msg or "supported" in msg or "use ai.load_arrays" in msg


def test_detect_hdf5_hints_load_arrays(tmp_path: Path):
    # Minimal HDF5 magic (signature) — should not be treated as safetensors.
    path = tmp_path / "weights.h5"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 32)
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.load(str(path))
    msg = str(exc.value).lower()
    assert "hdf5" in msg or "h5" in msg or "load_arrays" in msg or "load_h5" in msg


def test_extension_hint_on_bad_gguf(tmp_path: Path):
    path = tmp_path / "broken.gguf"
    path.write_bytes(b"NOTG" + b"\x00" * 40)
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.load(str(path))
    msg = str(exc.value).lower()
    assert "gguf" in msg


def test_universal_load_bin_stub(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "stub.bin"
    bot.save(str(path), format="bin")
    loaded = ai.load(str(path))
    assert loaded.vocab_size == 64
    assert loaded.d_model == 16
    assert loaded.n_layer == 1
