"""TFLite + legacy GGML/GGMF/GGJT readers (newest and oldest ecosystem files).

TFLite is validated against models converted by the installed TensorFlow;
legacy GGML files are built by an independent struct-based writer that
follows the historical llama.cpp spec.
"""

import struct

import numpy as np
import pytest

import magicmindnet as ai

FIXTURE = "tests/fixtures/simple.tflite"


def test_tflite_fixture_reads_known_weights():
    arrays = ai.load_tflite(FIXTURE)
    flat = sorted(v for _, values in arrays.items() for row in _rows(values) for v in row)
    # kernel 0.1..0.6 + bias [0.5, -0.5]
    np.testing.assert_allclose(
        flat, sorted([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, -0.5]), atol=1e-6
    )


def _rows(values):
    if values and isinstance(values[0], list):
        return values
    return [values]


def test_load_arrays_detects_tflite():
    assert ai.detect_arrays_format(FIXTURE) == "tflite"
    arrays = ai.load_arrays(FIXTURE)
    assert len(arrays) == 2


def test_tflite_live_conversion_matches(tmp_path):
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(3, name="d")]
    )
    kernel = np.arange(12, dtype=np.float32).reshape(4, 3) / 8.0
    bias = np.array([1.0, -1.0, 0.25], dtype=np.float32)
    model.get_layer("d").set_weights([kernel, bias])
    blob = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    path = tmp_path / "live.tflite"
    path.write_bytes(blob)
    arrays = ai.load_tflite(str(path))
    all_values = [np.array(v, dtype=np.float32).ravel() for v in arrays.values()]
    assert any(
        a.size == 12 and np.allclose(np.sort(a), np.sort(kernel.ravel()))
        for a in all_values
    ), "kernel not found"
    assert any(
        a.size == 3 and np.allclose(np.sort(a), np.sort(bias)) for a in all_values
    ), "bias not found"


def test_tflite_quantized_weights_dequantize(tmp_path):
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(8,)), tf.keras.layers.Dense(8, name="d")]
    )
    rng = np.random.default_rng(0)
    kernel = rng.standard_normal((8, 8)).astype(np.float32)
    bias = np.zeros(8, dtype=np.float32)
    model.get_layer("d").set_weights([kernel, bias])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range int8
    blob = converter.convert()
    path = tmp_path / "quant.tflite"
    path.write_bytes(blob)
    arrays = ai.load_tflite(str(path))
    candidates = [
        np.array(v, dtype=np.float32).ravel()
        for v in arrays.values()
        if np.array(v, dtype=np.float32).size == 64
    ]
    assert candidates, f"no 64-element tensor found: {list(arrays)}"
    best = min(
        float(np.abs(np.sort(c) - np.sort(kernel.ravel())).max()) for c in candidates
    )
    assert best < 0.05, f"dequantized kernel error too large: {best}"


# --- legacy GGML builders (independent of the Rust reader) -----------------

MAGIC = {"ggml": 0x67676D6C, "ggmf": 0x67676D66, "ggjt": 0x67676A74}


def build_legacy(container, version, tensors, scores=True):
    """Write a legacy file per the historical llama.cpp layout."""
    out = bytearray(struct.pack("<I", MAGIC[container]))
    if container != "ggml":
        out += struct.pack("<I", version)
    out += struct.pack("<7i", 2, 8, 1, 1, 1, 2, 0)  # hparams, n_vocab=2
    for token in [b"<s>", b"hi"]:
        out += struct.pack("<I", len(token)) + token
        if container != "ggml" and scores:
            out += struct.pack("<f", -1.5)
    for name, ne, ftype, data in tensors:
        out += struct.pack("<3I", len(ne), len(name), ftype)
        for d in ne:
            out += struct.pack("<I", d)
        out += name.encode()
        if container == "ggjt":
            out += b"\x00" * (-len(out) % 32)
        out += data
    return bytes(out)


def test_unversioned_ggml_loads(tmp_path):
    values = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
    blob = build_legacy(
        "ggml", 0, [("tok_embeddings.weight", [2, 2], 0, values.tobytes())]
    )
    path = tmp_path / "old.ggml"
    path.write_bytes(blob)
    loaded = ai.load_ggml_legacy(str(path))
    assert loaded["container"] == "ggml"
    assert loaded["hparams"]["n_embd"] == 8
    assert loaded["vocab"][0] == (b"<s>", 0.0)
    assert loaded["tensors"]["tok_embeddings.weight"] == [[1.0, -2.0], [3.0, 4.0]]


def test_ggjt_v1_legacy_q4_0_layout(tmp_path):
    # d=0.5; first byte packs q0=10 (low) and q1=6 (high): 1.0 and -1.0.
    block = struct.pack("<f", 0.5) + bytes([10 | (6 << 4)]) + bytes([8 | (8 << 4)]) * 15
    blob = build_legacy("ggjt", 1, [("w", [32], 2, block)])
    path = tmp_path / "old.ggjt"
    path.write_bytes(blob)
    loaded = ai.load_ggml_legacy(str(path))
    assert loaded["container"] == "ggjt"
    w = loaded["tensors"]["w"]
    assert w[0] == 1.0 and w[1] == -1.0
    assert all(v == 0.0 for v in w[2:])


def test_ggmf_scores_and_load_arrays_detection(tmp_path):
    values = np.array([0.25, -0.75], dtype=np.float32)
    blob = build_legacy("ggmf", 1, [("norm", [2], 0, values.tobytes())])
    path = tmp_path / "old.ggmf"
    path.write_bytes(blob)
    loaded = ai.load_ggml_legacy(str(path))
    assert loaded["vocab"][1] == (b"hi", -1.5)
    assert ai.detect_arrays_format(str(path)) == "ggml-legacy"
    assert ai.load_arrays(str(path))["norm"] == [0.25, -0.75]


def test_legacy_bad_version_errors(tmp_path):
    blob = build_legacy("ggjt", 9, [])
    path = tmp_path / "bad.ggjt"
    path.write_bytes(blob)
    with pytest.raises((ValueError, RuntimeError), match="version"):
        ai.load_ggml_legacy(str(path))


def test_legacy_llama_model_loads_as_chatbot(tmp_path):
    vocab, d, ffn = 16, 8, 16
    out = bytearray(struct.pack("<I", MAGIC["ggjt"]))
    out += struct.pack("<I", 3)
    out += struct.pack("<7i", vocab, d, 1, 2, 1, 4, 0)
    for i in range(vocab):
        token = f"tok{i}".encode()
        out += struct.pack("<I", len(token)) + token + struct.pack("<f", -float(i))
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
    for name, ne, values in tensors:
        out += struct.pack("<3I", len(ne), len(name), 0)
        for dim in ne:
            out += struct.pack("<I", dim)
        out += name.encode()
        out += b"\x00" * (-len(out) % 32)
        out += values.tobytes()
    path = tmp_path / "tiny.ggjt"
    path.write_bytes(bytes(out))
    bot = ai.load(str(path))
    assert bot.vocab_size == vocab
    assert bot.d_model == d
    assert bot.n_layer == 1
    reply = bot.chat("hello")
    assert isinstance(reply, str)


def test_gguf_v1_file_loads(tmp_path):
    # Hand-built GGUF v1: u32 counts/lengths/dims (oldest GGUF revision).
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 1)  # version
    out += struct.pack("<I", 1)  # tensor count (u32)
    out += struct.pack("<I", 1)  # kv count (u32)
    key = b"general.name"
    out += struct.pack("<I", len(key)) + key
    out += struct.pack("<I", 8)  # string type
    out += struct.pack("<I", 3) + b"old"
    name = b"w"
    out += struct.pack("<I", len(name)) + name
    out += struct.pack("<I", 1)  # n_dims
    out += struct.pack("<I", 4)  # ne[0] (u32)
    out += struct.pack("<I", 0)  # type F32
    out += struct.pack("<Q", 0)  # offset
    out += b"\x00" * (-len(out) % 32)
    out += np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
    path = tmp_path / "v1.gguf"
    path.write_bytes(bytes(out))
    info = ai.gguf_info(str(path))
    assert info["version"] == 1
    arrays = ai.load_gguf_arrays(str(path))
    assert arrays["w"] == [1.0, 2.0, 3.0, 4.0]
